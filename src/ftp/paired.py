"""Paired (stacked-weight) modules: two architecturally identical models fused
into ONE forward pass.

The two aux models of a DD pair are the same architecture with different
weights, and at decode batch sizes each one badly underutilizes the GPU —
running them sequentially doubles latency without doubling useful work.
``fuse_pair`` stacks every weight tensor along a new leading model dim
``[2, ...]`` and swaps each parameterized module for a paired version:

  * ``PairedLinear``    — one ``torch.bmm`` (cuBLAS strided-batched GEMM)
    computes both models' projections in a single kernel;
  * ``PairedEmbedding`` — one flat lookup with a per-model vocab offset;
  * ``PairedRMSNorm``   — stock RMSNorm math with a ``[2, 1, d]`` weight.

Everything without parameters (attention/SDPA, RoPE, activations, the HF
forward itself) runs untouched at the folded batch size. The batch layout is
PLANE-MAJOR: rows ``[0, B/2)`` belong to model 0, rows ``[B/2, B)`` to model 1,
so the fold inside ``PairedLinear`` is a free ``reshape`` view — no data
movement lands inside a captured CUDA graph. Weights are pre-transposed to bmm
orientation ``[2, in, out]`` and made contiguous at build time for the same
reason.

``fuse_pair`` mutates ``model0`` (the skeleton) in place and returns it; the
caller moves it to the target device/dtype exactly like a single model.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

# Config keys irrelevant to architectural equality.
_CONFIG_IGNORE = {
    "_name_or_path",
    "transformers_version",
    "_commit_hash",
    "torch_dtype",
    "dtype",
    "_attn_implementation",
    "_attn_implementation_autoset",
    "_attn_implementation_internal",
}


def check_fusable(cfg0, cfg1) -> tuple[bool, str]:
    """Can two model configs be weight-stacked into one fused forward?

    Returns ``(ok, reason)``; ``reason`` names the first blocking difference.
    Cheap (config-only) — callers use it with ``AutoConfig`` before loading
    any weights.
    """
    for cfg in (cfg0, cfg1):
        if getattr(cfg, "tie_word_embeddings", False):
            return False, (
                "tie_word_embeddings=True (stacked embedding and lm_head cannot share one tensor)"
            )
    d0 = {k: v for k, v in cfg0.to_dict().items() if k not in _CONFIG_IGNORE}
    d1 = {k: v for k, v in cfg1.to_dict().items() if k not in _CONFIG_IGNORE}
    if d0 != d1:
        for k in sorted(set(d0) | set(d1)):
            if d0.get(k) != d1.get(k):
                return False, f"config mismatch: {k} ({d0.get(k)!r} vs {d1.get(k)!r})"
    return True, ""


class PairedLinear(nn.Module):
    """Two same-shape linears as one batched matmul.

    ``weight`` is stored PRE-TRANSPOSED to bmm orientation ``[2, in, out]``
    and contiguous, so no transpose copy ever lands inside a captured graph.
    Input is plane-major folded batch: ``x[: B/2]`` = model 0.
    """

    def __init__(self, lin0: nn.Linear, lin1: nn.Linear) -> None:
        super().__init__()
        if lin0.weight.shape != lin1.weight.shape:
            raise ValueError("PairedLinear requires same-shape linears")
        self.in_features = lin0.in_features
        self.out_features = lin0.out_features
        w = torch.stack([lin0.weight.detach().t(), lin1.weight.detach().t()])
        self.weight = nn.Parameter(w.contiguous(), requires_grad=False)
        if (lin0.bias is None) != (lin1.bias is None):
            raise ValueError("PairedLinear requires both-or-neither bias")
        if lin0.bias is not None:
            b = torch.stack([lin0.bias.detach(), lin1.bias.detach()]).unsqueeze(1)
            self.bias = nn.Parameter(b.contiguous(), requires_grad=False)  # [2, 1, out]
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.shape
        x2 = x.reshape(2, -1, self.in_features)  # plane-major fold: free view
        if self.bias is not None:
            y = torch.baddbmm(self.bias, x2, self.weight)
        else:
            y = torch.bmm(x2, self.weight)
        return y.reshape(*s[:-1], self.out_features)


class PairedEmbedding(nn.Module):
    """Two same-shape embeddings as one flat lookup with a per-plane offset."""

    def __init__(self, emb0: nn.Embedding, emb1: nn.Embedding) -> None:
        super().__init__()
        if emb0.weight.shape != emb1.weight.shape:
            raise ValueError("PairedEmbedding requires same-shape embeddings")
        self.num_embeddings = emb0.num_embeddings
        self.embedding_dim = emb0.embedding_dim
        w = torch.stack([emb0.weight.detach(), emb1.weight.detach()])
        self.weight = nn.Parameter(w.contiguous(), requires_grad=False)  # [2, V, d]
        # Stable-address buffer (graph capture reads it in place).
        self.register_buffer(
            "plane_offset",
            torch.tensor([0, self.num_embeddings], dtype=torch.long).view(2, 1, 1),
            persistent=False,
        )

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        half = ids.shape[0] // 2
        ids2 = (ids.reshape(2, half, -1) + self.plane_offset).reshape(ids.shape)
        flat_w = self.weight.reshape(2 * self.num_embeddings, self.embedding_dim)
        return F.embedding(ids2, flat_w)


class PairedLookupRotary(nn.Module):
    """Two table-lookup rotary embeddings (``cos_cached``/``sin_cached`` caches
    indexed by position) as one plane-major lookup.

    Exists for checkpoint pairs whose PRECOMPUTED rope buffers differ — e.g.
    the FlexLlama flexdoc cooldowns, whose local-rope caches are per-run
    trained artifacts that each model adapted to (running plane 1 with plane
    0's cache is catastrophic, not approximate). Analytic rotaries
    (``inv_freq``-based) never land here: their buffers are config-derived and
    the fuse-time equality check keeps covering them.

    Rows are plane-major like every paired module; position lookups for rows
    ``[B/2, B)`` are offset into the second cache. Requires an even batch —
    a fused model cannot answer "which plane?" for a single unpaired row.
    """

    def __init__(self, r0: nn.Module, r1: nn.Module) -> None:
        super().__init__()
        if r0.cos_cached.shape != r1.cos_cached.shape:
            raise ValueError("PairedLookupRotary requires same-shape rope caches")
        self.cache_len = r0.cos_cached.shape[0]
        self.register_buffer(
            "cos_cached", torch.cat([r0.cos_cached.detach(), r1.cos_cached.detach()]),
            persistent=False,
        )
        self.register_buffer(
            "sin_cached", torch.cat([r0.sin_cached.detach(), r1.sin_cached.detach()]),
            persistent=False,
        )

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        B = position_ids.shape[0]
        if B % 2:
            raise ValueError(
                "PairedLookupRotary needs plane-major even-batch position_ids "
                f"(got batch {B}); pass explicit per-row position_ids"
            )
        # Bounds check is a host sync (.max() -> .item()) — illegal while a CUDA
        # graph is capturing. The graphed decode path pre-validates positions
        # (engine window <= cache_len), so skipping it under capture is safe.
        if (
            position_ids.numel()
            and not (position_ids.is_cuda and torch.cuda.is_current_stream_capturing())
            and int(position_ids.max()) >= self.cache_len
        ):
            raise ValueError(
                f"position id {int(position_ids.max())} >= rope cache_len {self.cache_len}"
            )
        half = B // 2
        idx = torch.cat([position_ids[:half], position_ids[half:] + self.cache_len])
        return self.cos_cached[idx].to(x.dtype), self.sin_cached[idx].to(x.dtype)


def _is_lookup_rotary(m: nn.Module) -> bool:
    if isinstance(m, PairedLookupRotary):
        return False
    return (
        isinstance(getattr(m, "cos_cached", None), torch.Tensor)
        and isinstance(getattr(m, "sin_cached", None), torch.Tensor)
        and next(m.parameters(), None) is None
    )


class PairedRMSNorm(nn.Module):
    """Two RMSNorms (stock HF Llama math) with a stacked ``[2, 1, d]`` weight."""

    def __init__(self, n0: nn.Module, n1: nn.Module) -> None:
        super().__init__()
        if n0.weight.shape != n1.weight.shape:
            raise ValueError("PairedRMSNorm requires same-shape norms")
        if n0.variance_epsilon != n1.variance_epsilon:
            raise ValueError("PairedRMSNorm requires equal variance_epsilon")
        self.variance_epsilon = n0.variance_epsilon
        w = torch.stack([n0.weight.detach(), n1.weight.detach()]).unsqueeze(1)
        self.weight = nn.Parameter(w.contiguous(), requires_grad=False)  # [2, 1, d]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        h = hidden_states.to(torch.float32)
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        h = h.to(input_dtype)
        s = h.shape
        return (h.reshape(2, -1, s[-1]) * self.weight).reshape(s)


def _is_rmsnorm(m: nn.Module) -> bool:
    if isinstance(m, (PairedLinear, PairedEmbedding, PairedRMSNorm, PairedLookupRotary)):
        return False
    w = getattr(m, "weight", None)
    return (
        type(m).__name__.endswith("RMSNorm")
        and isinstance(w, torch.Tensor)
        and w.ndim == 1
        and hasattr(m, "variance_epsilon")
    )


def _set_submodule(root: nn.Module, dotted: str, new: nn.Module) -> None:
    parts = dotted.split(".")
    parent = root.get_submodule(".".join(parts[:-1])) if len(parts) > 1 else root
    setattr(parent, parts[-1], new)


def fuse_pair(model0, model1, *, free_source: bool = True):
    """Fuse two architecturally identical models into one stacked-weight model.

    ``model0`` is the skeleton: every ``nn.Linear`` / ``nn.Embedding`` /
    RMSNorm in it is replaced by a paired module stacking its weights with the
    same-named module of ``model1``. Plane 0 = ``model0``, plane 1 =
    ``model1``; callers feed plane-major folded batches (row count must be
    even, first half = plane 0). Returns ``model0`` (mutated in place, on its
    current device/dtype — move/cast it afterwards like any single model).

    ``free_source=True`` drops both source modules' parameters as it goes
    (``model1`` is unusable afterwards); pass ``False`` when the sources must
    survive (tests with shared fixtures).
    """
    ok, why = check_fusable(model0.config, model1.config)
    if not ok:
        if "tie_word_embeddings" in why:
            raise NotImplementedError(f"cannot fuse: {why}")
        raise ValueError(f"cannot fuse: {why}")

    shapes0 = {n: tuple(p.shape) for n, p in model0.named_parameters()}
    shapes1 = {n: tuple(p.shape) for n, p in model1.named_parameters()}
    if shapes0 != shapes1:
        diff = sorted(set(shapes0.items()) ^ set(shapes1.items()))
        raise ValueError(f"cannot fuse: parameter name/shape mismatch: {diff[:4]}")
    expected_numel = sum(p.numel() for p in model0.parameters()) + sum(
        p.numel() for p in model1.parameters()
    )

    # Lookup rotaries (precomputed cos/sin caches) are PER-CHECKPOINT training
    # artifacts and may legitimately differ between the pair — they get their
    # own paired module below, so exempt their buffers from the equality check.
    rotary_prefixes = tuple(
        f"{name}." if name else ""
        for name, m in model0.named_modules()
        if _is_lookup_rotary(m)
    )

    # Remaining parameter-free buffers (e.g. RoPE inv_freq) are config-derived
    # and must be identical — plane 1 will silently run with plane 0's buffers.
    bufs1 = dict(model1.named_buffers())
    for bname, b0 in model0.named_buffers():
        if any(bname.startswith(p) for p in rotary_prefixes):
            continue
        b1 = bufs1.get(bname)
        if b1 is None or not torch.equal(b0, b1):
            raise ValueError(f"cannot fuse: buffer {bname!r} differs between the models")

    targets: list[tuple[str, nn.Module]] = []
    for name, m in model0.named_modules():
        if isinstance(m, (nn.Linear, nn.Embedding)) or _is_rmsnorm(m) or _is_lookup_rotary(m):
            targets.append((name, m))
    for name, m0 in targets:
        m1 = model1.get_submodule(name)
        # Lookup rotaries are duck-matched, not type-matched: trust_remote_code
        # loads each checkpoint's module file as its OWN dynamic class, so the
        # same source class compares unequal across the pair.
        if type(m0) is not type(m1) and not (_is_lookup_rotary(m0) and _is_lookup_rotary(m1)):
            raise ValueError(f"cannot fuse: module {name!r} type mismatch")
        if isinstance(m0, nn.Linear):
            paired: nn.Module = PairedLinear(m0, m1)
        elif isinstance(m0, nn.Embedding):
            paired = PairedEmbedding(m0, m1)
        elif _is_lookup_rotary(m0):
            paired = PairedLookupRotary(m0, m1)
        else:
            paired = PairedRMSNorm(m0, m1)
        _set_submodule(model0, name, paired)
        if free_source:
            m0._parameters.clear()
            m1._parameters.clear()

    # Coverage hard-asserts: a parameterized module the walk didn't recognize
    # would silently run plane 0's weights for both planes.
    for name, m in model0.named_modules():
        if isinstance(m, (nn.Linear, nn.Embedding)) or _is_rmsnorm(m) or _is_lookup_rotary(m):
            raise AssertionError(f"fuse_pair left an unfused module: {name!r}")
        owns_params = any(p is not None for p in m._parameters.values())
        if owns_params and not isinstance(m, (PairedLinear, PairedEmbedding, PairedRMSNorm)):  # noqa: E501 — PairedLookupRotary owns only buffers
            raise AssertionError(
                f"fuse_pair does not know how to stack module {name!r} ({type(m).__name__})"
            )
    fused_numel = sum(p.numel() for p in model0.parameters())
    if fused_numel != expected_numel:
        raise AssertionError(
            f"fused parameter count {fused_numel} != sum of sources {expected_numel}"
        )

    name0 = model0.config._name_or_path or type(model0).__name__
    name1 = model1.config._name_or_path or type(model1).__name__
    model0.config._name_or_path = f"fused({name0}+{name1})"
    return model0.eval()
