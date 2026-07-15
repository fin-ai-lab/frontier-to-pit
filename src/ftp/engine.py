"""Paged-KV batched aux engine for Divergence Decoding.

Why paged
---------
The original engine (see git history: a window-wide ``StaticCache`` slot per
request + one padded full-batch CUDA graph per step) had memory AND per-step
cost scaling with ``window x prewarmed_batch`` no matter what the requests
actually held (measured on 2xH100: 0.47 ms/row/step at window 2048 whether a
row contained 256 or 2048 tokens, and the whole prewarmed batch paid even when
13 rows were active). That design could not host the 32K-context aux
generation: 32K x 115 KB = 3.8 GB per slot per plane. This engine replaced it
outright — 3.4x faster at full window, fill-proportional below it, same
public API.

Design
------
KV lives in 16-token pages allocated on demand from per-layer-GROUP pools
(``[max_pages, 2, page_size, n_kv, head_dim]``, flashinfer NHD layout):

  * memory     = actual tokens cached (+ < 1 page per row of slack);
  * attention  = each row reads its ACTUAL length — flashinfer
    ``BatchDecodeWithPagedKVCacheWrapper`` on CUDA, a vectorized gather+SDPA
    fallback elsewhere (CPU tests, missing flashinfer);
  * step cost  = proportional to the rows active THIS step (no padded batch).

Hybrid (sliding + full) attention — the FlexLlama 32K aux
---------------------------------------------------------
``config.layer_types`` splits the layers into two page-pool groups. FULL
layers retain every token. SLIDING layers (per-layer window
``config.sliding_window``) keep a bounded page RING per row: pages that fall
entirely behind the window are freed, so 23 sliding layers cost ~``window_sw``
tokens each regardless of context length — this is what makes a 32K-context
3B pair affordable (~40 KB/context-token for the pair instead of ~230 KB).
Decode runs two flashinfer plans (full; sliding with ``window_left``); the
fallback masks each group to its own bound. Models without ``layer_types``
have one full group and behave exactly as before.

Decode runs the stock HF forward with a custom paged ``Cache`` (per-row tail
writes) and a registered ``"dd_paged"`` attention that reads the page pools
directly. Prefill is the SAME stock forward: the cache adapter in prefill
mode scatters each layer's K/V into fresh pages as it streams by and returns
them unchanged for the in-forward attention; attention masks are delegated to
the stock sdpa mask builders, so per-layer-type causal/sliding masks
(FlexLlama routes them itself) stay exact.

Decode host cost: staged, then CUDA-graph replayed
--------------------------------------------------
The eager decode forward was measured at ~30 ms/step of HOST time at B=112
(R=224 fused rows, 2xH100): an ~18 ms kernel-launch storm through python
dispatch, ~5 ms of per-row python building the two flashinfer plans, ~4.5 ms
of per-row staging stores, ~2 ms of per-row logits copy-out. The decode path
therefore (a) stages each step with VECTOR ops — page-table mirrors of the
per-row page lists (torch CPU int32, one slot per row) feed one pinned [6, R]
pack and the packed plan arrays — and (b) replays the whole forward (KV
scatter, mask/rotary dicts, flashinfer runs, lm head) as ONE captured CUDA
graph per batch-size bucket, with cudagraph-mode flashinfer wrappers (fixed
buffers, re-planned each step outside the graph; pages stay paged — none of
the old engine's padded-window cost returns). Pad rows in a bucket write to a
scratch page with length 1 and their logits are discarded; the pad layout is
plane-major (each plane's half pads independently) so the fused reshape stays
valid. Growing/reset page pools invalidate captured graphs (kv tensors are
replaced); prewarm-size the pools to avoid mid-run recapture.
``DD_PAGED_GRAPHS=0`` forces the eager decode (same math, A/B and debugging);
non-CUDA / no-flashinfer builds always run it.

Window semantics
----------------
``window`` is the aux context BUDGET (<= the model's trained context). Where
the old engine slid past it with per-step evictions (keys keeping stale rotary
positions), a row here that would exceed the window is RE-PRIMED: its pages
are freed and the tail ``window * (1 - DD_REPRIME_MARGIN)`` tokens of its full
context are re-prefilled with re-based positions. With a 32K-context aux and
``window`` = 32768 >= max_model_len, this never fires.

Scope: models whose attention routes through the HF attention interface
(standard Llama-family and derivatives like FlexLlama that reuse the stock
attention blocks).

Env knobs: ``DD_PAGED_FLASHINFER=0`` forces the fallback attention;
``DD_REPRIME_MARGIN`` (default 0.25); ``DD_PREFILL_TOKEN_BUDGET`` caps tokens
per eager prefill forward (default 32768).
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, PreTrainedModel

from ftp.paired import fuse_pair

_PAGE = 16
_DEBUG = os.environ.get("DD_DEBUG", "0") == "1"


# ── HF attention/mask interface plumbing ─────────────────────────────────────


def _paged_attention_forward(
    module,
    query,
    key,
    value,
    attention_mask,
    dropout: float = 0.0,
    scaling: float | None = None,
    is_causal: bool | None = None,
    **kwargs,
):
    """``"dd_paged"`` attention: single-token decode reads the engine's page
    pools directly (the paged cache already holds this step's token); prefill
    (q_len > 1) falls back to stock SDPA over the full K/V that the cache
    adapter passed through, with the masks the model routed in."""
    eng = getattr(module, "_dd_paged_engine", None)
    ctx = eng._step_ctx if eng is not None else None
    if ctx is None or ctx.mode != "decode" or query.shape[2] != 1:
        from transformers.integrations.sdpa_attention import sdpa_attention_forward

        return sdpa_attention_forward(
            module, query, key, value, attention_mask,
            dropout=dropout, scaling=scaling, is_causal=is_causal, **kwargs,
        )
    return eng._attend_decode(module.layer_idx, query, scaling), None


_REGISTERED: dict[str, bool] = {"done": False, "masks": False}


def _register_paged_attention() -> None:
    if _REGISTERED["done"]:
        return
    from transformers.modeling_utils import AttentionInterface

    AttentionInterface.register("dd_paged", _paged_attention_forward)
    # Delegate mask building to the stock sdpa mask functions: prefill then
    # gets exactly the masks the model would build under sdpa (incl. the
    # per-layer-type causal/sliding masks FlexLlama routes itself); decode
    # masks are built against the adapter and ignored by _attend_decode.
    try:
        from transformers.masking_utils import (
            ALL_MASK_ATTENTION_FUNCTIONS,
            AttentionMaskInterface,
        )

        AttentionMaskInterface.register("dd_paged", ALL_MASK_ATTENTION_FUNCTIONS["sdpa"])
        _REGISTERED["masks"] = True
    except Exception as e:  # noqa: BLE001 — older transformers; full-attn models still exact
        print(f"[AuxBatchedEngine] sdpa mask delegation unavailable ({e})", flush=True)
    _REGISTERED["done"] = True


class _PagedLayerAdapter:
    """Per-layer cache hook. Decode: scatters the step's new K/V into each
    active row's tail (page, slot). Prefill: scatters the layer's whole
    [R, n_kv, T, hd] block into the freshly allocated pages (full layers keep
    all T tokens; sliding layers keep the 16-aligned window tail) and passes
    K/V through unchanged for the in-forward attention."""

    __slots__ = ("_eng", "_li", "is_initialized", "is_sliding")

    def __init__(self, eng: AuxBatchedEngine, li: int) -> None:
        self._eng = eng
        self._li = li
        self.is_initialized = True
        self.is_sliding = li in eng._sw_set

    def update(self, key_states, value_states, *args, **kwargs):
        eng = self._eng
        ctx = eng._step_ctx
        if ctx is None:
            raise RuntimeError("paged cache update outside an engine step")
        sliding = self.is_sliding
        kv = (eng._pool_sw if sliding else eng._pool_full).kv[self._li]
        if ctx.mode == "decode":
            wr_page = ctx.wr_page_sw if sliding else ctx.wr_page_full
            kv.select(1, 0)[wr_page, ctx.wr_slot] = key_states.squeeze(2)
            kv.select(1, 1)[wr_page, ctx.wr_slot] = value_states.squeeze(2)
            return key_states, value_states
        # Prefill: indexed scatter of each row's kept token range (full
        # layers every real token, sliding layers the window tail; pads and
        # dropped tokens have no index entries and never reach the pools).
        eng._scatter_prefill(kv, key_states, value_states, ctx, sliding)
        return key_states, value_states


class _PagedCacheAdapter:
    """Duck Cache handed to the HF forward (prefill AND decode)."""

    def __init__(self, eng: AuxBatchedEngine, n_layers: int) -> None:
        self.layers = [_PagedLayerAdapter(eng, li) for li in range(n_layers)]
        self.layer_class_to_replicate = None
        self.offloading = False
        self._eng = eng

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        return self.layers[layer_idx].update(key_states, value_states, *args, **kwargs)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        ctx = self._eng._step_ctx
        if ctx is None:
            return 0
        return 0 if ctx.mode == "prefill" else int(ctx.max_kv_len)

    def get_query_offset(self, layer_idx: int = 0) -> int:
        # Where the query sits within the kv, which the mask builders ask for from
        # transformers 5.14 on (masking_utils._preprocess_mask_arguments). Cache's own
        # default is just the past length; this is a duck cache, not a Cache subclass,
        # so it inherits nothing and has to answer for itself — same source of truth as
        # get_seq_length: 0 in prefill (no past), max_kv_len in decode.
        return self.get_seq_length(layer_idx)

    def get_max_cache_shape(self, layer_idx: int = 0) -> int:
        return self._eng._window

    def get_mask_sizes(self, cache_position, layer_idx: int = 0) -> tuple[int, int]:
        # Prefill: fresh context of T tokens, no past -> the delegated sdpa
        # mask builders produce the exact causal/sliding masks. Decode: sizes
        # only need to be plausible (masks are ignored by _attend_decode).
        ctx = self._eng._step_ctx
        if ctx is not None:
            return (ctx.T if ctx.mode == "prefill" else int(ctx.max_kv_len)), 0
        n = cache_position if isinstance(cache_position, int) else len(cache_position)
        return int(n), 0

    @property
    def is_compileable(self) -> bool:
        return False


# ── Page pools ───────────────────────────────────────────────────────────────


class _PagePool:
    """One page pool shared by a GROUP of layers (same page ids index every
    member layer's kv tensor). Grows between steps only."""

    def __init__(self, name, layer_indices, n_kv, head_dim, dtype, device) -> None:
        self.name = name
        self.layer_indices = list(layer_indices)
        self._n_kv, self._hd, self._dtype, self._device = n_kv, head_dim, dtype, device
        self.kv: dict[int, torch.Tensor] = {}
        self.free: list[int] = []
        self.n_pages = 0
        # Bumped whenever the kv tensors are REPLACED (grow/reset): captured
        # decode graphs hold raw pointers into them and must be dropped.
        self.generation = 0

    def bytes_per_page(self) -> int:  # per layer
        return 2 * _PAGE * self._n_kv * self._hd * self._dtype.itemsize

    def total_bytes(self) -> int:
        return self.n_pages * self.bytes_per_page() * len(self.layer_indices)

    def reset(self) -> None:
        self.kv, self.free, self.n_pages = {}, [], 0
        self.generation += 1

    def grow(self, n_pages_target: int, *, warn: bool = True) -> None:
        if n_pages_target <= self.n_pages:
            return
        self.generation += 1
        if self.n_pages and warn:
            print(
                f"[AuxBatchedEngine] WARNING: {self.name} page pool growing "
                f"{self.n_pages} -> {n_pages_target} pages "
                f"({n_pages_target * self.bytes_per_page() * len(self.layer_indices) / 1e9:.1f}"
                " GB); size the pool up front (prewarm/pool_gb) to avoid mid-run reallocation",
                flush=True,
            )
        old = self.kv
        self.kv = {}
        for li in self.layer_indices:
            t = torch.zeros(
                n_pages_target, 2, _PAGE, self._n_kv, self._hd,
                dtype=self._dtype, device=self._device,
            )
            if self.n_pages:
                t[: self.n_pages] = old[li]
            self.kv[li] = t
        self.free.extend(range(self.n_pages, n_pages_target))
        self.n_pages = n_pages_target

    def alloc(self, n: int, default_pages: int) -> list[int]:
        if len(self.free) < n:
            need = n - len(self.free)
            self.grow(
                max(self.n_pages * 2, self.n_pages + need) if self.n_pages
                else max(default_pages, need),
                warn=self.n_pages > 0,
            )
        out = self.free[-n:]
        del self.free[-n:]
        return out


# ── Engine ───────────────────────────────────────────────────────────────────


@dataclass
class _RowState:
    seq_len: int = 0  # tokens currently cached (restarts on re-prime)
    primed: bool = False
    pages_full: list[list[int]] = field(default_factory=list)  # per plane
    pages_sw: list[list[int]] = field(default_factory=list)  # per plane (ring)
    sw_drop: int = 0  # tokens evicted from the sliding ring (16-multiple)
    slot: int = -1  # row index into the engine's page-TABLE mirrors


@dataclass
class _StepCtx:
    """Staging for one forward (consumed by the cache adapter + attention
    hook, layer by layer)."""

    mode: str  # "decode" | "prefill"
    max_kv_len: int = 0
    # decode
    wr_page_full: torch.Tensor | None = None  # [R]
    wr_page_sw: torch.Tensor | None = None  # [R]
    wr_slot: torch.Tensor | None = None  # [R]
    kv_lens: torch.Tensor | None = None  # [R] full-group lengths
    sw_lens: torch.Tensor | None = None  # [R] sliding-table lengths
    planned: bool = False
    gather_full: torch.Tensor | None = None  # [R * n_pg_max_full]
    gather_sw: torch.Tensor | None = None
    n_pg_full: int = 0
    n_pg_sw: int = 0
    # prefill: parallel [N_kept] index tensors (device) — source position in
    # the flattened [R * T] hidden stream, destination (page, slot) in the
    # group's pool. Pads / sliding-dropped tokens simply have no entries.
    pf_src_full: torch.Tensor | None = None
    pf_pg_full: torch.Tensor | None = None
    pf_slot_full: torch.Tensor | None = None
    pf_src_sw: torch.Tensor | None = None
    pf_pg_sw: torch.Tensor | None = None
    pf_slot_sw: torch.Tensor | None = None
    T: int = 0  # PADDED group length


# Row layout of the packed per-step staging block (one pinned [6, Rb] tensor,
# ONE H2D copy per step instead of six).
_ST_TOK, _ST_POS, _ST_WRPF, _ST_WRPS, _ST_WRSLOT, _ST_SWDROP = range(6)


@dataclass
class _Staged:
    """One decode step's staging, laid out for bucket size ``Rb`` (>= R; the
    pad rows point at the scratch page with length 1). Plane-major WITH the
    bucket's stride: plane p's real rows live at [p * Rb/replicas, ... + n)."""

    n: int  # real items
    Rb: int  # bucket row count (== R when not padding)
    half: int  # plane stride = Rb // replicas
    pack: torch.Tensor  # pinned [6, Rb] int64 — see _ST_* rows
    max_kv_len: int = 0
    indptr_full: torch.Tensor | None = None  # pinned [Rb + 1] int32
    indices_full: torch.Tensor | None = None  # pinned [sum(npg)] int32 (slice)
    lpl_full: torch.Tensor | None = None  # pinned [Rb] int32
    indptr_sw: torch.Tensor | None = None
    indices_sw: torch.Tensor | None = None
    lpl_sw: torch.Tensor | None = None


@dataclass
class _DecodeGraph:
    """One captured decode forward at a fixed bucket size ``Rb``: persistent
    device inputs (the graph reads them in place), cudagraph-mode flashinfer
    wrappers (fixed-buffer plan each step, run recorded in the graph), and the
    stable-address logits output."""

    Rb: int
    pack: torch.Tensor  # device [6, Rb] int64 (views feed the forward)
    fi_full: object
    fi_sw: object | None
    graph: torch.cuda.CUDAGraph | None = None
    logits: torch.Tensor | None = None  # [Rb, 1, V] captured output ref
    generation: int = -1  # sum of pool generations at capture time


class AuxBatchedEngine:
    """Batched paged-KV inference engine for an aux LLM (or a fused pair).

    Public surface: ``register`` / ``unregister`` / ``step`` / ``step_pairs``
    / ``rewind`` / ``prewarm``; ``step`` returns ``[N, V]`` logits
    (``[2, N, V]`` for a fused pair). Decode replays captured CUDA graphs per
    batch bucket (see the module docstring; ``DD_PAGED_GRAPHS=0`` for the
    eager A/B); ``compile_model`` is accepted for signature compatibility and
    ignored. ``pool_gb`` caps the TOTAL page-pool footprint (weights
    excluded); without it the full-group pool is prewarmed to the ``batch x
    window`` worst case — fine for small windows, impossible for 32K ones, so
    long-context runs must set it.
    """

    def __init__(
        self,
        model: str | PreTrainedModel,
        device: torch.device,
        dtype: torch.dtype,
        window: int,
        compile_model: bool = False,
        model2: str | PreTrainedModel | None = None,
        pool_tokens: int | None = None,
        pool_gb: float | None = None,
    ) -> None:
        device = torch.device(device)
        if isinstance(model, str):
            name = model.rstrip("/").rsplit("/", 1)[-1]
            model = AutoModelForCausalLM.from_pretrained(model, dtype=dtype, trust_remote_code=True)
        else:
            name = model.config._name_or_path or type(model).__name__
        self._replicas = 1
        if model2 is not None:
            if isinstance(model2, str):
                name2 = model2.rstrip("/").rsplit("/", 1)[-1]
                model2 = AutoModelForCausalLM.from_pretrained(
                    model2, dtype=dtype, trust_remote_code=True
                )
            else:
                name2 = model2.config._name_or_path or type(model2).__name__
            model = fuse_pair(model, model2)
            name = f"fused({name}+{name2})"
            self._replicas = 2

        _register_paged_attention()
        model.set_attn_implementation("dd_paged")
        model = model.to(device=device, dtype=dtype).eval()
        # The dd_paged attention hook finds the engine through an attribute on
        # each attention module. Two engines built over the SAME model instance
        # (legal: identical weights as p and q) would fight over it, so
        # ownership is (re)claimed per step — see _claim_model.
        self._attn_modules = [
            m for m in model.modules()
            if hasattr(m, "layer_idx") and hasattr(m, "num_key_value_groups")
        ]
        self._model = model
        self._text_cfg = model.config.get_text_config()
        self._vocab_size: int = self._text_cfg.vocab_size
        self._n_layers: int = self._text_cfg.num_hidden_layers
        n_heads = self._text_cfg.num_attention_heads
        self._n_kv: int = getattr(self._text_cfg, "num_key_value_heads", n_heads)
        self._head_dim: int = getattr(
            self._text_cfg, "head_dim", self._text_cfg.hidden_size // n_heads
        )
        self._sm_scale = self._head_dim ** -0.5

        # Layer groups: hybrid (FlexLlama-style) configs mark sliding layers in
        # layer_types; everything else is one full group.
        layer_types = list(getattr(self._text_cfg, "layer_types", None) or [])
        if layer_types and "sliding_attention" in layer_types:
            self._sw_idx = [i for i, t in enumerate(layer_types) if t == "sliding_attention"]
            self._full_idx = [i for i in range(self._n_layers) if i not in set(self._sw_idx)]
            self._sw = int(self._text_cfg.sliding_window)
            if self._sw % _PAGE:
                raise ValueError(f"sliding_window {self._sw} must be a multiple of {_PAGE}")
            if not _REGISTERED["masks"]:
                raise RuntimeError(
                    "hybrid sliding/full aux model needs the sdpa mask delegation "
                    "(transformers masking_utils) for exact prefill masks"
                )
        else:
            self._sw_idx, self._full_idx, self._sw = [], list(range(self._n_layers)), 0
        self._sw_set = set(self._sw_idx)
        # Max pages a sliding ring can span: window + one page of hysteresis. The
        # model's sliding window is clamped by the ENGINE window: rows are re-primed
        # to <= `window` tokens, so ring pages beyond it can never hold live KV.
        # (The v4 32K aux ships sliding_window=32768 = "full attention below 32K";
        # sizing rings by it would ask ~3 GiB/row -> OOM at prewarm 64. Attention
        # masks still use self._sw, so semantics are unchanged — with ctx <= window
        # < sliding_window nothing ever evicts either way.)
        _eff_sw = min(self._sw, window) if self._sw else 0
        self._sw_ring_pages = (_eff_sw + _PAGE) // _PAGE + 1 if _eff_sw else 0

        self._claim_model()
        self._device = device
        self._dtype = dtype
        self._window = window
        # The graphed decode path skips PairedLookupRotary's per-step position
        # bounds check (a host sync, illegal under capture), so bound positions
        # against the rope cache here instead. Through step() positions are
        # <= window - 1 (rows at seq_len + 1 > window re-prime first; prefill
        # feeds <= window tokens), so window == cache_len is exact — the v4
        # 32K pair runs precisely there — and only window > cache_len is an
        # impossible config. step_pairs may transiently CROSS the window
        # (bridge bursts; step() re-primes right after), so it carries its own
        # per-feed guard against _rope_cache_len below.
        self._rope_cache_len: int | None = None
        for _m in self._model.modules():
            _clen = getattr(_m, "cache_len", None)
            if isinstance(_clen, int):
                if window > _clen:
                    raise ValueError(
                        f"engine window {window} > paired rope cache_len {_clen}: "
                        "decode positions would index the rope cache out of bounds "
                        "(the graphed decode path cannot catch this per-step)"
                    )
                self._rope_cache_len = (_clen if self._rope_cache_len is None
                                        else min(self._rope_cache_len, _clen))
        margin = float(os.environ.get("DD_REPRIME_MARGIN", "0.25"))
        self._reprime_keep = max(1, window - int(window * margin))
        self._prefill_budget = int(os.environ.get("DD_PREFILL_TOKEN_BUDGET", "32768"))
        # DD_PRIME_STATS=1: per-prefill-group host-issue/GPU cost (printed as
        # each group's events complete — event queries only, no syncs).
        self._pstat_on = os.environ.get("DD_PRIME_STATS", "0") == "1"
        self._pstat: list[tuple] = []  # (rows, T, host_ms, ev0, ev1)
        self._pstat_tot = [0, 0, 0.0, 0.0]  # groups, row-tokens, host_ms, gpu_ms

        self._states: dict[int, _RowState] = {}
        self._pool_full = _PagePool("full", self._full_idx, self._n_kv, self._head_dim,
                                    dtype, device)
        self._pool_sw = (
            _PagePool("sliding", self._sw_idx, self._n_kv, self._head_dim, dtype, device)
            if self._sw_idx else None
        )
        self._pool_tokens_hint = pool_tokens
        self._pool_gb = pool_gb
        self._cache_adapter = _PagedCacheAdapter(self, self._n_layers)
        self._step_ctx: _StepCtx | None = None
        self._cpos0 = torch.zeros(1, dtype=torch.long, device=device)

        # Page-TABLE mirrors of the per-row page lists (torch CPU int32,
        # indexed by _RowState.slot): the decode staging reads page ids and
        # builds flashinfer plan inputs with vector ops instead of per-row
        # python. The python lists in _RowState remain the source of truth;
        # every list mutation goes through a _tbl_* helper to keep the mirror
        # in sync (DD_DEBUG=1 validates per step).
        self._max_pg_full = math.ceil((window + 1) / _PAGE) + 1
        self._tbl_full: torch.Tensor | None = None  # [S, replicas, max_pg_full]
        self._npg_full: torch.Tensor | None = None  # [S, replicas]
        self._tbl_sw: torch.Tensor | None = None
        self._npg_sw: torch.Tensor | None = None
        self._slot_free: list[int] = []
        self._n_slots = 0
        # Reusable pinned staging (one [6, R] pack + plan arrays per step).
        self._pin_cap = 0
        # Captured decode graphs by bucket row count (CUDA + flashinfer only).
        self._graphs: dict[int, _DecodeGraph] = {}
        self._graph_pool = None
        self._scratch_full: int | None = None  # page ids pad rows write into
        self._scratch_sw: int | None = None

        # flashinfer decode wrappers (CUDA only; env-gatable for A/B). The
        # sliding group gets its own wrapper: window_left differs per plan.
        # One float workspace is shared by every wrapper on this device (they
        # only ever run sequentially on one stream), incl. the per-bucket
        # cudagraph-mode wrappers.
        self._fi_full = None
        self._fi_sw = None
        self._fi_ws = None
        if device.type == "cuda" and os.environ.get("DD_PAGED_FLASHINFER", "1") == "1":
            try:
                import flashinfer

                self._fi_ws = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=device)

                def _wrapper():
                    return flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                        self._fi_ws, "NHD", use_tensor_cores=True
                    )

                self._fi_full = _wrapper()
                if self._sw_idx:
                    self._fi_sw = _wrapper()
            except Exception as e:  # noqa: BLE001 — fallback is exact, just slower
                print(f"[AuxBatchedEngine] flashinfer unavailable ({e}); gather+SDPA fallback",
                      flush=True)
        # CUDA-graphed decode: the whole fused forward (launch storm, mask/
        # rotary dicts, flashinfer runs, KV scatter) replays as ONE graph per
        # batch bucket. DD_PAGED_GRAPHS=0 forces the eager decode for A/B.
        self._graphs_enabled = (
            self._fi_full is not None and os.environ.get("DD_PAGED_GRAPHS", "1") == "1"
        )
        groups = (f"{len(self._full_idx)} full + {len(self._sw_idx)} sliding(w={self._sw})"
                  if self._sw_idx else f"{len(self._full_idx)} full")
        print(
            f"[AuxBatchedEngine] {name}: paged KV (page={_PAGE}), window {window}, "
            f"layers {groups}, attention={'flashinfer' if self._fi_full else 'gather+sdpa'}",
            flush=True,
        )
        if compile_model:
            print("[AuxBatchedEngine] compile_model requested: runs eager (flag ignored)",
                  flush=True)

    def _claim_model(self) -> None:
        """Point the model's attention modules at THIS engine. Cheap ownership
        guard, re-checked each step: engines sharing one model instance would
        otherwise read each other's page pools."""
        if getattr(self._attn_modules[0], "_dd_paged_engine", None) is not self:
            for m in self._attn_modules:
                m._dd_paged_engine = self

    # ── Pool sizing ───────────────────────────────────────────────────────────

    def _default_full_pages(self) -> int:
        """Lazy full-pool sizing (first allocation without a prewarm call).
        pool_gb caps it exactly like prewarm does — the batch x window worst
        case below is only sane for small windows."""
        if self._pool_gb is not None:
            rows = max(64, len(self._states) or 64) * self._replicas
            sw_bytes = (self._pool_sw.total_bytes()
                        or rows * self._sw_ring_pages * self._pool_sw.bytes_per_page()
                        * len(self._sw_idx)) if self._pool_sw else 0
            per_page = self._pool_full.bytes_per_page() * len(self._full_idx)
            return max(64, int((self._pool_gb * 1e9 - sw_bytes) // per_page))
        if self._pool_tokens_hint:
            return math.ceil(self._pool_tokens_hint / _PAGE)
        rows = max(64, len(self._states) or 64) * self._replicas
        return rows * (math.ceil((self._window + 1) / _PAGE) + 1)

    def _alloc_full(self, n: int) -> list[int]:
        return self._pool_full.alloc(n, self._default_full_pages())

    def _alloc_sw(self, n: int) -> list[int]:
        rows = max(64, len(self._states) or 64) * self._replicas
        return self._pool_sw.alloc(n, rows * self._sw_ring_pages)

    def _free_row_pages(self, st: _RowState) -> None:
        for plane_pages in st.pages_full:
            self._pool_full.free.extend(plane_pages)
            plane_pages.clear()
        if self._pool_sw is not None:
            for plane_pages in st.pages_sw:
                self._pool_sw.free.extend(plane_pages)
                plane_pages.clear()
        st.sw_drop = 0
        if self._npg_full is not None:
            self._npg_full[st.slot].zero_()
        if self._npg_sw is not None:
            self._npg_sw[st.slot].zero_()

    # ── Page-table mirrors (decode staging reads these, never the lists) ──────

    def _grow_slots(self) -> None:
        new = max(64, self._n_slots * 2)

        def _grow(tbl, npg, max_pg):
            t = torch.zeros(new, self._replicas, max_pg, dtype=torch.int32)
            g = torch.zeros(new, self._replicas, dtype=torch.int32)
            if tbl is not None:
                t[: self._n_slots] = tbl
                g[: self._n_slots] = npg
            return t, g

        self._tbl_full, self._npg_full = _grow(self._tbl_full, self._npg_full,
                                               self._max_pg_full)
        if self._sw_idx:
            self._tbl_sw, self._npg_sw = _grow(self._tbl_sw, self._npg_sw,
                                               self._sw_ring_pages)
        self._slot_free.extend(range(self._n_slots, new))
        self._n_slots = new

    def _tbl_pair(self, sliding: bool):
        return (self._tbl_sw, self._npg_sw) if sliding else (self._tbl_full, self._npg_full)

    def _tbl_set(self, st: _RowState, plane: int, sliding: bool, pages: list[int]) -> None:
        tbl, npg = self._tbl_pair(sliding)
        if pages:
            tbl[st.slot, plane, : len(pages)] = torch.tensor(pages, dtype=torch.int32)
        npg[st.slot, plane] = len(pages)

    def _tbl_append(self, st: _RowState, plane: int, sliding: bool, page: int) -> None:
        tbl, npg = self._tbl_pair(sliding)
        k = int(npg[st.slot, plane])
        tbl[st.slot, plane, k] = page
        npg[st.slot, plane] = k + 1

    def _tbl_popleft(self, st: _RowState, plane: int) -> None:
        tbl, npg = self._tbl_sw, self._npg_sw
        k = int(npg[st.slot, plane])
        tbl[st.slot, plane, : k - 1] = tbl[st.slot, plane, 1:k].clone()
        npg[st.slot, plane] = k - 1

    def _validate_tables(self) -> None:
        """DD_DEBUG cross-check: mirrors must equal the python page lists."""
        for rid, st in self._states.items():
            for plane in range(self._replicas):
                checks = [(False, st.pages_full[plane])]
                if self._sw_idx:
                    checks.append((True, st.pages_sw[plane]))
                for sliding, pages in checks:
                    tbl, npg = self._tbl_pair(sliding)
                    k = int(npg[st.slot, plane])
                    got = tbl[st.slot, plane, :k].tolist()
                    if k != len(pages) or got != pages:
                        raise AssertionError(
                            f"page-table mirror desync req={rid} plane={plane} "
                            f"sliding={sliding}: table {got} != list {pages}"
                        )

    def _ensure_staging(self, rows: int) -> None:
        if self._pin_cap >= rows:
            return
        cap = max(64, 1 << (rows - 1).bit_length())
        pin = self._device.type == "cuda"
        self._pin_pack = torch.zeros(6, cap, dtype=torch.long, pin_memory=pin)
        self._pin_iptr_full = torch.zeros(cap + 1, dtype=torch.int32, pin_memory=pin)
        self._pin_lpl_full = torch.zeros(cap, dtype=torch.int32, pin_memory=pin)
        self._pin_ind_full = torch.zeros(cap * self._max_pg_full, dtype=torch.int32,
                                         pin_memory=pin)
        if self._sw_idx:
            self._pin_iptr_sw = torch.zeros(cap + 1, dtype=torch.int32, pin_memory=pin)
            self._pin_lpl_sw = torch.zeros(cap, dtype=torch.int32, pin_memory=pin)
            self._pin_ind_sw = torch.zeros(cap * self._sw_ring_pages, dtype=torch.int32,
                                           pin_memory=pin)
        self._pin_cap = cap

    # ── Public API ────────────────────────────────────────────────────────────

    def register(self, req_id: int) -> None:
        if not self._slot_free:
            self._grow_slots()
        self._states[req_id] = _RowState(
            pages_full=[[] for _ in range(self._replicas)],
            pages_sw=[[] for _ in range(self._replicas)],
            slot=self._slot_free.pop(),
        )

    def unregister(self, req_id: int) -> None:
        st = self._states.pop(req_id, None)
        if st is not None:
            self._free_row_pages(st)
            self._slot_free.append(st.slot)

    def rewind(self, req_id: int, k: int) -> None:
        """Logically drop the last ``k`` fed tokens (universal-bridge repair).
        Tail slots are overwritten as new tokens feed in; surplus tail pages
        stay owned by the row. Exact while the rewind stays inside the sliding
        ring (bridge repairs are <= a few tokens; the ring holds >= window)."""
        st = self._states[req_id]
        if not st.primed:
            raise ValueError(f"rewind on unprimed request {req_id}")
        if k > st.seq_len:
            raise ValueError(f"rewind({k}) exceeds cached length {st.seq_len}")
        if self._sw_idx and st.seq_len - k < st.sw_drop:
            raise ValueError(
                f"rewind({k}) crosses the evicted sliding window (drop={st.sw_drop})"
            )
        st.seq_len -= k

    @torch.no_grad()
    def prewarm(self, batch_size: int) -> None:
        """Size the page pools for ``batch_size`` rows and warm the kernels
        (flashinfer plan/run JIT, cuBLAS autotune). No graphs — cheap.

        Sliding pools are exact-sized (ring x rows). The full pool takes the
        remainder of ``pool_gb`` when given, else the ``batch x window`` worst
        case (only sane for small windows)."""
        rows = batch_size * self._replicas
        if self._pool_sw is not None:
            need_sw = rows * self._sw_ring_pages
            if self._pool_sw.n_pages < need_sw:
                self._pool_sw.reset()
                self._pool_sw.grow(need_sw, warn=False)
        if self._pool_gb is not None:
            sw_bytes = self._pool_sw.total_bytes() if self._pool_sw else 0
            budget = self._pool_gb * 1e9 - sw_bytes
            per_page = self._pool_full.bytes_per_page() * len(self._full_idx)
            n_full = max(rows, int(budget // per_page))
        else:
            n_full = rows * (math.ceil((self._window + 1) / _PAGE) + 1)
            if self._pool_tokens_hint:
                n_full = max(n_full, math.ceil(self._pool_tokens_hint / _PAGE))
        if self._pool_full.n_pages < n_full:
            self._pool_full.reset()
            self._pool_full.grow(n_full, warn=False)

        dummies = [-(i + 1) for i in range(batch_size)]
        for rid in dummies:
            self.register(rid)
        prompt = list(range(1, 9))
        self.step([(rid, prompt, []) for rid in dummies])
        for k in range(3):
            self.step([(rid, prompt, [1] * (k + 1)) for rid in dummies])
        self.step_pairs([(rid, [1, 2]) for rid in dummies])
        # Capture the whole decode-graph bucket ladder NOW, on the caller's
        # thread. Lazy capture during serving would fire mid-ramp — on the
        # aux WORKER thread when DD_AUX_THREAD is on — and each capture is a
        # ~0.1 s stall the ramp steps shouldn't pay anyway.
        if self._graphs_enabled:
            by_bucket: dict[int, int] = {}
            for n_items in range(1, batch_size + 1):
                by_bucket[self._bucket_for(n_items * self._replicas)] = n_items
            for _b, n_items in sorted(by_bucket.items()):
                self.step([(rid, prompt, [1, 1]) for rid in dummies[:n_items]])
        for rid in dummies:
            self.unregister(rid)
        pool_gb = (self._pool_full.total_bytes()
                   + (self._pool_sw.total_bytes() if self._pool_sw else 0)) / 1e9
        ctx_cap = self._pool_full.n_pages * _PAGE // max(1, self._replicas)
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)
            print(
                f"[AuxBatchedEngine] prewarmed batch={batch_size} on {self._device} "
                f"(pools={pool_gb:.1f}GB, full-group capacity ~{ctx_cap:,} context tokens, "
                f"alloc={torch.cuda.memory_allocated(self._device) / 1e9:.1f}GB)",
                flush=True,
            )
        else:
            print(f"[AuxBatchedEngine] prewarmed batch={batch_size} "
                  f"(pools={pool_gb:.2f}GB)", flush=True)

    def _pstat_flush(self) -> None:
        """Print prime-stat entries whose CUDA events have completed (query
        only — never syncs; CPU builds print with gpu_ms = host_ms)."""
        done = 0
        for rows, T, host_ms, ev0, ev1 in self._pstat:
            if ev1 is not None and not ev1.query():
                break
            gpu_ms = ev0.elapsed_time(ev1) if ev1 is not None else host_ms
            t = self._pstat_tot
            t[0] += 1
            t[1] += rows * T
            t[2] += host_ms
            t[3] += gpu_ms
            print(
                f"[prime] rows={rows} T={T} host={host_ms:.1f}ms gpu={gpu_ms:.1f}ms | "
                f"totals: groups={t[0]} row-toks={t[1]} host={t[2] / 1e3:.2f}s "
                f"gpu={t[3] / 1e3:.2f}s",
                flush=True,
            )
            done += 1
        del self._pstat[:done]

    @torch.no_grad()
    def step(
        self,
        requests: list[tuple[int, list[int] | None, list[int]]],
    ) -> torch.Tensor:
        """One batched inference step: requests = [(req_id, prompt_ids,
        output_ids)], returns last-token logits [N, V] ([2, N, V] fused).
        Unprimed rows prefill; primed rows decode their newest token; rows
        that would exceed the window re-prime."""
        if self._pstat_on and self._pstat:
            self._pstat_flush()
        N = len(requests)
        logits_out = torch.empty(self._out_shape(N), device=self._device, dtype=self._dtype)
        if N == 0:
            return logits_out

        to_prefill, to_decode = [], []
        for i, (rid, pids, oids) in enumerate(requests):
            st = self._states[rid]
            if st.primed and st.seq_len + 1 > self._window:
                self._free_row_pages(st)  # re-prime: re-based tail re-prefill
                st.seq_len = 0
                st.primed = False
            if st.primed:
                to_decode.append((i, rid, oids[-1]))
            else:
                ctx = list(pids or []) + list(oids)
                # A prime is only worth its cost while decode steps fit under
                # the window afterwards. len(ctx) == window used to prime the
                # full window and then re-prime on the very NEXT step — the
                # whole initial prefill bought exactly one token (measured as
                # a doubled prime bill at input == window). Contexts at or
                # past the window start straight at the re-prime length.
                keep = self._window if len(ctx) < self._window else self._reprime_keep
                to_prefill.append((i, rid, ctx[-keep:]))

        if to_prefill:
            # Mixed lengths share one padded forward (real traffic never
            # aligns lengths; per-length forwards would serialize the prime
            # bill). Sorted ascending, so the incoming item is the group max:
            # the budget counts PADDED tokens, and the waste guard splits
            # when the shortest row would pay more padding than content.
            to_prefill.sort(key=lambda x: len(x[2]))
            group: list[tuple[int, int, list[int]]] = []
            for item in to_prefill:
                if group and (
                    (len(group) + 1) * len(item[2]) * self._replicas > self._prefill_budget
                    or len(group[0][2]) * 2 < len(item[2])
                ):
                    self._batch_prefill(group, logits_out)
                    group = []
                group.append(item)
            self._batch_prefill(group, logits_out)
        if to_decode:
            self._batch_decode(to_decode, logits_out)
        return logits_out

    @torch.no_grad()
    def step_pairs(self, requests: list[tuple[int, list[int]]]) -> torch.Tensor:
        """Feed 1-2 new tokens per (primed) request; logits are for the LAST
        fed token. Two-token rows run as two chained single-token sub-steps
        (exact; the pairs burst is rare — universal-mode retokenization only)."""
        N = len(requests)
        logits_out = torch.empty(self._out_shape(N), device=self._device, dtype=self._dtype)
        if N == 0:
            return logits_out
        for rid, toks in requests:
            st = self._states[rid]
            if not st.primed:
                raise ValueError(f"step_pairs on unprimed request {rid}")
            if not 1 <= len(toks) <= 2:
                raise ValueError(f"step_pairs takes 1-2 tokens per request, got {len(toks)}")
            # Pairs feeds may transiently cross the WINDOW (step() re-primes
            # the row on its next regular step), but never the ROPE CACHE: the
            # graphed decode skips the per-step bounds check, so an
            # out-of-range position would silently gather garbage rotary rows.
            # Only reachable at window == cache_len (bridge windows normally
            # sit far below the trained context).
            if (self._rope_cache_len is not None
                    and st.seq_len + len(toks) > self._rope_cache_len):
                raise ValueError(
                    f"step_pairs feed of {len(toks)} tokens would index past the "
                    f"paired rope cache ({st.seq_len} cached, cache_len "
                    f"{self._rope_cache_len}) for request {rid}"
                )
        self._batch_decode([(i, rid, toks[0]) for i, (rid, toks) in enumerate(requests)],
                           logits_out)
        second = [(i, rid, toks[1]) for i, (rid, toks) in enumerate(requests) if len(toks) == 2]
        if second:
            sub = torch.empty(self._out_shape(len(second)), device=self._device,
                              dtype=self._dtype)
            self._batch_decode([(j, rid, tok) for j, (_, rid, tok) in enumerate(second)], sub)
            for j, (i, _, _) in enumerate(second):
                if self._replicas == 1:
                    logits_out[i] = sub[j]
                else:
                    logits_out[:, i] = sub[:, j]
        return logits_out

    # ── Internals ─────────────────────────────────────────────────────────────

    def _out_shape(self, n: int) -> tuple[int, ...]:
        return (n, self._vocab_size) if self._replicas == 1 else (2, n, self._vocab_size)

    def _plane_rows(self, items: list) -> list[tuple[int, int]]:
        """Plane-major (plane, item_index) order for R = replicas * n rows."""
        n = len(items)
        return [(p, j) for p in range(self._replicas) for j in range(n)]

    def _scatter_prefill(self, kv, key_states, value_states, ctx: _StepCtx,
                         sliding: bool) -> None:
        """Indexed scatter of key/value [R, n_kv, T, hd] into pages: gather
        each row's kept tokens from the flattened [R * T] stream, write them
        to their precomputed (page, slot)."""
        src = ctx.pf_src_sw if sliding else ctx.pf_src_full
        pg = ctx.pf_pg_sw if sliding else ctx.pf_pg_full
        slot = ctx.pf_slot_sw if sliding else ctx.pf_slot_full
        R, n_kv, T, hd = key_states.shape
        for sel, s in ((0, key_states), (1, value_states)):
            flat = s.permute(0, 2, 1, 3).reshape(R * T, n_kv, hd)
            kv.select(1, sel)[pg, slot] = flat[src]

    def _batch_prefill(self, items: list[tuple[int, int, list[int]]], logits_out) -> None:
        """Grouped prefill: ONE stock forward for a batch of contexts whose
        lengths may DIFFER. Shorter rows are LEFT-padded to the group max so
        every row's last real token sits at index T-1 (one logits_to_keep=1
        read serves the group); pad positions clamp to 0 and are hidden from
        attention by a standard 2D padding mask, delegated to the stock
        sdpa/FlexLlama mask builders — UNIFORM groups keep attention_mask=None
        (the is_causal fast path), bitwise-identical to the pre-padding path.
        The cache adapter scatters each layer's K/V through precomputed
        (src, page, slot) index tensors (full layers every real token,
        sliding layers the 16-aligned window tail), so pads never enter the
        page pools and later decode steps are indistinguishable from an
        unpadded prime."""
        n = len(items)
        lens = [len(it[2]) for it in items]
        T = max(lens)
        rows = self._plane_rows(items)
        R = len(rows)
        dev = self._device
        ev0 = None
        t_issue0 = 0.0
        if self._pstat_on:
            t_issue0 = time.perf_counter()
            if dev.type == "cuda":
                ev0 = torch.cuda.Event(enable_timing=True)
                ev0.record(torch.cuda.current_stream(dev))
        # Async H2D (pinned + non_blocking) like the decode path: a blocking
        # torch.tensor(..., device=dev) stream-syncs the aux device, and inside
        # vLLM's single-threaded loop (current device = P's) that host stall
        # serializes into the loop on every prefill/re-prime.
        pin = dev.type == "cuda"

        def _to_dev(t):
            return (t.pin_memory() if pin else t).to(dev, non_blocking=True)

        uniform = min(lens) == T
        input_ids = _to_dev(torch.tensor(
            [[0] * (T - lens[j]) + items[j][2] for _, j in rows], dtype=torch.long))
        if _DEBUG:
            print(f"[paged dbg] prefill: n={n} T={T} lens={sorted(set(lens))}", flush=True)

        # Per-row page allocation for both groups (sliding rows keep the
        # 16-aligned window tail of their OWN length).
        drops = [0] * n
        if self._sw_idx:
            drops = [((L - self._sw) // _PAGE) * _PAGE if self._sw < L else 0
                     for L in lens]
        slots = torch.tensor([self._states[it[1]].slot for it in items], dtype=torch.long)
        for p, j in rows:
            st = self._states[items[j][1]]
            pages = self._alloc_full(math.ceil(lens[j] / _PAGE))
            st.pages_full[p] = pages
            self._tbl_set(st, p, False, pages)
            if self._sw_idx:
                pages = self._alloc_sw(math.ceil((lens[j] - drops[j]) / _PAGE))
                st.pages_sw[p] = pages
                self._tbl_set(st, p, True, pages)

        # (src, page, slot) index tensors, built with [R, T]-grid vector ops
        # from the page-table mirrors (no per-token python).
        lens_r = torch.tensor(lens, dtype=torch.long).repeat(self._replicas)
        drops_r = torch.tensor(drops, dtype=torch.long).repeat(self._replicas)
        planes_r = torch.arange(self._replicas, dtype=torch.long
                                ).repeat_interleave(n)
        slots_r = slots.repeat(self._replicas)
        ar = torch.arange(T, dtype=torch.long)
        k = ar.unsqueeze(0) - (T - lens_r).unsqueeze(1)  # token idx in row; <0 = pad
        src_grid = torch.arange(R, dtype=torch.long).unsqueeze(1) * T + ar.unsqueeze(0)

        def _indices(tbl, k_g, keep):
            kc = k_g.clamp_min(0)
            pages = tbl[slots_r, planes_r].long()  # [R, max_pg]
            pg_grid = pages.gather(1, kc // _PAGE)
            return (_to_dev(src_grid[keep]), _to_dev(pg_grid[keep]),
                    _to_dev((kc % _PAGE)[keep]))

        ctx = _StepCtx(mode="prefill", T=T)
        ctx.pf_src_full, ctx.pf_pg_full, ctx.pf_slot_full = _indices(
            self._tbl_full, k, k >= 0)
        if self._sw_idx:
            k_sw = k - drops_r.unsqueeze(1)
            ctx.pf_src_sw, ctx.pf_pg_sw, ctx.pf_slot_sw = _indices(
                self._tbl_sw, k_sw, k_sw >= 0)

        # Explicit per-row positions: a fused pair's lookup rotaries (paired
        # per-plane caches) need plane-major [R, T] positions, not the [1, T]
        # broadcast the model would default to. Pad positions clamp to 0
        # (attention never reads them).
        cpos = _to_dev(ar)
        pos = _to_dev(k.clamp_min(0))
        attn_mask = None if uniform else _to_dev((k >= 0).to(torch.long))
        self._claim_model()
        self._step_ctx = ctx
        try:
            out = self._model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                position_ids=pos,
                past_key_values=self._cache_adapter,
                cache_position=cpos,
                use_cache=True,
                logits_to_keep=1,
            )
        finally:
            self._step_ctx = None

        lg = out.logits[:, -1].to(self._dtype)  # [R, V]
        if self._pstat_on:
            ev1 = None
            if ev0 is not None:
                ev1 = torch.cuda.Event(enable_timing=True)
                ev1.record(torch.cuda.current_stream(dev))
            self._pstat.append(
                (R, T, (time.perf_counter() - t_issue0) * 1e3, ev0, ev1)
            )
        for j, (out_idx, rid, ctx_toks) in enumerate(items):
            st = self._states[rid]
            st.seq_len = len(ctx_toks)
            st.sw_drop = drops[j]
            st.primed = True
            if self._replicas == 1:
                logits_out[out_idx] = lg[j]
            else:
                logits_out[0, out_idx] = lg[j]
                logits_out[1, out_idx] = lg[n + j]

    def _batch_decode(self, items: list[tuple[int, int, int]], logits_out) -> None:
        """One forward over the ACTIVE rows (both planes fused): a captured
        CUDA-graph replay when available (CUDA + flashinfer, DD_PAGED_GRAPHS),
        the eager forward otherwise (exact same math)."""
        n = len(items)
        R = n * self._replicas
        g = self._graph_for(self._bucket_for(R)) if self._graphs_enabled else None
        staged = self._stage_decode(items, g.Rb if g is not None else R)
        if g is not None and self._pool_gen() != g.generation:
            # _stage_decode's tail-page allocation grew a pool (the free list
            # was exhausted), REPLACING the kv tensors this graph baked in as
            # raw pointers — replaying now would read freed memory. Drop the
            # stale graphs and run this step eager (staged is bucket-padded,
            # which _eager_decode/_copy_out handle); the next step recaptures
            # against the grown pool.
            self._graphs.clear()
            self._graph_pool = None
            g = None
        if g is not None:
            # Pin the device: inside vLLM this thread's current device is P's,
            # while the plan copies and the replay must hit the aux device.
            with torch.cuda.device(self._device):
                g.pack.copy_(staged.pack, non_blocking=True)
                self._plan_wrapper(g.fi_full, staged.indptr_full, staged.indices_full,
                                   staged.lpl_full, sliding=False)
                if g.fi_sw is not None:
                    self._plan_wrapper(g.fi_sw, staged.indptr_sw, staged.indices_sw,
                                       staged.lpl_sw, sliding=True)
                self._mark_staging_consumed()
                g.graph.replay()
            lg = g.logits
        else:
            lg = self._eager_decode(staged, items)
        self._copy_out(lg, staged, items, logits_out)
        for _, rid, _ in items:
            self._states[rid].seq_len += 1

    def _stage_decode(self, items, Rb: int) -> _Staged:
        """Vectorized staging for one decode step, laid out for ``Rb`` rows
        (plane-major with stride ``Rb // replicas``; pad rows point at the
        scratch page with length 1). Performs the page-granularity mutations
        (ring eviction, tail-page allocation) as a side effect — both are
        guarded by state so re-staging the same step is idempotent."""
        n = len(items)
        rep = self._replicas
        half = Rb // rep
        has_sw = bool(self._sw_idx)
        sts = [self._states[it[1]] for it in items]

        for st in sts:
            # Sliding-ring eviction (page granularity, once per crossing; the
            # drop is shared by the planes).
            if has_sw and st.seq_len + 1 - st.sw_drop > self._sw + _PAGE:
                for p, plane_pages in enumerate(st.pages_sw):
                    self._pool_sw.free.append(plane_pages.pop(0))
                    self._tbl_popleft(st, p)
                st.sw_drop += _PAGE
            if st.seq_len % _PAGE == 0:
                for p in range(rep):
                    pages = st.pages_full[p]
                    if len(pages) * _PAGE <= st.seq_len:
                        pg = self._alloc_full(1)[0]
                        pages.append(pg)
                        self._tbl_append(st, p, False, pg)
                if has_sw:
                    ring = st.seq_len - st.sw_drop  # same 16-alignment as seq_len
                    for p in range(rep):
                        pages = st.pages_sw[p]
                        if len(pages) * _PAGE <= ring:
                            pg = self._alloc_sw(1)[0]
                            pages.append(pg)
                            self._tbl_append(st, p, True, pg)

        # Order matters: drain the previous step's in-flight H2D copies (they
        # read the current pinned buffers) BEFORE _ensure_staging can reallocate
        # them — otherwise a bucket-size growth drops the old pinned buffer while
        # its DMA is still running, reading freed pinned host memory.
        self._wait_staging_free()
        self._ensure_staging(Rb)
        seq = torch.tensor([st.seq_len for st in sts], dtype=torch.long)
        toks = torch.tensor([it[2] for it in items], dtype=torch.long)
        slots = torch.tensor([st.slot for st in sts], dtype=torch.long)
        drop = (torch.tensor([st.sw_drop for st in sts], dtype=torch.long)
                if has_sw else None)

        pk = self._pin_pack[:, :Rb]
        if Rb != n * rep:
            pk.zero_()  # pad rows: token 0 at position 0, write slot 0
            pk[_ST_WRPF].fill_(self._scratch_full)
            if has_sw:
                pk[_ST_WRPS].fill_(self._scratch_sw)
        wrslot = seq % _PAGE
        pgidx = seq // _PAGE
        ring = (seq - drop) if has_sw else None
        for p in range(rep):
            s = slice(p * half, p * half + n)
            pk[_ST_TOK, s] = toks
            pk[_ST_POS, s] = seq
            pk[_ST_WRSLOT, s] = wrslot
            pk[_ST_WRPF, s] = self._tbl_full[slots, p, pgidx].long()
            if has_sw:
                pk[_ST_WRPS, s] = self._tbl_sw[slots, p, ring // _PAGE].long()
                pk[_ST_SWDROP, s] = drop

        staged = _Staged(n=n, Rb=Rb, half=half, pack=pk,
                         max_kv_len=int(seq.max()) + 1)
        if self._fi_full is not None:
            staged.indptr_full, staged.indices_full, staged.lpl_full = (
                self._stage_plan_group(Rb, half, n, seq, slots, sliding=False))
            if has_sw:
                staged.indptr_sw, staged.indices_sw, staged.lpl_sw = (
                    self._stage_plan_group(Rb, half, n, ring, slots, sliding=True))
        if _DEBUG:
            self._validate_tables()
        return staged

    def _stage_plan_group(self, Rb, half, n, seq_g, slots, *, sliding: bool):
        """flashinfer plan inputs (indptr / packed indices / last-page-len) for
        one layer group, built with vector ops from the page-table mirror.
        ``seq_g`` is each item's index of the incoming token within the group's
        table (full: seq_len; sliding: seq_len - sw_drop)."""
        tbl, _ = self._tbl_pair(sliding)
        if sliding:
            iptr, ind, lpl = self._pin_iptr_sw, self._pin_ind_sw, self._pin_lpl_sw
        else:
            iptr, ind, lpl = self._pin_iptr_full, self._pin_ind_full, self._pin_lpl_full
        npg = seq_g // _PAGE + 1  # == ceil((seq_g + 1) / _PAGE)
        maxpg = int(npg.max())
        scratch = (self._scratch_sw if sliding else self._scratch_full) or 0
        npg_rb = torch.ones(Rb, dtype=torch.long)
        lpl_rb = lpl[:Rb]
        lpl_rb.fill_(1)
        gath = torch.full((Rb, maxpg), scratch, dtype=torch.int32)
        lpl_n = (seq_g + 1 - (npg - 1) * _PAGE).to(torch.int32)
        for p in range(self._replicas):
            s = slice(p * half, p * half + n)
            npg_rb[s] = npg
            lpl_rb[s] = lpl_n
            gath[s] = tbl[slots, p, :maxpg]
        ip = iptr[: Rb + 1]
        ip[0] = 0
        ip[1:] = torch.cumsum(npg_rb, 0)
        L = int(ip[Rb])
        mask = torch.arange(maxpg).unsqueeze(0) < npg_rb.unsqueeze(1)
        torch.masked_select(gath, mask, out=ind[:L])
        return ip, ind[:L], lpl_rb

    def _plan_wrapper(self, wrapper, indptr, indices, lpl, *, sliding: bool) -> None:
        wrapper.plan(
            indptr, indices, lpl,
            num_qo_heads=self._text_cfg.num_attention_heads,
            num_kv_heads=self._n_kv,
            head_dim=self._head_dim,
            page_size=_PAGE,
            pos_encoding_mode="NONE",
            window_left=(self._sw - 1) if sliding else -1,
            q_data_type=self._dtype,
            kv_data_type=self._dtype,
            sm_scale=self._sm_scale,
            non_blocking=True,
        )

    def _wait_staging_free(self) -> None:
        """The pinned staging is read by async H2D copies (pack copy + the
        wrappers' plan copies); wait for the previous step's copies before
        overwriting it. The event is recorded right after the copies are
        enqueued, so by the next step it has almost always already fired."""
        evt = getattr(self, "_stage_evt", None)
        if evt is not None:
            evt.synchronize()

    def _mark_staging_consumed(self) -> None:
        if self._device.type != "cuda":
            return
        if getattr(self, "_stage_evt", None) is None:
            self._stage_evt = torch.cuda.Event()
        self._stage_evt.record(torch.cuda.current_stream(self._device))

    def _eager_decode(self, staged: _Staged, items) -> torch.Tensor:
        """The pre-graph decode forward (exact reference; CPU tests, A/B, and
        the fallback when capture is unavailable)."""
        R = staged.Rb
        dev = self._device
        has_sw = bool(self._sw_idx)
        dp = staged.pack.to(dev, non_blocking=True)
        ctx = _StepCtx(
            mode="decode",
            wr_page_full=dp[_ST_WRPF],
            wr_page_sw=dp[_ST_WRPS] if has_sw else None,
            wr_slot=dp[_ST_WRSLOT],
            kv_lens=(dp[_ST_POS] + 1).to(torch.int32),
            sw_lens=((dp[_ST_POS] - dp[_ST_SWDROP]) + 1).to(torch.int32) if has_sw else None,
            max_kv_len=staged.max_kv_len,
        )
        if self._fi_full is not None:
            self._plan_wrapper(self._fi_full, staged.indptr_full, staged.indices_full,
                               staged.lpl_full, sliding=False)
            if has_sw:
                self._plan_wrapper(self._fi_sw, staged.indptr_sw, staged.indices_sw,
                                   staged.lpl_sw, sliding=True)
            ctx.planned = True
        else:
            rows = self._plane_rows(items)
            ctx.gather_full, ctx.n_pg_full = self._gather_plan(rows, items, sliding=False)
            if has_sw:
                ctx.gather_sw, ctx.n_pg_sw = self._gather_plan(rows, items, sliding=True)
        self._mark_staging_consumed()
        self._claim_model()
        self._step_ctx = ctx
        try:
            return self._fwd(dp, R)
        finally:
            self._step_ctx = None

    def _fwd(self, pack: torch.Tensor, R: int) -> torch.Tensor:
        return self._model(
            input_ids=pack[_ST_TOK].view(R, 1),
            position_ids=pack[_ST_POS].view(R, 1),
            past_key_values=self._cache_adapter,
            cache_position=self._cpos0,
            use_cache=True,
            logits_to_keep=1,
        ).logits

    def _copy_out(self, lg, staged: _Staged, items, logits_out) -> None:
        """Scatter the step's last-token logits into the caller's output.
        Everything here must be ASYNC on the device stream: a blocking H2D
        (e.g. torch.tensor(list, device=cuda)) stream-synchronizes and makes
        the host wait out the just-enqueued replay — measured as +14 ms of
        fake 'host' time per step inside vLLM."""
        n = staged.n
        half = staged.half
        if all(it[0] == j for j, it in enumerate(items)):  # steady-state decode
            if self._replicas == 1:
                logits_out[:n].copy_(lg[:n, -1], non_blocking=True)
            else:
                logits_out[0, :n].copy_(lg[:n, -1], non_blocking=True)
                logits_out[1, :n].copy_(lg[half: half + n, -1], non_blocking=True)
            return
        pin = self._device.type == "cuda"
        oidx = torch.tensor([it[0] for it in items], dtype=torch.long,
                            pin_memory=pin).to(self._device, non_blocking=True)
        if self._replicas == 1:
            logits_out.index_copy_(0, oidx, lg[:n, -1].to(self._dtype))
        else:
            logits_out[0].index_copy_(0, oidx, lg[:n, -1].to(self._dtype))
            logits_out[1].index_copy_(0, oidx, lg[half: half + n, -1].to(self._dtype))

    # ── Decode CUDA graphs ────────────────────────────────────────────────────

    def _bucket_for(self, R: int) -> int:
        for b in (8, 16, 32, 64, 128):
            if b >= R:
                return b
        return math.ceil(R / 64) * 64

    def _pool_gen(self) -> int:
        """Summed page-pool generation; bumped whenever a pool's kv tensors are
        replaced (grow/reset). A captured graph is valid only while this equals
        the value at capture time."""
        return self._pool_full.generation + (self._pool_sw.generation if self._pool_sw else 0)

    def _graph_for(self, Rb: int) -> _DecodeGraph | None:
        gen = self._pool_gen()
        g = self._graphs.get(Rb)
        if g is not None and g.generation != gen:
            print("[AuxBatchedEngine] page pools reallocated -> dropping captured "
                  "decode graphs (size the pools up front to avoid this)", flush=True)
            self._graphs.clear()
            # Fresh mempool for the recaptures: reusing the handle while the
            # old graphs' blocks are still being released trips the caching
            # allocator ("use_count > 0" assert). The old pool dies with its
            # graphs; new captures start their own.
            self._graph_pool = None
            g = None
        if g is None:
            try:
                g = self._capture_bucket(Rb, gen)
            except Exception as e:  # noqa: BLE001 — graphs are an optimization; eager is exact
                print(f"[AuxBatchedEngine] decode graph capture failed ({e!r}); "
                      "running eager decode", flush=True)
                self._graphs_enabled = False
                return None
            self._graphs[Rb] = g
        return g

    def _capture_bucket(self, Rb: int, gen: int) -> _DecodeGraph:
        """Capture the whole decode forward at bucket size ``Rb``: persistent
        device inputs, cudagraph-mode flashinfer wrappers (fixed buffers,
        re-planned every step), warmup on a side stream, then record."""
        import flashinfer

        t0 = time.perf_counter()
        dev = self._device
        if self._scratch_full is None:
            self._scratch_full = self._alloc_full(1)[0]
            if self._pool_sw is not None:
                self._scratch_sw = self._alloc_sw(1)[0]
        with torch.cuda.device(dev):
            pack = torch.zeros(6, Rb, dtype=torch.long, device=dev)
            pack[_ST_WRPF].fill_(self._scratch_full)
            if self._sw_idx:
                pack[_ST_WRPS].fill_(self._scratch_sw)

            def _gwrap(max_pg):
                return flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                    self._fi_ws, "NHD", use_cuda_graph=True, use_tensor_cores=True,
                    paged_kv_indptr_buffer=torch.zeros(Rb + 1, dtype=torch.int32, device=dev),
                    paged_kv_indices_buffer=torch.zeros(Rb * max_pg, dtype=torch.int32,
                                                        device=dev),
                    paged_kv_last_page_len_buffer=torch.ones(Rb, dtype=torch.int32,
                                                             device=dev),
                )

            g = _DecodeGraph(Rb=Rb, pack=pack, fi_full=_gwrap(self._max_pg_full),
                             fi_sw=_gwrap(self._sw_ring_pages) if self._sw_idx else None)
            # Plan the capture state: every row one token in the scratch page.
            iptr = torch.arange(Rb + 1, dtype=torch.int32)
            lpl1 = torch.ones(Rb, dtype=torch.int32)
            self._plan_wrapper(
                g.fi_full, iptr,
                torch.full((Rb,), self._scratch_full, dtype=torch.int32), lpl1,
                sliding=False)
            if g.fi_sw is not None:
                self._plan_wrapper(
                    g.fi_sw, iptr,
                    torch.full((Rb,), self._scratch_sw, dtype=torch.int32), lpl1,
                    sliding=True)

            ctx = _StepCtx(
                mode="decode", planned=True, max_kv_len=1,
                wr_page_full=pack[_ST_WRPF],
                wr_page_sw=pack[_ST_WRPS] if self._sw_idx else None,
                wr_slot=pack[_ST_WRSLOT],
            )
            saved = (self._fi_full, self._fi_sw)
            self._fi_full, self._fi_sw = g.fi_full, g.fi_sw
            self._claim_model()
            self._step_ctx = ctx
            try:
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    for _ in range(2):
                        self._fwd(pack, Rb)
                torch.cuda.current_stream().wait_stream(side)
                if self._graph_pool is None:
                    self._graph_pool = torch.cuda.graph_pool_handle()
                graph = torch.cuda.CUDAGraph()
                # thread_local capture scope: with the aux block on its own
                # worker thread, the DEFAULT (global) mode is poisoned by the
                # engine-core loop's concurrent CUDA calls for P (allocs,
                # event queries — measured as cudaErrorIllegalAddress on a
                # mid-ramp capture). The loop thread never touches the aux
                # device while a job is in flight, so thread-local is exact.
                with torch.cuda.graph(graph, pool=self._graph_pool,
                                      capture_error_mode="thread_local"):
                    g.logits = self._fwd(pack, Rb)
                g.graph = graph
            finally:
                self._step_ctx = None
                self._fi_full, self._fi_sw = saved
            torch.cuda.synchronize()
        g.generation = gen
        print(f"[AuxBatchedEngine] captured decode graph rows={Rb} "
              f"({time.perf_counter() - t0:.1f}s)", flush=True)
        return g

    def _row_table(self, st: _RowState, plane: int, sliding: bool) -> tuple[list[int], int]:
        """(page list, table length in tokens) for one row-plane, INCLUDING
        this step's write."""
        if sliding:
            length = st.seq_len + 1 - st.sw_drop
            return st.pages_sw[plane], length
        return st.pages_full[plane], st.seq_len + 1

    def _gather_plan(self, rows, items, *, sliding: bool):
        max_pg = 1
        for p, j in rows:
            st = self._states[items[j][1]]
            _, length = self._row_table(st, p, sliding)
            max_pg = max(max_pg, math.ceil(length / _PAGE))
        gat = torch.zeros(len(rows), max_pg, dtype=torch.long)
        for r, (p, j) in enumerate(rows):
            st = self._states[items[j][1]]
            pages, length = self._row_table(st, p, sliding)
            m = min(len(pages), max_pg)
            gat[r, :m] = torch.tensor(pages[:m], dtype=torch.long)
        return gat.view(-1).to(self._device), max_pg

    def _attend_decode(self, layer_idx: int, query: torch.Tensor, scaling) -> torch.Tensor:
        """Decode attention for one layer: q [R, Hq, 1, E] -> [R, 1, Hq, E]."""
        ctx = self._step_ctx
        sliding = layer_idx in self._sw_set
        pool = self._pool_sw if sliding else self._pool_full
        if ctx.planned:
            if scaling is not None and abs(scaling - self._sm_scale) > 1e-9:
                raise RuntimeError(
                    f"attention scaling {scaling} != planned sm_scale {self._sm_scale}; "
                    "set DD_PAGED_FLASHINFER=0 for this architecture"
                )
            q = query.squeeze(2).contiguous()
            out = (self._fi_sw if sliding else self._fi_full).run(q, pool.kv[layer_idx])
            return out.unsqueeze(1)
        # Fallback: gather each row's pages, mask to the group's bound, SDPA.
        R = query.shape[0]
        gather = ctx.gather_sw if sliding else ctx.gather_full
        n_pg = ctx.n_pg_sw if sliding else ctx.n_pg_full
        lens = ctx.sw_lens if sliding else ctx.kv_lens
        S = n_pg * _PAGE
        kv = pool.kv[layer_idx]
        k = kv.select(1, 0).index_select(0, gather)
        v = kv.select(1, 1).index_select(0, gather)
        k = k.view(R, S, self._n_kv, self._head_dim).transpose(1, 2)
        v = v.view(R, S, self._n_kv, self._head_dim).transpose(1, 2)
        ar = torch.arange(S, device=query.device).view(1, 1, 1, S)
        lens4 = lens.view(R, 1, 1, 1)
        invalid = ar >= lens4
        if sliding:  # only the last `sw` table entries are visible
            invalid |= ar < (lens4 - self._sw)
        mask = torch.zeros(R, 1, 1, S, dtype=query.dtype, device=query.device)
        mask.masked_fill_(invalid, torch.finfo(query.dtype).min)
        out = F.scaled_dot_product_attention(
            query, k, v, attn_mask=mask, scale=scaling, enable_gqa=(self._n_kv != query.shape[1])
        )
        return out.transpose(1, 2)  # [R, 1, Hq, E]


# The paged engine shipped as PagedAuxEngine before replacing the original
# engine outright; keep the name importable.
PagedAuxEngine = AuxBatchedEngine
