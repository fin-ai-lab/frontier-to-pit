"""SAE-feature steering of the **P model**, applied inside the vLLM worker.

Divergence Decoding shifts P's *logits* with a forget/retain aux pair. This
module is the orthogonal lever: it shifts P's *residual stream* by adding a
multiple of an SAE feature's decoder vector at a chosen decoder block — the
TopK-SAE port of ``run_steer.py`` (clamp_floor amplification), now running
against the model vLLM already serves instead of a separate HF copy.

The two compose: install the steering hooks AND pass the DD logits processor and
P is steered in the residual stream *and* fused in logit space in the same
``generate()``.

Scope (what's ported)
---------------------
Only the **clamp_floor** mode — the single intervention ``run_steer.py`` uses to
amplify candidate features. At decoder block ``L``, over the target features
``fs``, at every position::

    pre_f = x · W_encᵀ[fs] + b_enc[fs]      # (T, nf) — only the targets, no full encode
    feat  = gate(pre_f)                      # see below — per-feature, no full encode
    delta = clamp(clamp_value − feat, min=0) # raise feat up to clamp_value; else 0
    x    += delta · W_dec[fs]                # inject along the decoder vectors

Two SAE families, two gates (selected by ``SaeSource.family``):

* **TopK / Qwen-Scope** (``family="topk"``) → ``feat = relu(pre_f)``. relu is a
  *fast approximation* of the SAE's trained global top-k membership — ~100×
  cheaper than a full encode, and for clamp_floor at our clamp values the two
  agree to <~0.3% (``verify_steer.py`` in eval-dd).
* **JumpReLU / Gemma-Scope** (``family="jumprelu"``) →
  ``feat = pre_f · (pre_f > threshold_f)``. This is the SAE's *exact* gate (a
  fixed per-feature threshold), not an approximation — and still per-feature, so
  no more expensive than relu. Bit-equivalent to the HF Gemma steering hook.

``clamp_value=0`` is a no-op for either gate (``delta=0`` since ``feat≥0``), so
the unsteered baseline needs no hook at all.

vLLM residual layout
--------------------
vLLM decoder blocks (Qwen3, Llama, …) return ``(hidden_states, residual)`` with
the residual add *deferred* to the next block's input norm, so the true
``resid_post`` the SAE was trained on is ``hidden_states + residual`` — not the
layer output verbatim (the HF convention). The hook reconstructs that sum to
encode, then injects the delta back into ``hidden_states`` (``set_residual=True``
in :func:`make_steer_hook`). On a plain HF model the layer output already *is*
``resid_post`` (``set_residual=False``), which is what the CPU tests exercise.

Caveat — post-build hooks are eager only
----------------------------------------
Forward hooks fire only when the block runs in eager Python. Under vLLM's CUDA
graphs the captured replay skips them, so steering installed AFTER engine build
(this module's :func:`install_steering`) is silently a no-op unless the engine
was built with ``enforce_eager=True``; :func:`install_steering` refuses to
install on a non-eager engine.

For production serving there is a second route that keeps full cudagraph speed:
install the hooks BEFORE vLLM traces and captures the model, so the hook's
kernels are recorded inside the graphs — see ``steering_worker.DDSteeringWorker``
(``serve.build_llm(steer_precapture=True)``). Measured on the 2xH100 box: the
steered engine then decodes at unsteered throughput (the eager route halves it).
"""

from __future__ import annotations

import functools
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from vllm import LLM

# Attribute the installed hook handles are stashed under, on the worker's model.
_HANDLES_ATTR = "_dd_steer_handles"

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


@dataclass
class SteerSpec:
    """One clamp_floor steering intervention at decoder block ``layer``.

    Args:
        layer: Decoder block whose ``resid_post`` is steered (the block the
            ``layer{L}.sae.pt`` SAE was trained on).
        feature_id: One SAE feature id, or several applied additively in one
            pass — the multi-steer case (2-3 candidate features at once).
        clamp_value: ``feat`` is raised up to this value wherever it falls short
            (a large value amplifies the feature). ``0`` is a no-op baseline.
    """

    layer: int
    feature_id: int | list[int]
    clamp_value: float = 0.0

    def feature_ids(self) -> list[int]:
        f = self.feature_id
        return [int(f)] if isinstance(f, int) else [int(x) for x in f]


@dataclass
class SaeSource:
    """Where the worker loads an SAE checkpoint from (must be picklable — it is
    shipped to the vLLM worker by :meth:`vllm.LLM.apply_model`).

    Two SAE families, selected by ``family``:

    * ``"topk"`` (default) — **Qwen-Scope** TopK SAE. A torch ``layer{L}.sae.pt``
      state dict: ``W_enc`` ``(d_sae, d_model)``, ``b_enc`` ``(d_sae,)``,
      ``W_dec`` ``(d_model, d_sae)`` (transposed to feature-major on load). The
      hook gates with relu (the fast top-k approximation).
    * ``"jumprelu"`` — **Gemma-Scope-2** JumpReLU SAE (pairs with
      ``google/gemma-3-27b-it``). A ``params.safetensors`` under
      ``resid_post/layer_{L}_width_{width}_l0_{l0_size}/`` with ``w_enc``
      ``(d_model, d_sae)``, ``b_enc``, ``threshold`` ``(d_sae,)``, ``w_dec``
      ``(d_sae, d_model)``. The hook gates with the exact JumpReLU threshold,
      matching the HF Gemma steering — and just as cheap as relu (per-feature
      gate, no full encode). Feature ids index the chosen ``width``/``l0_size``
      SAE, so use the same ones the features were identified at (default
      65k/medium, matching forecasting-thoughts' ``load_gemma_sae``).

    Provide ``path`` (a staged checkpoint) OR ``repo_id`` (+ ``layer``).

    Args:
        path: Direct path to the checkpoint (skips the Hub).
        repo_id: HF Hub repo. topk e.g. ``"Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_50"``;
            jumprelu e.g. ``"google/gemma-scope-2-27b-it"``.
        layer: Layer index for the Hub filename. Defaults to the paired
            :class:`SteerSpec`'s ``layer``.
        cache_dir: HF download cache (topk also falls back to
            ``$QWEN_SCOPE_CACHE``).
        dtype: Precision the SAE weights are held/encoded in (the injection
            casts back to the model dtype).
        family: ``"topk"`` (Qwen-Scope) or ``"jumprelu"`` (Gemma-Scope).
        width: jumprelu only — Gemma-Scope width bucket (e.g. ``"65k"``).
        l0_size: jumprelu only — Gemma-Scope L0 bucket (``"small"``/``"medium"``
            /``"big"``).
    """

    path: str | None = None
    repo_id: str | None = None
    layer: int | None = None
    cache_dir: str | None = None
    dtype: str = "float32"
    family: str = "topk"
    width: str = "65k"
    l0_size: str = "medium"

    def __post_init__(self) -> None:
        if not self.path and not self.repo_id:
            raise ValueError("SaeSource needs either a path or a repo_id")
        if self.family not in ("topk", "jumprelu"):
            raise ValueError(f"SaeSource.family must be 'topk' or 'jumprelu', got {self.family!r}")


@dataclass
class _SteerWeights:
    """The feature rows the hook closes over (live on the model device)."""

    W_enc_f: torch.Tensor  # (nf, d_model) — encoder rows for the targets
    b_enc_f: torch.Tensor  # (nf,)
    W_dec_f: torch.Tensor  # (nf, d_model) — feature-major decoder rows
    thr_f: torch.Tensor | None = None  # (nf,) JumpReLU thresholds; None → relu gate


# ── The hook (model-agnostic core) ──────────────────────────────────────────


def make_steer_hook(w: _SteerWeights, spec: SteerSpec, *, set_residual: bool):
    """Build a forward hook that adds clamp_floor ``delta · W_dec[fs]`` to
    ``resid_post``. All math in fp32 (the SAE's trained precision); the
    injection casts back to the block's dtype.

    Gate (how ``feat`` — the SAE activation — is computed from the target
    pre-activations):

    * ``w.thr_f is None`` → **relu** (``feat = max(pre_f, 0)``): the fast
      per-feature approximation of a TopK SAE's global top-k membership
      (Qwen-Scope). Cheap because it never does the full encode.
    * ``w.thr_f`` set → **JumpReLU** (``feat = pre_f · (pre_f > threshold_f)``):
      the *exact* gate for a JumpReLU SAE (Gemma-Scope). Also cheap — JumpReLU
      is gated per-feature by a fixed threshold, so no full encode / top-k is
      needed; this is bit-equivalent to the HF Gemma steering hook.

    Both then inject the same clamp_floor delta (``clamp(clamp_value − feat, 0)``).

    ``set_residual`` selects the residual convention:

    * ``False`` — the block output *is* ``resid_post`` (plain HF / the tests):
      ``x`` is ``output`` (or ``output[0]`` if the block returns a tuple) and the
      delta is added to it.
    * ``True`` — the block returns vLLM's ``(hidden_states, residual)`` pair:
      ``x = hidden + residual`` is the true ``resid_post`` used to encode, and
      the delta is added back into ``hidden`` so the next block's input norm sees
      ``resid_post + delta``.
    """
    cv = float(spec.clamp_value)
    sdt = w.W_enc_f.dtype
    W_enc_f, b_enc_f, W_dec_f = w.W_enc_f, w.b_enc_f, w.W_dec_f
    # Gate threshold: a (nf,) JumpReLU tensor (Gemma-Scope) or 0.0 → relu
    # (Qwen-Scope fast). Broadcasts over the trailing nf axis of pre_f.
    thr = w.thr_f if w.thr_f is not None else 0.0

    def _resid_and_pack(output):
        """Return (x, repack): x is resid_post; repack(inj) rebuilds the output
        with the injection added into the residual stream."""
        if isinstance(output, tuple):
            hidden = output[0]
            if set_residual and len(output) >= 2 and torch.is_tensor(output[1]):
                resid_post = hidden + output[1]
            else:
                resid_post = hidden
            return resid_post, lambda inj: (hidden + inj, *output[1:])
        return output, lambda inj: output + inj

    def hook(_module, _input, output):
        x, repack = _resid_and_pack(output)
        pre_f = torch.matmul(x.to(sdt), W_enc_f.T) + b_enc_f  # (..., nf)
        feat = pre_f * (pre_f > thr)  # JumpReLU (thr=threshold) / relu (thr=0)
        delta = (cv - feat).clamp(min=0)  # clamp_floor: raise below-clamp up to cv
        inj = torch.matmul(delta, W_dec_f).to(x.dtype)  # (..., d_model)
        return repack(inj)

    return hook


# ── SAE loading (runs inside the vLLM worker) ───────────────────────────────


def _hub_download(repo_id: str, filename: str, cache_dir: str | None) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=cache_dir)


def _load_weights_topk(source: SaeSource, spec: SteerSpec, device: torch.device) -> _SteerWeights:
    """Qwen-Scope TopK ``.pt``: W_enc (d_sae, d_model), b_enc, W_dec (d_model, d_sae).

    The full ``W_enc`` (~1.6 GB fp32) is read to CPU and sliced; only the tiny
    ``(nf, d_model)`` feature rows move to GPU. Gate is relu (thr_f=None).
    """
    import os

    path = source.path
    if not path:
        ell = source.layer if source.layer is not None else spec.layer
        cache_dir = source.cache_dir or os.environ.get("QWEN_SCOPE_CACHE")
        path = _hub_download(source.repo_id, f"layer{ell}.sae.pt", cache_dir)
    state = torch.load(path, map_location="cpu", weights_only=True)
    dtype = _DTYPES[source.dtype]
    idx = torch.tensor(spec.feature_ids(), dtype=torch.long)

    W_enc_f = state["W_enc"][idx].to(device=device, dtype=dtype).contiguous()  # (nf, d_model)
    b_enc_f = state["b_enc"][idx].to(device=device, dtype=dtype).contiguous()  # (nf,)
    # W_dec is (d_model, d_sae) on disk → feature-major (nf, d_model).
    W_dec_f = state["W_dec"][:, idx].T.to(device=device, dtype=dtype).contiguous()
    return _SteerWeights(W_enc_f=W_enc_f, b_enc_f=b_enc_f, W_dec_f=W_dec_f, thr_f=None)


def _load_weights_jumprelu(
    source: SaeSource, spec: SteerSpec, device: torch.device
) -> _SteerWeights:
    """Gemma-Scope-2 JumpReLU ``params.safetensors``: w_enc (d_model, d_sae),
    b_enc, threshold (d_sae,), w_dec (d_sae, d_model).

    Only the target feature rows move to GPU; ``thr_f`` carries the per-feature
    JumpReLU thresholds so the hook gates exactly like the HF Gemma steering.
    """
    from safetensors import safe_open

    path = source.path
    if not path:
        ell = source.layer if source.layer is not None else spec.layer
        base = f"resid_post/layer_{ell}_width_{source.width}_l0_{source.l0_size}"
        path = _hub_download(source.repo_id, f"{base}/params.safetensors", source.cache_dir)
    dtype = _DTYPES[source.dtype]
    idx = torch.tensor(spec.feature_ids(), dtype=torch.long)
    with safe_open(path, framework="pt") as f:
        # w_enc is (d_model, d_sae) on disk → encoder rows (nf, d_model).
        W_enc_f = f.get_tensor("w_enc")[:, idx].T.to(device=device, dtype=dtype).contiguous()
        b_enc_f = f.get_tensor("b_enc")[idx].to(device=device, dtype=dtype).contiguous()
        thr_f = f.get_tensor("threshold")[idx].to(device=device, dtype=dtype).contiguous()
        # w_dec is (d_sae, d_model) on disk → already feature-major (nf, d_model).
        W_dec_f = f.get_tensor("w_dec")[idx].to(device=device, dtype=dtype).contiguous()
    return _SteerWeights(W_enc_f=W_enc_f, b_enc_f=b_enc_f, W_dec_f=W_dec_f, thr_f=thr_f)


def _load_weights(source: SaeSource, spec: SteerSpec, device: torch.device) -> _SteerWeights:
    """Load only the target feature rows onto ``device`` (dispatch by family)."""
    if source.family == "jumprelu":
        return _load_weights_jumprelu(source, spec, device)
    return _load_weights_topk(source, spec, device)


# ── vLLM install / remove (shipped to the worker via apply_model) ────────────


def _decoder_layers(model):
    """Locate the decoder block list across the common vLLM/HF layouts."""
    candidates = [
        ("model", "layers"),  # Qwen3 / Llama (vLLM and HF base)
        ("model", "language_model", "layers"),
        ("language_model", "model", "layers"),
        ("language_model", "layers"),
        ("transformer", "h"),
    ]
    for path in candidates:
        obj = model
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        else:
            return obj
    raise RuntimeError(f"cannot locate decoder layers on {type(model).__name__}")


def _install_on_model(model, pairs: list[tuple[SaeSource, SteerSpec]]) -> int:
    """Worker-side: load each SAE and register its steering hook. Returns count."""
    layers = _decoder_layers(model)
    handles = list(getattr(model, _HANDLES_ATTR, []))
    for source, spec in pairs:
        if not (0 <= spec.layer < len(layers)):
            raise IndexError(f"layer {spec.layer} out of range for {len(layers)} blocks")
        block = layers[spec.layer]
        device = next(block.parameters()).device
        w = _load_weights(source, spec, device)
        handles.append(block.register_forward_hook(make_steer_hook(w, spec, set_residual=True)))
    setattr(model, _HANDLES_ATTR, handles)
    return len(handles)


def _remove_from_model(model) -> int:
    handles = getattr(model, _HANDLES_ATTR, [])
    for h in handles:
        h.remove()
    setattr(model, _HANDLES_ATTR, [])
    return len(handles)


def _require_eager(llm: LLM) -> None:
    """Post-build hooks are skipped under compiled/captured execution — a
    non-eager engine would serve silently UNSTEERED, so refuse outright."""
    try:
        enforce_eager = bool(
            getattr(llm.llm_engine.vllm_config.model_config, "enforce_eager", False)
        )
    except AttributeError:  # vllm config layout drift — best-effort check only
        return
    if not enforce_eager:
        raise RuntimeError(
            "install_steering on a non-eager engine: forward hooks do not fire "
            "under vLLM's compiled/captured execution, so steering would be "
            "silently skipped. Build with steer_precapture=True (full-speed, "
            "fixed clamps) or enforce_eager=True (post-build hooks, e.g. for "
            "in-process clamp sweeps)."
        )


def install_steering(llm: LLM, pairs: list[tuple[SaeSource, SteerSpec]]) -> None:
    """Install SAE steering hooks on P inside the vLLM worker(s).

    Each ``(SaeSource, SteerSpec)`` pair adds one hook at ``spec.layer``; several
    features at one layer go in a single spec (``feature_id=[a, b, c]``), several
    layers as several pairs. Effects compose additively. Hooks are GLOBAL — every
    sequence in every batch is steered identically (one steering config per run,
    like ``run_steer.py``), independent of any per-request ``dd_alpha``.

    Requires ``enforce_eager=True`` on the ``LLM`` (raises otherwise). vLLM's V1
    engine core runs in a separate process, so the install callable is shipped
    over an RPC whose safe serializer rejects callables — set
    ``VLLM_ALLOW_INSECURE_SERIALIZATION=1`` (pickle fallback) *before*
    constructing the ``LLM``/engine.
    """
    if not pairs:
        return
    _require_eager(llm)
    # A top-level function bound via partial (not a lambda): apply_model pickles
    # the callable to the worker under the multiproc executor (TP>1), where a
    # lambda would not survive.
    llm.apply_model(functools.partial(_install_on_model, pairs=pairs))


def remove_steering(llm: LLM) -> None:
    """Remove all steering hooks previously installed on P."""
    llm.apply_model(_remove_from_model)


@contextmanager
def steered_vllm(llm: LLM, pairs: list[tuple[SaeSource, SteerSpec]]):
    """Scope steering to a block::

        with steered_vllm(llm, [(src, spec)]):
            out = llm.generate(prompts, sampling_params)

    Hooks are installed on entry and removed on exit (even on error).
    """
    install_steering(llm, pairs)
    try:
        yield
    finally:
        remove_steering(llm)
