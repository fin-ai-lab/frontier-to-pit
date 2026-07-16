"""Central vLLM runner: build P with Divergence Decoding + SAE steering once,
for both batch evals (offline ``LLM``) and streaming chat (``AsyncLLM``).

This is the single place that wires the two interventions onto a vLLM engine:
  * DD — ``make_processor(DDConfig(...))`` as a logits processor;
  * steering — ``install_steering``/``apply_model`` forward hooks on P.

Both force ``enforce_eager=True`` when steering (hooks don't fire under CUDA
graphs) and set ``VLLM_ALLOW_INSECURE_SERIALIZATION=1`` (the steering install
ships a callable to the engine-core process). Not imported from the package
``__init__`` so the core package stays importable without vLLM.
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass

from ftp.config import DDConfig
from ftp.guard import GuardConfig
from ftp.steering import SaeSource, SteerSpec, _install_on_model
from ftp.vllm import DDLogitsProcessor, GuardLogitsProcessor


@dataclass
class SteerArgs:
    """SAE source + the ``L:feature:clamp`` triples for steering.

    ``family`` selects the SAE / gate: ``"topk"`` (Qwen-Scope, relu gate, the
    default) or ``"jumprelu"`` (Gemma-Scope-2, exact per-feature threshold gate).
    ``width``/``l0_size`` apply to jumprelu only (Gemma-Scope bucket).
    """

    triples: list[tuple[int, int, float]]
    sae_repo: str | None = None
    sae_cache: str | None = None
    sae_dir: str | None = None  # topk only: local dir of layer{L}.sae.pt (overrides repo)
    family: str = "topk"
    width: str = "65k"
    l0_size: str = "medium"

    def pairs(self) -> list[tuple[SaeSource, SteerSpec]]:
        out = []
        for layer, feat, val in self.triples:
            if self.sae_dir and self.family == "topk":
                src = SaeSource(path=f"{self.sae_dir}/layer{layer}.sae.pt", family="topk")
            else:
                src = SaeSource(
                    repo_id=self.sae_repo, layer=layer, cache_dir=self.sae_cache,
                    family=self.family, width=self.width, l0_size=self.l0_size,
                )
            out.append((src, SteerSpec(layer=layer, feature_id=feat, clamp_value=val)))
        return out


def parse_steer(spec: str) -> list[tuple[int, int, float]]:
    """``"48:28961:35,27:24365:20"`` -> ``[(48, 28961, 35.0), (27, 24365, 20.0)]``."""
    out = []
    for tri in (t.strip() for t in (spec or "").split(",")):
        if not tri:
            continue
        layer, feat, val = tri.split(":")
        out.append((int(layer), int(feat), float(val)))
    return out


def _dd_config(model: str, aux_p: str, aux_q: str, **kw) -> DDConfig:
    return DDConfig(aux_p=aux_p, aux_q=aux_q, tokenizer=model, **kw)


def _resolve_guard(guard, dd_cfg) -> GuardConfig | None:
    """Resolve the ``guard`` argument of the builders.

    ``None`` = AUTO (the default): the live degeneration guard is ON whenever
    DD is configured — a strong DD push is exactly what produces the decoding
    collapse the guard repairs, and its steady-state cost is ~free (one gated
    judge sweep per ``interval`` engine steps). ``False`` disables it
    explicitly (A/B baselines, benches); ``True`` or a :class:`GuardConfig`
    force it on.
    """
    if guard is None:
        return GuardConfig() if dd_cfg is not None else None
    if guard is False:
        return None
    if guard is True:
        return GuardConfig()
    return guard


def steer_precapture_kwargs(steer: SteerArgs) -> dict:
    """Engine kwargs (+ env, mutated in place) for the PRE-CAPTURE steering
    route: ``DDSteeringWorker`` installs the hooks before vLLM's memory
    profiling / dynamo tracing / graph capture, so steered serving keeps full
    cudagraphs. Config travels by ``DD_STEER_*`` env because vLLM instantiates
    the worker by import path inside the engine-core process.

    ``VLLM_DISABLE_COMPILE_CACHE=1`` is load-bearing: vLLM's compile-cache
    hash EXCLUDES worker_cls and dynamo skips nn-module hook guards, so a warm
    artifact from an unsteered (or different-clamp) build would replay WITHOUT
    this run's steering, silently (measured). setdefault: only an explicit
    ``0`` in the environment overrides, at the caller's own risk.

    Shared by :func:`build_llm`/:func:`build_async_llm` and the lmeval ``dd``
    backend (which constructs its engine through lm-eval, not through here).
    """
    os.environ.setdefault("VLLM_DISABLE_COMPILE_CACHE", "1")
    if os.environ["VLLM_DISABLE_COMPILE_CACHE"] != "1":
        print(
            "[steer_precapture] WARNING: VLLM_DISABLE_COMPILE_CACHE="
            f"{os.environ['VLLM_DISABLE_COMPILE_CACHE']!r} (not '1') — a warm compile "
            "artifact from an unsteered build can be reused and serve SILENTLY "
            "UNSTEERED (vLLM's cache hash excludes worker_cls). Unset it to be safe.",
            flush=True,
        )
    os.environ["DD_STEER_TRIPLES"] = ",".join(
        f"{L}:{f}:{v:g}" for L, f, v in steer.triples)
    os.environ["DD_STEER_FAMILY"] = steer.family
    os.environ["DD_STEER_SAE_DIR"] = steer.sae_dir or ""
    os.environ["DD_STEER_SAE_REPO"] = steer.sae_repo or ""
    os.environ["DD_STEER_SAE_CACHE"] = steer.sae_cache or ""
    os.environ["DD_STEER_WIDTH"] = steer.width
    os.environ["DD_STEER_L0"] = steer.l0_size
    return {"worker_cls": "ftp.steering_worker.DDSteeringWorker"}


def _common_kwargs(model, *, dd_cfg, steering, tp, gpu_mem, max_len,
                   steer_precapture=None, gdn_prefill_backend=None, force_eager=False,
                   guard_cfg=None):
    """Engine args shared by the offline and async builders.

    Steering has two install routes:
      * post-build hooks (default): shipped to the workers by RPC AFTER the
        engine is built — after CUDA-graph capture — so P must run
        ``enforce_eager`` or the hooks silently never fire;
      * PRE-CAPTURE (``steer_precapture``): a custom worker class installs the
        hooks right after model load, so the hook kernels (two rank-1 matmuls
        + clamp + add on fixed pointers) are recorded INSIDE the captured
        graphs — full cudagraph speed with steering. Clamp values become
        capture-time constants (one build per steering config).
    """
    if steering and not steer_precapture:
        # The steering install ships a callable to the engine-core process; vLLM's
        # safe RPC serializer rejects callables, so enable the pickle fallback.
        # Must be set before the engine starts.
        os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    kw = dict(
        model=model,
        # steering hooks fire only in eager mode UNLESS installed pre-capture;
        # force_eager (fast-startup) additionally skips torch.compile + graph capture.
        enforce_eager=(bool(steering) and not steer_precapture) or force_eager,
        tensor_parallel_size=tp,
        gpu_memory_utilization=gpu_mem,
        max_model_len=max_len,
        enable_prefix_caching=False,
    )
    if gdn_prefill_backend is not None:
        # "triton" skips the flashinfer GDN nvcc JIT (~15-20 min first run) at a small
        # per-token cost — good for a quick try. Default (None) = vLLM auto = flashinfer.
        kw["gdn_prefill_backend"] = gdn_prefill_backend
    if steer_precapture is not None:
        kw.update(steer_precapture_kwargs(steer_precapture))
    procs = []
    if dd_cfg is not None:
        # Ship the module-level (picklable) processor class + config via DD_* env.
        # vLLM spawns the engine-core whenever CUDA is already initialized (e.g.
        # FP8 device-capability probing initializes it before the fork point), and
        # make_processor's dynamically created subclass pickles under an
        # unimportable name ("abc.ConfiguredDDLogitsProcessor") and dies under
        # spawn. The base class resolves config from env via DDConfig.from_env in
        # the worker — surviving both fork and spawn; the round-trip is lossless
        # (config.to_env/from_env cover every field). env is inherited by spawn.
        dd_cfg.apply_env()
        procs.append(DDLogitsProcessor)
    if guard_cfg is not None:
        # Same env route (DD_GUARD_*). Guard runs LAST so a trip's forced marker
        # overrides the DD fusion and steering for that row.
        guard_cfg.apply_env()
        procs.append(GuardLogitsProcessor)
    if procs:
        kw["logits_processors"] = procs
    return kw


def build_llm(
    model: str,
    *,
    aux_p: str | None = None,
    aux_q: str | None = None,
    dd_kwargs: dict | None = None,
    steer: SteerArgs | None = None,
    steer_precapture: bool = True,
    guard: GuardConfig | bool | None = None,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.90,
    max_model_len: int = 4096,
    max_num_seqs: int = 256,
    gdn_prefill_backend: str | None = None,
    **llm_kwargs,
):
    """Offline batch ``LLM`` with DD (if aux given) and steering (if any) installed.

    Steering defaults to the PRE-CAPTURE route (``steer_precapture=True``):
    hooks are installed before vLLM's graph capture via ``DDSteeringWorker``,
    keeping full cudagraphs — steered throughput equals unsteered (see
    ``_common_kwargs``). The trade: clamp values are capture-time constants.
    Pass ``steer_precapture=False`` ONLY when the hooks must change in-process
    after build (e.g. clamp sweeps via remove+reinstall) — that route forces
    ``enforce_eager`` and roughly halves throughput.

    ``guard`` controls the live degeneration guard (:mod:`ftp.guard`).
    Default ``None`` = AUTO: the guard is ON whenever DD is configured (see
    :func:`_resolve_guard`); pass ``False`` to disable it explicitly, or a
    :class:`GuardConfig` to customize. The engine side is the DETECTION half
    only — pair generation with :func:`ftp.guard.rollback_generate` (and its
    marker in ``stop_token_ids``) to get the rewind-and-resample behavior."""
    from vllm import LLM

    from ftp.steering import install_steering

    dd_cfg = _dd_config(model, aux_p, aux_q, **(dd_kwargs or {})) if aux_p and aux_q else None
    pairs = steer.pairs() if steer else []
    kw = _common_kwargs(
        model, dd_cfg=dd_cfg, steering=bool(pairs),
        tp=tensor_parallel_size, gpu_mem=gpu_memory_utilization, max_len=max_model_len,
        steer_precapture=steer if (pairs and steer_precapture) else None,
        gdn_prefill_backend=gdn_prefill_backend,
        guard_cfg=_resolve_guard(guard, dd_cfg),
    )
    kw["max_num_seqs"] = max_num_seqs
    if pairs and steer_precapture and ({"worker_cls", "enforce_eager"} & llm_kwargs.keys()):
        raise ValueError(
            "llm_kwargs sets worker_cls/enforce_eager on a steer_precapture build — "
            "that clobbers DDSteeringWorker and serves silently unsteered; drop them "
            "or pass steer_precapture=False for the eager post-build route")
    kw.update(llm_kwargs)
    llm = LLM(**kw)
    if pairs and not steer_precapture:
        install_steering(llm, pairs)
    return llm, dd_cfg


def build_async_llm(
    model: str,
    *,
    aux_p: str | None = None,
    aux_q: str | None = None,
    dd_kwargs: dict | None = None,
    steer: SteerArgs | None = None,
    steer_precapture: bool = True,
    guard: GuardConfig | bool | None = None,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.90,
    max_model_len: int = 4096,
    enforce_eager: bool = False,
    gdn_prefill_backend: str | None = None,
):
    """Streaming ``AsyncLLM`` with DD installed.

    Steering defaults to the PRE-CAPTURE route (worker installs the hooks
    before graph capture — full cudagraphs; see ``build_llm``); ``pairs``
    comes back empty then, so an unconditional ``install_steering_async``
    call is a no-op instead of a double install. With
    ``steer_precapture=False`` the engine is built ``enforce_eager`` and the
    caller installs the returned ``pairs`` with
    ``install_steering_async(engine, pairs)`` inside a running event loop.

    ``enforce_eager=True`` is the fast-startup path: it skips torch.compile +
    CUDA-graph capture (~1-3 min) so the engine is ready in roughly model-load
    time, at ~2x per-token cost. The pre-capture route needs those graphs, so
    when steering is on this transparently falls back to the eager post-build
    hook route (``pairs`` comes back non-empty for the caller to install).
    """
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM

    dd_cfg = _dd_config(model, aux_p, aux_q, **(dd_kwargs or {})) if aux_p and aux_q else None
    pairs = steer.pairs() if steer else []
    # enforce_eager and the pre-capture route are mutually exclusive (no graphs to
    # bake hooks into) — drop to the post-build eager install when both are asked for.
    use_precapture = bool(pairs) and steer_precapture and not enforce_eager
    kw = _common_kwargs(
        model, dd_cfg=dd_cfg, steering=bool(pairs),
        tp=tensor_parallel_size, gpu_mem=gpu_memory_utilization, max_len=max_model_len,
        steer_precapture=steer if use_precapture else None,
        gdn_prefill_backend=gdn_prefill_backend,
        force_eager=enforce_eager,
        guard_cfg=_resolve_guard(guard, dd_cfg),
    )
    engine = AsyncLLM.from_engine_args(AsyncEngineArgs(**kw))
    return engine, dd_cfg, [] if use_precapture else pairs


async def install_steering_async(engine, pairs) -> None:
    """Install steering hooks on an ``AsyncLLM`` (worker apply_model via RPC)."""
    if not pairs:
        return
    await engine.collective_rpc(
        "apply_model", args=(functools.partial(_install_on_model, pairs=pairs),)
    )


async def stream(engine, prompt: str, sampling_params, *, request_id: str):
    """Yield incremental text deltas for one request from an ``AsyncLLM``."""
    printed = ""
    async for out in engine.generate(prompt, sampling_params, request_id=request_id):
        text = out.outputs[0].text
        yield text[len(printed):]
        printed = text
