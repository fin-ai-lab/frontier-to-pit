"""Divergence Decoding as a vLLM v1 logits processor.

DD formula: l̂ = l_P + α·(l_q − l_p)

Two auxiliary models run *inside* vLLM's engine core; every decode step their
forwards are batched into one call per model, and the fused distribution is
written back into vLLM's logits before sampling.

Library usage (preferred)::

    from ftp import DDConfig
    from ftp.vllm import make_processor

    cfg = DDConfig(aux_p="fin-ai-lab/aux-2024", aux_q="fin-ai-lab/aux-2015")
    llm = LLM(model=..., logits_processors=[make_processor(cfg)],
              enable_prefix_caching=False)
    out = llm.generate(prompts, SamplingParams(..., extra_args={"dd_alpha": 1.5}))

``vllm serve`` usage (environment fallback)::

    DD_AUX_P=... DD_AUX_Q=... vllm serve <model> \\
        --logits-processors ftp.vllm:DDLogitsProcessor

Per-request via ``SamplingParams.extra_args``:
  ``dd_alpha`` (float) — DD strength; ``0.0`` = pure-P baseline with zero aux
  overhead. Defaults to ``DDConfig.alpha_default``.
  ``dd_rank_k`` (int, default 0 = off) — rank-based DD: additionally suppress the
  ``dd_rank_k`` tokens most divergent toward the forget aux (top-k of
  ``l_p − l_q``), all others unaffected. Orthogonal to ``dd_alpha`` and composes
  with it: ``dd_alpha=0`` + ``dd_rank_k>0`` = pure rank masking; both nonzero =
  linear + rank together.

Set ``DD_TIMING=1`` to print per-step aux/fusion latency percentiles.
Whitelist pinning (EOS/control tokens kept at P's probability) is ON by default;
disable it with ``DDConfig(fuse_pin=False)`` (or ``DD_FUSE_PIN=0`` for ``vllm serve``).
"""

from __future__ import annotations

import os
import threading
import time
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import torch
from transformers import AutoConfig, AutoTokenizer
from vllm import SamplingParams
from vllm.v1.sample.logits_processor import AdapterLogitsProcessor
from vllm.v1.sample.logits_processor.interface import BatchUpdate

from ftp.config import DDConfig
from ftp.core import build_special_ids, dd_fuse
from ftp.engine import AuxBatchedEngine
from ftp.paired import check_fusable
from ftp.translate import (
    TokenTextTable,
    UniversalBridge,
    VocabMapper,
    make_chat_adapter,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def make_processor(
    cfg: DDConfig, *, name: str = "ConfiguredDDLogitsProcessor"
) -> type[DDLogitsProcessor]:
    """Bake a :class:`DDConfig` into a logits-processor class for ``LLM(...)``.

    Returns a subclass with ``dd_config = cfg`` for
    ``LLM(logits_processors=[make_processor(cfg)])``. The config is *also*
    written to ``os.environ`` so that engine-core subprocesses started with the
    ``spawn`` method (where a dynamically created class may not survive
    pickling by reference) transparently fall back to the environment route.
    """
    cfg.apply_env()
    return type(name, (DDLogitsProcessor,), {"dd_config": cfg})


class _CudaTimers:
    """CUDA-event section timers (``DD_TIMING=1``). Events are collected without
    synchronizing; every ``every`` steps we sync once, print mean/p50/p99 per
    section, and reset — so timing doesn't perturb the pipeline."""

    def __init__(self, names: list[str], every: int = 200) -> None:
        self._names = names
        self._every = every
        self._pairs: dict[str, list[list[torch.cuda.Event]]] = {n: [] for n in names}
        self._steps = 0

    def start(self, name: str) -> None:
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        self._pairs[name].append([ev, ev])

    def stop(self, name: str) -> None:
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        self._pairs[name][-1][1] = ev

    def step(self) -> None:
        self._steps += 1
        if self._steps % self._every:
            return
        torch.cuda.synchronize()
        parts = []
        for n in self._names:
            ms = sorted(s.elapsed_time(e) for s, e in self._pairs[n])
            self._pairs[n] = []
            if not ms:
                continue
            mean = sum(ms) / len(ms)
            p50 = ms[len(ms) // 2]
            p99 = ms[min(len(ms) - 1, int(len(ms) * 0.99))]
            parts.append(f"{n} mean={mean:.2f} p50={p50:.2f} p99={p99:.2f}")
        print(f"[DD_TIMING step {self._steps}] " + " | ".join(parts) + " (ms)", flush=True)


#: sentinel logits value in a prefetch tuple: "result lives on the aux worker"
_ON_WORKER = object()


class _AuxWorker:
    """One background thread that owns the aux block in overlap mode.

    Why: the aux block's HOST work — the eager prefill launch storm on
    prime/re-prime steps (tens of ms per group), plus ~1-3 ms of staging on
    every decode step — used to run on vLLM's single-threaded engine-core
    loop between ``update_state`` and P's forward launch, fully serialized
    into every step even though the GPU work lands on the aux device's own
    stream. On this thread it runs concurrently with P's forward
    issue/execution; ``apply()`` only waits out whatever P's forward didn't
    already hide (the same contract as GPU-side overlap).

    Threading contract (why this is race-free):

    * vLLM's engine core calls ``update_state(t)`` → P forward → ``apply(t)``
      strictly in order on one thread; ``submit`` happens only in
      ``update_state``, ``result`` only in ``apply``.
    * At most one job is ever in flight: ``submit`` drains an unconsumed
      prior job first (a step whose ``apply`` never ran — empty forwards),
      and ``apply`` blocks on the in-flight job before any inline engine
      work (late rows, misses). The engines are therefore touched by
      exactly one thread at a time.
    * Between ``submit(t)`` and ``result(t)`` the job reads prompt/output id
      lists that vLLM appends to only AFTER ``apply(t)`` (sampler commit),
      so job inputs are immutable while it runs.
    * All aux GPU work stays on the aux device's default stream regardless
      of the issuing thread, so kernel ORDER (and numerics) are identical
      to the inline path; the consuming cross-device copy is enqueued after
      ``result`` returns, i.e. after every aux op of the step.

    Exceptions re-raise in ``result`` on the loop thread. ``DD_AUX_THREAD=0``
    disables the worker (aux block runs inline as before).
    """

    def __init__(self, aux_device: torch.device) -> None:
        self._dev = aux_device
        self._job = None
        self._go = threading.Semaphore(0)
        self._done = threading.Event()
        self._result = None
        self._exc: BaseException | None = None
        self.pending = False
        self.job_ms: list[float] = []  # host duration of each job (diagnostics)
        self._thread = threading.Thread(target=self._run, name="dd-aux", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        if self._dev.type == "cuda":
            torch.cuda.set_device(self._dev)
        # Run every job under inference_mode so this thread's inference state
        # matches vLLM's engine-core loop (the inline path). A new request is
        # registered on the loop thread INSIDE inference_mode
        # (register() -> _grow_slots() allocates the aux page tables as
        # inference tensors); mutating those tensors here — off the loop
        # thread, where inference_mode does NOT carry over — would otherwise
        # raise "Inplace update to inference tensor outside InferenceMode".
        # This keeps the worker bitwise-identical to inline execution.
        with torch.inference_mode():
            while True:
                self._go.acquire()
                t0 = time.perf_counter()
                try:
                    self._result = self._job()
                except BaseException as e:  # noqa: BLE001 — re-raised in result()
                    self._exc = e
                self.job_ms.append((time.perf_counter() - t0) * 1e3)
                self._done.set()

    def submit(self, fn) -> None:
        if self.pending:
            self.result()  # drain a job whose apply() never ran
        self._exc = None
        self._result = None
        self._job = fn
        self._done.clear()
        self.pending = True
        self._go.release()

    def result(self):
        self._done.wait()
        self.pending = False
        if self._exc is not None:
            exc, self._exc = self._exc, None
            raise exc
        return self._result


class _DDReqCtx:
    """Lightweight per-request context (carries the fusion arm: alpha + rank_k).

    The 3-arg __call__ signature causes AdapterLogitsProcessor._new_state to
    pre-bind (prompt_ids, output_ids) into the returned partial, giving
    DDLogitsProcessor.apply() stable references to both.
    """

    __slots__ = ("alpha", "rank_k")

    def __init__(self, alpha: float, rank_k: int = 0) -> None:
        self.alpha = alpha
        self.rank_k = rank_k

    def __call__(
        self,
        prompt_ids: list[int] | None,
        output_ids: list[int],
        logits: torch.Tensor,
    ) -> torch.Tensor:
        # Never called — DDLogitsProcessor.apply() overrides dispatch entirely.
        raise RuntimeError("_DDReqCtx.__call__ must not be invoked directly")


class DDLogitsProcessor(AdapterLogitsProcessor):
    """Divergence Decoding as a vLLM v1 LogitsProcessor.

    Configuration comes from the ``dd_config`` class attribute (set by
    :func:`make_processor`) or, when that is ``None``, from ``DD_*``
    environment variables via :meth:`DDConfig.from_env` — the route used by
    ``vllm serve``.
    """

    dd_config: ClassVar[DDConfig | None] = None

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        is_pin_memory: bool,
    ) -> None:
        super().__init__(vllm_config, device, is_pin_memory)

        cfg = type(self).dd_config or DDConfig.from_env()
        self._cfg = cfg
        dtype = _DTYPES[cfg.dtype]

        # Tensor parallelism: vLLM gathers logits to TP rank 0 (CUDA uses
        # tensor_model_parallel_gather — rank > 0 receives None) and only that
        # rank's sampler ever sees real logits. The processor class is still
        # constructed in EVERY rank's worker, and without this guard each rank
        # loads a full aux copy onto cfg.aux_device (measured: TP=2 put 2x the
        # aux footprint on one card). Non-zero ranks become inert.
        tp_rank, tp_world = 0, 1
        try:
            from vllm.distributed import (
                get_tensor_model_parallel_rank,
                get_tensor_model_parallel_world_size,
            )

            tp_rank = get_tensor_model_parallel_rank()
            tp_world = get_tensor_model_parallel_world_size()
        except Exception:  # noqa: BLE001 — outside a TP worker there is no group
            tp_rank, tp_world = 0, 1
        self._inert = tp_rank > 0
        self._bridge = None
        self._overlap = False
        self._timers = None
        self._prefetch: tuple | None = None
        self._engines_built = False
        if self._inert:
            print(
                f"[DDLogitsProcessor] TP rank {tp_rank}: inert "
                "(aux engines live on the logits-gather rank only)",
                flush=True,
            )
            return

        try:
            if vllm_config.cache_config.enable_prefix_caching:
                warnings.warn(
                    "vLLM prefix caching is enabled. The DD engine handles the "
                    "resulting staggered prefills at full speed, but our "
                    "published benchmarks ran with enable_prefix_caching=False; "
                    "validate throughput for your workload.",
                    stacklevel=1,
                )
        except AttributeError:  # vllm config layout drift — warning is best-effort
            pass

        self._device = device
        self._dtype = dtype
        self._p_tok_path = cfg.tokenizer
        if self._p_tok_path is None:
            try:
                self._p_tok_path = vllm_config.model_config.tokenizer
            except AttributeError:  # vllm config layout drift
                self._p_tok_path = None

        # Under TP, vLLM's memory profiling runs on EVERY card after this
        # constructor; aux allocations resident during profiling reproducibly
        # crashed the co-hosting worker (2xH100 TP=2: native worker death at
        # determine_available_memory regardless of aux size or compile mode).
        # Defer all aux construction to the first real batch — strictly after
        # profiling and vLLM's own graph captures. Costs one-time first-batch
        # latency (engine load + prewarm); TP=1 keeps the eager path (prewarm
        # before profiling = automatic memory accounting).
        if tp_world > 1:
            print(
                "[DDLogitsProcessor] TP detected: deferring aux engine "
                "construction to the first batch (post-profiling)",
                flush=True,
            )
        else:
            self._build_engines()

    def _build_engines(self) -> None:
        cfg = self._cfg
        device, dtype = self._device, self._dtype
        aux_device = torch.device(cfg.aux_device) if cfg.aux_device else device
        if aux_device != device:
            print(
                f"[DDLogitsProcessor] aux engines on {aux_device} (P samples on {device})",
                flush=True,
            )
        self._aux_device = aux_device

        # Fuse resolution: stack the two aux models into ONE forward when
        # their architectures are identical (cfg.fuse_aux: auto|on|off). The
        # two-engine path stays as the reference implementation and the route
        # for heterogeneous pairs (different sizes, tied embeddings).
        fuse = cfg.fuse_aux
        if fuse != "off":
            ok, why = check_fusable(
                AutoConfig.from_pretrained(cfg.aux_p, trust_remote_code=True),
                AutoConfig.from_pretrained(cfg.aux_q, trust_remote_code=True),
            )
            if not ok:
                if fuse == "on":
                    raise ValueError(f"fuse_aux='on' but the aux pair is not fusable: {why}")
                print(
                    f"[DDLogitsProcessor] WARNING: aux pair not fusable ({why}); "
                    "falling back to the two-engine path",
                    flush=True,
                )
                fuse = "off"
        self._fused = fuse != "off"
        if self._fused:
            print(
                f"[DDLogitsProcessor] loading FUSED aux pair from {cfg.aux_p} + {cfg.aux_q}",
                flush=True,
            )
            self._aux = AuxBatchedEngine(
                cfg.aux_p, aux_device, dtype, cfg.window, cfg.compile_aux, model2=cfg.aux_q,
                pool_gb=cfg.aux_kv_gb or None,
            )
            self._aux_engines: tuple[AuxBatchedEngine, ...] = (self._aux,)
        else:
            print(f"[DDLogitsProcessor] loading aux_p from {cfg.aux_p}", flush=True)
            self._aux_p = AuxBatchedEngine(
                cfg.aux_p, aux_device, dtype, cfg.window, cfg.compile_aux,
                pool_gb=(cfg.aux_kv_gb / 2) or None,
            )
            print(f"[DDLogitsProcessor] loading aux_q from {cfg.aux_q}", flush=True)
            self._aux_q = AuxBatchedEngine(
                cfg.aux_q, aux_device, dtype, cfg.window, cfg.compile_aux,
                pool_gb=(cfg.aux_kv_gb / 2) or None,
            )
            self._aux_engines = (self._aux_p, self._aux_q)

        # Prewarm BEFORE vLLM's memory profiling/graph capture: allocates the aux
        # StaticCaches and captures the compiled decode graphs with no
        # concurrent CUDA activity.
        if cfg.prewarm > 0:
            for eng in self._aux_engines:
                eng.prewarm(cfg.prewarm)
        # EXPERIMENTAL (DD_AUX_STREAMS=1 to enable): run the two aux models
        # on separate CUDA streams. Off by default — implicated in corrupted
        # aux outputs under sustained load (degenerate sampling + retokenizer
        # re-prime storms); needs a race investigation before it ships on.
        # Quarantined to the two-engine path: a fused engine is one forward,
        # there is nothing to overlap.
        self._aux_streams: tuple | None = None
        if (
            aux_device.type == "cuda"
            and os.environ.get("DD_AUX_STREAMS", "0") == "1"
            and not self._fused
        ):
            self._aux_streams = (
                torch.cuda.Stream(device=aux_device),
                torch.cuda.Stream(device=aux_device),
            )
        elif os.environ.get("DD_AUX_STREAMS", "0") == "1" and self._fused:
            print("[DDLogitsProcessor] DD_AUX_STREAMS ignored: aux pair is fused", flush=True)
        # Overlap mode: with the aux pair on its OWN GPU, the aux block for
        # step t is enqueued at update_state — before vLLM launches P's
        # forward — so both devices compute concurrently and apply() only
        # waits out whatever the P forward didn't already hide. Falls back to
        # the serial path on any step where last-step tokens aren't host-
        # visible yet (vLLM async scheduling). DD_OVERLAP=0 disables.
        self._overlap = aux_device != device and os.environ.get("DD_OVERLAP", "1") == "1"
        self._prefetch: tuple | None = None
        self._pf_hits = 0
        self._pf_misses = 0
        self._pf_host_ms: list[float] = []
        if self._overlap:
            print("[DDLogitsProcessor] overlap mode: aux prefetch at step start", flush=True)
        self._aux_worker: _AuxWorker | None = None
        print("[DDLogitsProcessor] aux engines ready", flush=True)

        # Tokenizer-mode resolution. "shared": P and the aux pair share one
        # tokenizer and ids pass straight through (the pristine fast path).
        # "universal": P's stream is retokenized for the aux models and their
        # distributions are mapped onto P's vocab. "auto" compares the vocabs.
        mode = cfg.mode
        p_tok_path = self._p_tok_path
        if mode != "shared":
            if p_tok_path is None:
                if mode == "universal":
                    raise ValueError(
                        "mode='universal' needs P's tokenizer path (vLLM model "
                        "config or DDConfig.tokenizer)"
                    )
                mode = "shared"  # auto, but nothing to compare against
            else:
                tok_P = AutoTokenizer.from_pretrained(p_tok_path)
                aux_tok = AutoTokenizer.from_pretrained(cfg.aux_p, trust_remote_code=True)
                if mode == "auto":
                    same = tok_P.get_vocab() == aux_tok.get_vocab()
                    mode = "shared" if same else "universal"
        print(f"[DDLogitsProcessor] tokenizer mode: {mode}", flush=True)

        if mode == "universal":
            tok = tok_P
            table = TokenTextTable(tok_P)
            cache_dir = (
                Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
                / "ftp"
            )
            mapper = VocabMapper(aux_tok, table, device, cache_dir=cache_dir)
            adapter = make_chat_adapter(cfg.retemplate)
            self._bridge = UniversalBridge(
                aux_tok,
                table,
                mapper,
                self._aux_engines[0],
                self._aux_engines[1] if not self._fused else None,
                max_feeds_per_step=cfg.max_feeds_per_step,
                window=cfg.window,
                aux_streams=self._aux_streams,
                adapter=adapter,
            )
            print(
                f"[DDLogitsProcessor] universal decoding: P tokenizer {p_tok_path!r}, "
                f"aux tokenizer {cfg.aux_p!r}, vocab map coverage "
                f"{mapper.coverage.float().mean().item():.1%}"
                + (f", retemplate={cfg.retemplate!r}" if adapter is not None else ""),
                flush=True,
            )
        else:
            tok = AutoTokenizer.from_pretrained(cfg.tokenizer_path)
        wl = build_special_ids(tok)
        self._whitelist = torch.tensor(wl, device=device, dtype=torch.long)
        self._wl_mask: torch.Tensor | None = None  # built on first apply (needs V)
        # Whitelist pinning protects EOS/control tokens from DD distortion (see
        # core.dd_fuse); ON by default, disable with DDConfig(fuse_pin=False).
        self._pin = cfg.fuse_pin
        self._suppress_ids = cfg.resolve_suppress_ids(tok)
        print(
            f"[DDLogitsProcessor] whitelist: {len(wl)} ids, "
            f"suppressed: {self._suppress_ids}, "
            f"fuse_pin={'ON' if self._pin else 'off'}",
            flush=True,
        )

        # Cached batch metadata, rebuilt only when the persistent batch OR the
        # set of progressing requests changes.
        self._batch_dirty = True
        self._aux_reqs: list[tuple[int, list[int] | None, list[int]]] = []
        self._idx_t: torch.Tensor | None = None
        # [((alpha, rank_k), row_positions | None, batch_indices)];
        # row_positions None == all rows share one arm
        self._alpha_groups: list[
            tuple[tuple[float, int], torch.Tensor | None, torch.Tensor]
        ] = []
        self._active_sig: tuple | None = None
        # Per-request output length last consumed by the aux engines. vLLM pauses
        # and recomputes requests under KV pressure; a paused request's output
        # doesn't grow, and stepping its aux caches anyway would both desync them
        # and splinter the batch into many length groups. None == not prefilled.
        self._fed: dict[int, int | None] = {}
        # rids (id(output_ids)) currently registered with the aux engines.
        self._registered: set[int] = set()

        # Aux worker thread: the prefetch job's HOST work (prime launch
        # storms, decode staging) runs off the engine-core loop, concurrent
        # with P's forward issue/execution. Shared-tokenizer path only — the
        # universal bridge keeps its inline prefetch (its finalize
        # interleaves engine work with loop-side state in ways a single-job
        # worker doesn't cover). DD_AUX_THREAD=0 disables.
        if (
            self._overlap
            and self._bridge is None
            and os.environ.get("DD_AUX_THREAD", "1") == "1"
        ):
            self._aux_worker = _AuxWorker(self._aux_device)
            print(
                "[DDLogitsProcessor] aux worker thread: aux host work runs off "
                "the engine loop (DD_AUX_THREAD=0 to disable)",
                flush=True,
            )

        sections = (["translate"] if self._bridge else ["aux"]) + ["aux_wait", "fuse"]
        self._timers = _CudaTimers(sections) if os.environ.get("DD_TIMING", "0") == "1" else None
        self._engines_built = True

    # ── Per-request setup ─────────────────────────────────────────────────────

    def _req_arm(self, params: SamplingParams) -> tuple[float, int]:
        extra = params.extra_args or {}
        alpha = float(extra.get("dd_alpha", self._cfg.alpha_default))
        rank_k = int(extra.get("dd_rank_k", 0))
        return alpha, rank_k

    def new_req_logits_processor(self, params: SamplingParams) -> _DDReqCtx | None:
        alpha, rank_k = self._req_arm(params)
        if alpha == 0.0 and rank_k == 0:  # both adjustments off = pure-P request
            return None
        return _DDReqCtx(alpha, rank_k)

    # ── Batch state management ────────────────────────────────────────────────

    def update_state(self, batch_update: BatchUpdate | None) -> None:
        super().update_state(batch_update)
        if self._inert:
            return
        if not self._engines_built:
            if not batch_update:
                return  # nothing real yet (profiling / empty steps)
            self._build_engines()  # TP: deferred past vLLM's memory profiling
        if self._aux_worker is not None and self._aux_worker.pending:
            # A job whose apply() never ran (empty forward): wait it out
            # BEFORE the registration reconcile below mutates engine state
            # the job may still be reading.
            self._aux_worker.result()
        if batch_update:
            self._batch_dirty = True

            # Reconcile engine registrations against req_info (the ground
            # truth) rather than interpreting added/removed/moved semantics:
            # requests can leave the persistent batch via index reuse without
            # appearing in `removed`, and a missed unregister leaks a KV slot —
            # the cache then grows past the prewarmed batch and forces graph
            # recaptures.
            current = {id(p.args[1]) for p in self.req_info.values()}
            for rid in self._registered - current:
                for eng in self._aux_engines:
                    eng.unregister(rid)
                self._fed.pop(rid, None)
                if self._bridge:
                    self._bridge.drop(rid)
            for rid in current - self._registered:
                for eng in self._aux_engines:
                    eng.register(rid)
                self._fed[rid] = None  # None == not yet prefilled
            self._registered = current
        if self._overlap:
            self._launch_prefetch()

    # ── Batched apply ─────────────────────────────────────────────────────────

    def _collect_rows(self):
        """Rows whose output progressed since the aux engines last saw them
        (pure — no state changes).

        Paused requests (vLLM recompute-preemption under KV pressure: output
        unchanged) are excluded — their aux caches stay frozen and resume
        seamlessly when the request progresses again. Rows whose last token is
        a -1 placeholder (async scheduling: not yet materialized on host) are
        treated as not-yet-progressed. Returns (rows, resets): resets are
        requests that jumped/shrunk and need re-registration."""
        rows, resets = [], []
        for batch_idx, partial_fn in self.req_info.items():
            prompt_ids = partial_fn.args[0]
            output_ids = partial_fn.args[1]
            rid = id(output_ids)
            n = len(output_ids)
            fed = self._fed.get(rid)
            if fed is not None:
                if n == fed:  # paused — no new token for the aux engines
                    continue
                if output_ids and output_ids[-1] == -1:
                    continue  # placeholder; collected later once materialized
                if n > fed + 1 or n < fed:
                    # Gap (vLLM advanced this request without us seeing it) or
                    # a SHRUNK output — the latter means a new request reused
                    # the freed output_ids address of an old one (id
                    # collision). Either way: re-register, forcing a full
                    # re-prefill from the request's complete context.
                    resets.append(rid)
            ctx = partial_fn.func
            rows.append((batch_idx, prompt_ids, output_ids, (ctx.alpha, ctx.rank_k)))
        return rows, resets

    def _commit_rows(self, rows, resets) -> None:
        """Apply the state changes for rows about to be fed to the engines."""
        for rid in resets:
            for eng in self._aux_engines:
                eng.unregister(rid)
                eng.register(rid)
            if self._bridge:
                self._bridge.mark_reset(rid)
        for _, _, output_ids, _ in rows:
            self._fed[id(output_ids)] = len(output_ids)

    def _run_aux(self, aux_reqs):
        """Enqueue the aux work for `aux_reqs`; returns raw (l_p, l_q) outputs
        (engine dtype, aux device). A fused pair is ONE forward returning both
        planes; an unfused pair optionally overlaps p and q on streams, except
        when a prefill is present (eager activation peaks — see translate.py)."""
        if self._fused:
            out = self._aux.step(aux_reqs)
            return out[0], out[1]
        has_prefill = any(not self._aux_p._states[rid].primed for rid, _, _ in aux_reqs)
        if self._aux_streams is not None and not has_prefill:
            s1, s2 = self._aux_streams
            cur = torch.cuda.current_stream(self._aux_device)
            s1.wait_stream(cur)
            s2.wait_stream(cur)
            with torch.cuda.stream(s1):
                l_p = self._aux_p.step(aux_reqs)
            with torch.cuda.stream(s2):
                l_q = self._aux_q.step(aux_reqs)
            cur.wait_stream(s1)
            cur.wait_stream(s2)
        else:
            if self._aux_streams is not None:
                # keep prefill work off the default stream so it still
                # overlaps P's forward in prefetch mode, but p and q serial
                s1 = self._aux_streams[0]
                s1.wait_stream(torch.cuda.current_stream(self._aux_device))
                with torch.cuda.stream(s1):
                    l_p = self._aux_p.step(aux_reqs)
                    l_q = self._aux_q.step(aux_reqs)
                torch.cuda.current_stream(self._aux_device).wait_stream(s1)
            else:
                l_p = self._aux_p.step(aux_reqs)
                l_q = self._aux_q.step(aux_reqs)
        return l_p, l_q

    def _launch_prefetch(self) -> None:
        """Overlap mode: enqueue this step's aux block before P's forward.

        All-GPU work lands on the aux device's streams; nothing touches P's
        stream. apply() consumes the result, computing any late-arriving rows
        (async-scheduling stragglers) serially and merging."""
        t0 = time.perf_counter()
        self._prefetch = None
        if not self.req_info:
            return
        rows, resets = self._collect_rows()
        if not rows:
            return
        self._commit_rows(rows, resets)
        aux_reqs = [(id(o), p, o) for _, p, o, _ in rows]
        if self._bridge is None:
            if self._aux_worker is not None:
                self._aux_worker.submit(lambda: self._run_aux(aux_reqs))
                self._prefetch = (rows, _ON_WORKER, None)
            else:
                l_p, l_q = self._run_aux(aux_reqs)
                self._prefetch = (rows, l_p, l_q)
        else:
            self._bridge.prefetch(aux_reqs)
            self._prefetch = (rows, None, None)
        # Host time spent enqueueing — on the critical path under sync
        # scheduling; printed with the DD_TIMING blocks.
        self._pf_host_ms.append((time.perf_counter() - t0) * 1e3)
        if len(self._pf_host_ms) == 500:
            ms = sorted(self._pf_host_ms)
            extra = ""
            if self._aux_worker is not None and self._aux_worker.job_ms:
                jm = sorted(self._aux_worker.job_ms)
                self._aux_worker.job_ms = []
                p99 = jm[min(len(jm) - 1, int(len(jm) * 0.99))]
                extra = (
                    f" | worker job ms: mean={sum(jm) / len(jm):.2f} "
                    f"p50={jm[len(jm) // 2]:.2f} p99={p99:.2f}"
                )
            print(
                f"[DDLogitsProcessor] prefetch host ms: mean="
                f"{sum(ms) / len(ms):.2f} p50={ms[250]:.2f} p99={ms[494]:.2f}" + extra,
                flush=True,
            )
            self._pf_host_ms.clear()

    def _rebuild_batch_meta(self, rows, device: torch.device) -> None:
        """Rebuild cached request order / index tensors / alpha groups.

        Only runs when the persistent batch or the progressing-row set changes;
        steady-state decode steps reuse the cached tensors and allocate nothing."""
        self._aux_reqs = [(id(o), p, o) for _, p, o, _ in rows]
        self._idx_t = torch.tensor([r[0] for r in rows], dtype=torch.long, device=device)

        by_arm: dict[tuple[float, int], list[int]] = {}
        for pos, (_, _, _, arm) in enumerate(rows):
            by_arm.setdefault(arm, []).append(pos)
        if len(by_arm) == 1:
            self._alpha_groups = [(next(iter(by_arm)), None, self._idx_t)]
        else:
            self._alpha_groups = []
            for arm, poss in by_arm.items():
                pos_t = torch.tensor(poss, dtype=torch.long, device=device)
                self._alpha_groups.append((arm, pos_t, self._idx_t.index_select(0, pos_t)))
        self._batch_dirty = False

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        """Fuse the aux shift into vLLM's logits (aux forwards here too,
        unless overlap mode already ran them during P's forward)."""
        if self._inert or not self._engines_built or not self.req_info:
            return logits

        pf, self._prefetch = self._prefetch, None
        late, resets = self._collect_rows()
        self._commit_rows(late, resets)
        rows = (pf[0] + late) if pf is not None else late
        if not rows:
            return logits
        sig = tuple(r[0] for r in rows)
        if self._batch_dirty or sig != self._active_sig:
            self._rebuild_batch_meta(rows, logits.device)
            self._active_sig = sig
        V = logits.shape[-1]
        if self._wl_mask is None or self._wl_mask.shape[0] < V:
            wl_mask = torch.zeros(V, dtype=torch.bool, device=logits.device)
            wl_mask[self._whitelist[self._whitelist < V]] = True
            self._wl_mask = wl_mask

        tm = self._timers
        if pf is not None:
            # Overlap hit: the aux block ran concurrently with P's forward.
            # aux_wait measures only the EXPOSED remainder (stragglers, the
            # cross-device hop, and any aux tail P's forward didn't cover).
            self._pf_hits += 1
            if tm:
                tm.start("aux_wait")
            if self._bridge is None:
                l_p, l_q = pf[1], pf[2]
                if l_p is _ON_WORKER:
                    # Blocks until the worker's job is done — after this the
                    # worker is idle, so the inline engine work below (late
                    # rows) is single-threaded again.
                    l_p, l_q = self._aux_worker.result()
                if late:
                    lp2, lq2 = self._run_aux([(id(o), p, o) for _, p, o, _ in late])
                    l_p = torch.cat([l_p, lp2])
                    l_q = torch.cat([l_q, lq2])
                if l_p.device != logits.device:
                    l_p = l_p.to(logits.device)
                    l_q = l_q.to(logits.device)
            else:
                if late:
                    self._bridge.prefetch([(id(o), p, o) for _, p, o, _ in late])
                l_p, l_q = self._bridge.finalize(V, rids=[id(o) for _, _, o, _ in rows])
            if tm:
                tm.stop("aux_wait")
        else:
            self._pf_misses += 1
            if self._bridge is None:
                if tm:
                    tm.start("aux")
                l_p, l_q = self._run_aux(self._aux_reqs)
                if tm:
                    tm.stop("aux")
                if l_p.device != logits.device:  # aux_device: to P's GPU
                    l_p = l_p.to(logits.device)
                    l_q = l_q.to(logits.device)
            else:
                if tm:
                    tm.start("translate")
                # Retokenize each request's new P token, drain the aux
                # engines, and map both distributions onto P's vocab.
                l_p, l_q = self._bridge.step(self._aux_reqs, V)
                if tm:
                    tm.stop("translate")
        if self._overlap and (self._pf_hits + self._pf_misses) % 500 == 0:
            total = self._pf_hits + self._pf_misses
            print(
                f"[DDLogitsProcessor] overlap prefetch hit rate: "
                f"{self._pf_hits}/{total} ({self._pf_hits / total:.1%})",
                flush=True,
            )
        if tm:
            tm.start("fuse")

        P_rows = logits.index_select(0, self._idx_t)  # [N, V]
        for (alpha, rank_k), pos_t, batch_idx_t in self._alpha_groups:
            if pos_t is None:
                P_g, lp_g, lq_g = P_rows, l_p, l_q
            else:
                P_g = P_rows.index_select(0, pos_t)
                lp_g = l_p.index_select(0, pos_t)
                lq_g = l_q.index_select(0, pos_t)
            fused = dd_fuse(
                P_g,
                lp_g,
                lq_g,
                alpha=alpha,
                rank_k=rank_k,
                wl_mask=self._wl_mask,
                pin=self._pin,
            )  # [n_g, V]
            for tid in self._suppress_ids:
                fused[:, tid] = float("-inf")
            logits.index_copy_(0, batch_idx_t, fused.to(logits.dtype))

        if tm:
            tm.stop("fuse")
            tm.step()
        return logits

    # ── Misc ──────────────────────────────────────────────────────────────────

    def is_argmax_invariant(self) -> bool:
        return False

    @classmethod
    def validate_params(cls, sampling_params: SamplingParams) -> None:
        extra = sampling_params.extra_args or {}
        if "dd_alpha" in extra:
            try:
                alpha = float(extra["dd_alpha"])
            except (TypeError, ValueError) as e:
                raise ValueError(f"dd_alpha must be numeric, got {extra['dd_alpha']!r}") from e
            if alpha < 0:
                raise ValueError(f"dd_alpha must be >= 0, got {alpha}")
        if "dd_rank_k" in extra:
            try:
                rank_k = int(extra["dd_rank_k"])
            except (TypeError, ValueError) as e:
                raise ValueError(f"dd_rank_k must be an int, got {extra['dd_rank_k']!r}") from e
            if rank_k < 0:
                raise ValueError(f"dd_rank_k must be >= 0, got {rank_k}")
