"""Pre-capture SAE steering: a vLLM worker that installs the steering hooks
BEFORE memory profiling and CUDA-graph capture.

Why: the steering hook is ~4 tiny capture-safe kernels (two rank-1 matmuls, a
clamp, an add) on fixed weight pointers — but installed AFTER engine build
(the ``install_steering`` route) the hooks can't fire inside the already-
captured graphs, forcing ``enforce_eager=True`` on P and roughly halving
throughput. Installed BEFORE capture, the hook's kernels are recorded inside
the graphs and replay with zero per-step python: full cudagraphs + steering.

Trade: clamp values become capture-time constants — one engine build per
steering configuration (production serving; the in-process clamp-sweep evals
keep the eager path).

Compile-cache hazard (measured, not theoretical): vLLM's torch.compile cache
hash EXCLUDES worker_cls (ParallelConfig's ignored-fields list) and dynamo
skips nn-module hook guards, so a warm artifact from an unsteered build is a
valid cache hit for a steered one — the model is then served with the hooks
silently dropped (verified token-identical to unsteered on the 2xH100 box,
vllm 0.23). serve.py therefore sets VLLM_DISABLE_COMPILE_CACHE=1 for every
pre-capture build; only an explicit "0" in the environment overrides it.

Selected via ``worker_cls`` (see ``serve.build_llm(steer_precapture=True)``);
configuration travels by ``DD_STEER_*`` env because vLLM instantiates the
worker class by import path inside the engine-core process:

  DD_STEER_TRIPLES   "48:28961:35,27:24365:20"
  DD_STEER_FAMILY    "topk" (default) | "jumprelu"
  DD_STEER_SAE_DIR   dir with layer{L}.sae.pt (topk)
  DD_STEER_SAE_REPO / DD_STEER_SAE_CACHE / DD_STEER_WIDTH / DD_STEER_L0
"""

from __future__ import annotations

import functools
import os

from vllm.v1.worker.gpu_worker import Worker


def _steer_args_from_env():
    from ftp.serve import SteerArgs, parse_steer

    triples = parse_steer(os.environ.get("DD_STEER_TRIPLES", ""))
    if not triples:
        return None
    return SteerArgs(
        triples=triples,
        family=os.environ.get("DD_STEER_FAMILY", "topk"),
        sae_dir=os.environ.get("DD_STEER_SAE_DIR") or None,
        sae_repo=os.environ.get("DD_STEER_SAE_REPO") or None,
        sae_cache=os.environ.get("DD_STEER_SAE_CACHE") or None,
        width=os.environ.get("DD_STEER_WIDTH", "65k"),
        l0_size=os.environ.get("DD_STEER_L0", "medium"),
    )


class DDSteeringWorker(Worker):
    """GPU worker that installs SAE steering hooks right after model load —
    strictly before vLLM's memory profiling and CUDA-graph capture, so the
    hook kernels are captured into the graphs (and the SAE slices are
    accounted by the profiler)."""

    def load_model(self, *args, **kwargs):
        super().load_model(*args, **kwargs)
        steer = _steer_args_from_env()
        if steer is None:
            return
        from ftp.steering import _install_on_model

        self.apply_model(functools.partial(_install_on_model, pairs=steer.pairs()))
        print(
            f"[DDSteeringWorker] steering installed PRE-CAPTURE: "
            f"{os.environ.get('DD_STEER_TRIPLES')} "
            "(hook kernels will be recorded inside the CUDA graphs)",
            flush=True,
        )
