"""CPU tests for serve.py's steering install routes (no engine is built).

The pre-capture route ships its config to the engine-core process via
DD_STEER_* env and a worker_cls string — these tests pin the round-trip
(serve writes → steering_worker reads) and the engine kwargs each route
produces, most importantly the compile-cache guard: vLLM's compile cache
hash EXCLUDES worker_cls (ParallelConfig's ignored-fields list) and never
sees forward hooks, and a warm unsteered artifact measurably drops the
hooks (token-identical to unsteered; 2xH100 box, vllm 0.23, 2026-07-04).
"""

import pytest

pytest.importorskip("vllm")  # serve/steering_worker pull in vLLM interfaces

from ftp import steering_worker
from ftp.serve import SteerArgs, _common_kwargs, parse_steer

TRIPLES = [(48, 28961, 35.0), (27, 24365, 20.0)]


def _clear_env(monkeypatch):
    for k in ("DD_STEER_TRIPLES", "DD_STEER_FAMILY", "DD_STEER_SAE_DIR",
              "DD_STEER_SAE_REPO", "DD_STEER_SAE_CACHE", "DD_STEER_WIDTH",
              "DD_STEER_L0", "VLLM_DISABLE_COMPILE_CACHE",
              "VLLM_ALLOW_INSECURE_SERIALIZATION"):
        monkeypatch.delenv(k, raising=False)


def test_precapture_kwargs_and_env_roundtrip(monkeypatch):
    _clear_env(monkeypatch)
    st = SteerArgs(triples=TRIPLES, sae_dir="/tmp/saes")
    kw = _common_kwargs("model", dd_cfg=None, steering=True, tp=1,
                        gpu_mem=0.9, max_len=2048, steer_precapture=st)

    assert kw["worker_cls"] == "ftp.steering_worker.DDSteeringWorker"
    assert kw["enforce_eager"] is False  # the whole point: keep cudagraphs
    import os
    assert os.environ["VLLM_DISABLE_COMPILE_CACHE"] == "1"  # cache trap guard
    # the eager-route pickle fallback is NOT needed (no callable ships by RPC)
    assert "VLLM_ALLOW_INSECURE_SERIALIZATION" not in os.environ

    back = steering_worker._steer_args_from_env()
    assert back.triples == TRIPLES
    assert back.family == "topk"
    assert back.sae_dir == "/tmp/saes"
    assert back.sae_repo is None and back.sae_cache is None
    # pairs resolve to the same SAE files and specs on the worker side
    for (src, spec), (layer, feat, clamp) in zip(back.pairs(), TRIPLES, strict=True):
        assert src.path == f"/tmp/saes/layer{layer}.sae.pt"
        assert (spec.layer, spec.feature_id, spec.clamp_value) == (layer, feat, clamp)


def test_precapture_roundtrip_jumprelu_hub(monkeypatch):
    _clear_env(monkeypatch)
    st = SteerArgs(triples=[(31, 7, 140.0)], family="jumprelu",
                   sae_repo="google/gemma-scope-2-27b-it", sae_cache="/tmp/hf",
                   width="131k", l0_size="big")
    _common_kwargs("model", dd_cfg=None, steering=True, tp=1,
                   gpu_mem=0.9, max_len=2048, steer_precapture=st)
    back = steering_worker._steer_args_from_env()
    assert back.triples == [(31, 7, 140.0)]
    assert (back.family, back.width, back.l0_size) == ("jumprelu", "131k", "big")
    src, spec = back.pairs()[0]
    assert src.repo_id == "google/gemma-scope-2-27b-it"
    assert (src.family, src.width, src.l0_size) == ("jumprelu", "131k", "big")
    assert src.cache_dir == "/tmp/hf" and spec.clamp_value == 140.0


def test_eager_route_unchanged(monkeypatch):
    _clear_env(monkeypatch)
    kw = _common_kwargs("model", dd_cfg=None, steering=True, tp=1,
                        gpu_mem=0.9, max_len=2048)
    assert kw["enforce_eager"] is True  # hooks only fire eagerly post-build
    assert "worker_cls" not in kw
    import os
    assert os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] == "1"
    assert "VLLM_DISABLE_COMPILE_CACHE" not in os.environ


def test_precapture_cache_guard_respects_explicit_zero(monkeypatch):
    """An explicit 0 in the environment is the documented (at-your-own-risk)
    override; setdefault must not stomp it."""
    _clear_env(monkeypatch)
    monkeypatch.setenv("VLLM_DISABLE_COMPILE_CACHE", "0")
    st = SteerArgs(triples=TRIPLES, sae_dir="/tmp/saes")
    _common_kwargs("model", dd_cfg=None, steering=True, tp=1,
                   gpu_mem=0.9, max_len=2048, steer_precapture=st)
    import os
    assert os.environ["VLLM_DISABLE_COMPILE_CACHE"] == "0"


def test_worker_env_parse_empty_is_none(monkeypatch):
    """Default Worker semantics: no DD_STEER_TRIPLES (or empty) → no install."""
    _clear_env(monkeypatch)
    assert steering_worker._steer_args_from_env() is None
    monkeypatch.setenv("DD_STEER_TRIPLES", "")
    assert steering_worker._steer_args_from_env() is None


def test_precapture_is_the_default_route():
    """Production contract: build_llm/build_async_llm steer via pre-capture
    unless the caller explicitly opts into the eager post-build route."""
    import inspect

    from ftp.serve import build_async_llm, build_llm

    assert inspect.signature(build_llm).parameters["steer_precapture"].default is True
    assert inspect.signature(build_async_llm).parameters["steer_precapture"].default is True


def test_install_steering_refuses_non_eager_engine():
    """Post-build hooks never fire under compiled/captured execution — the
    install must raise, not silently serve unsteered."""
    from types import SimpleNamespace

    from ftp.steering import SaeSource, SteerSpec, install_steering

    fake_llm = SimpleNamespace(llm_engine=SimpleNamespace(vllm_config=SimpleNamespace(
        model_config=SimpleNamespace(enforce_eager=False))))
    pairs = [(SaeSource(path="/tmp/x.pt"), SteerSpec(layer=0, feature_id=1, clamp_value=5.0))]
    with pytest.raises(RuntimeError, match="steer_precapture"):
        install_steering(fake_llm, pairs)
    # empty install stays a no-op regardless of engine mode
    install_steering(fake_llm, [])


def test_parse_steer_formats():
    assert parse_steer("48:28961:35,27:24365:20") == TRIPLES
    assert parse_steer(" 48:28961:35 , ") == [(48, 28961, 35.0)]
    assert parse_steer("") == []
    # clamp survives the env round-trip's %g formatting for non-integral values
    assert parse_steer("5:1:2.5") == [(5, 1, 2.5)]
