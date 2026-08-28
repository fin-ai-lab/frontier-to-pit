"""CPU-only regression tests for the lm-eval DD engine configuration."""

import pytest

pytest.importorskip("vllm")
pytest.importorskip("lm_eval")

from evals.lmeval.backends import DDVLLM
from lm_eval.models.vllm_causallms import VLLM


def _capture_engine_kwargs(monkeypatch, **kwargs):
    captured = {}

    def fake_init(_self, *, pretrained, **engine_kwargs):
        captured["pretrained"] = pretrained
        captured.update(engine_kwargs)

    monkeypatch.setattr(VLLM, "__init__", fake_init)
    DDVLLM("model", aux_p="forget", aux_q="retain", **kwargs)
    return captured


def test_dd_lmeval_disables_async_scheduling_by_default(monkeypatch):
    """An omitted scheduler request resolves to the overlap-safe path."""
    captured = _capture_engine_kwargs(monkeypatch)
    assert captured["async_scheduling"] is False


@pytest.mark.parametrize("supplied", [None, False])
def test_dd_lmeval_accepts_default_like_sync_values(monkeypatch, supplied):
    """Explicit None must not resolve back to vLLM's async default."""
    captured = _capture_engine_kwargs(monkeypatch, async_scheduling=supplied)
    assert captured["async_scheduling"] is False


def test_dd_lmeval_preserves_explicit_true_for_processor_error(monkeypatch):
    """The processor, not the wrapper, owns the targeted incompatibility error."""
    captured = _capture_engine_kwargs(monkeypatch, async_scheduling=True)
    assert captured["async_scheduling"] is True


def test_vllm_serve_example_disables_async_scheduling():
    """The documented CLI invocation must satisfy DDLogitsProcessor's contract."""
    import ftp.vllm

    assert "--logits-processors ftp.vllm:DDLogitsProcessor" in ftp.vllm.__doc__
    assert "--no-async-scheduling" in ftp.vllm.__doc__
