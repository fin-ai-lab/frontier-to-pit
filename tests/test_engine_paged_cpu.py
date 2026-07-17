"""CPU ground-truth tests for the engine's PAGED-specific semantics.

The general engine behavior (lockstep/staggered ground truth, pairs, rewind,
capacity enforcement) is covered by test_engine_cpu.py /
test_engine_fused_cpu.py. This file covers what only the paged design does:
long generations spanning many pages, full-capacity primes, and the page pool
lifecycle (recycling on unregister, mid-run growth without corrupting live
rows).
"""

from copy import deepcopy

import pytest
import torch

from ftp import AuxBatchedEngine

DEV = torch.device("cpu")
WINDOW = 32
V = 512


class _Plane0Facade:
    """Run every case against the fused path via plane 0."""

    def __init__(self, eng: AuxBatchedEngine) -> None:
        self._eng = eng

    def step(self, requests):
        return self._eng.step(requests)[0]

    def step_pairs(self, requests):
        return self._eng.step_pairs(requests)[0]

    def __getattr__(self, name):
        return getattr(self._eng, name)


@pytest.fixture(params=["single", "fused"])
def make_engine(request, tiny_llama, tiny_llama_q):
    def make(**kw):
        if request.param == "single":
            return AuxBatchedEngine(deepcopy(tiny_llama), DEV, torch.float32, WINDOW, **kw)
        return _Plane0Facade(
            AuxBatchedEngine(
                deepcopy(tiny_llama), DEV, torch.float32, WINDOW,
                model2=deepcopy(tiny_llama_q), **kw,
            )
        )

    return make


@pytest.fixture()
def engine(make_engine):
    return make_engine()


def ref_logits(model, seq: list[int]) -> torch.Tensor:
    with torch.no_grad():
        return model(input_ids=torch.tensor([seq])).logits[0, -1]


def rand_tokens(g: torch.Generator, n: int) -> list[int]:
    return torch.randint(2, V, (n,), generator=g).tolist()


def test_long_generation_many_pages(tiny_llama):
    """A generation spanning many pages (capacity >> one page) stays exact at
    every step — no truncation anywhere: the reference always runs the FULL
    context. (Replaces the old re-prime ground truth: the engine-level sliding
    window was removed 2026-07-16.)"""
    g = torch.Generator().manual_seed(5)
    cap = 4 * WINDOW  # 128 tokens ~ 8 pages per row
    eng = AuxBatchedEngine(deepcopy(tiny_llama), DEV, torch.float32, cap)
    n, t0, steps = 3, 12, cap - 12 - 1
    prompts = [rand_tokens(g, t0) for _ in range(n)]
    conts = [rand_tokens(g, steps) for _ in range(n)]

    for i in range(n):
        eng.register(i)
    out = eng.step([(i, prompts[i], []) for i in range(n)])
    for i in range(n):
        torch.testing.assert_close(
            out[i].float(), ref_logits(tiny_llama, prompts[i]), rtol=1e-4, atol=1e-4
        )
    for s in range(steps):
        out = eng.step([(i, prompts[i], conts[i][: s + 1]) for i in range(n)])
    for i in range(n):  # exact after ~120 decode steps across many pages
        torch.testing.assert_close(
            out[i].float(),
            ref_logits(tiny_llama, prompts[i] + conts[i][:steps]),
            rtol=1e-4, atol=1e-4,
        )


def test_full_capacity_prompt_primes_whole(tiny_llama, engine):
    """A context of exactly the capacity primes IN FULL (the old engine primed
    it on a truncated KEEP tail) and is exact; the next decode step raises the
    capacity error."""
    g = torch.Generator().manual_seed(9)
    prompt = rand_tokens(g, WINDOW)
    engine.register(0)

    out = engine.step([(0, prompt, [])])
    assert engine._states[0].seq_len == WINDOW
    torch.testing.assert_close(
        out[0].float(), ref_logits(tiny_llama, prompt), rtol=1e-4, atol=1e-4
    )
    with pytest.raises(RuntimeError, match="capacity"):
        engine.step([(0, prompt, rand_tokens(g, 1))])


def test_mixed_length_prompts_prime_in_one_forward(tiny_llama, engine, monkeypatch):
    """Contexts of different lengths share ONE padded prefill forward (real
    traffic never aligns lengths; per-length forwards would serialize the
    prime bill), exactly. The 2x waste guard splits a group whose shortest
    row would pay more padding than content."""
    g = torch.Generator().manual_seed(12)
    prompts = {0: rand_tokens(g, 17), 1: rand_tokens(g, 25),
               2: rand_tokens(g, 29)}
    calls: list[list[int]] = []
    orig = type(engine._eng if hasattr(engine, "_eng") else engine)._batch_prefill
    eng_obj = engine._eng if hasattr(engine, "_eng") else engine

    def spy(self, items, logits_out):
        calls.append(sorted(len(it[2]) for it in items))
        return orig(self, items, logits_out)

    monkeypatch.setattr(type(eng_obj), "_batch_prefill", spy)
    for i in prompts:
        engine.register(i)
    out = engine.step([(i, prompts[i], []) for i in prompts])
    # 17, 25 and 29 merge: max 29 < 2 * min 17
    assert calls == [[17, 25, 29]]
    for k, i in enumerate(prompts):
        torch.testing.assert_close(
            out[k].float(), ref_logits(tiny_llama, prompts[i]), rtol=1e-4, atol=1e-4
        )
    # decode steps stay exact after the padded prime (pads never entered KV)
    cont = rand_tokens(g, 3)
    for s in range(3):
        out = engine.step([(i, prompts[i], cont[: s + 1]) for i in prompts])
        for k, i in enumerate(prompts):
            torch.testing.assert_close(
                out[k].float(), ref_logits(tiny_llama, prompts[i] + cont[: s + 1]),
                rtol=1e-4, atol=1e-4,
            )

    # waste guard: 5 tokens vs 12 tokens (12 > 2*5) -> two forwards
    calls.clear()
    engine.register(10)
    engine.register(11)
    engine.step([(10, prompts[0][:5], []), (11, prompts[1][:12], [])])
    assert calls == [[5], [12]]


def test_page_recycling_and_pool_growth(tiny_llama, make_engine):
    """A deliberately tiny pool grows mid-run without corrupting live rows,
    and pages freed by unregister are recycled by later requests."""
    g = torch.Generator().manual_seed(8)
    eng = make_engine(pool_tokens=2 * WINDOW)  # tiny: forces at least one growth
    prompts = [rand_tokens(g, 13) for _ in range(6)]

    eng.register(0)
    eng.register(1)
    eng.step([(0, prompts[0], []), (1, prompts[1], [])])
    free_before = len(eng._pool_full.free)
    eng.unregister(0)
    assert len(eng._pool_full.free) > free_before  # pages returned

    eng.register(2)  # recycles request 0's pages
    out = eng.step([(2, prompts[2], []), (1, prompts[1], [rand_tokens(g, 1)[0]])])
    ref = ref_logits(tiny_llama, prompts[2])
    torch.testing.assert_close(out[0].float(), ref, rtol=1e-4, atol=1e-4)

    # Grow: register enough rows that the tiny pool must expand, then verify
    # an existing row still decodes exactly (growth preserved its pages).
    for rid in (3, 4, 5):
        eng.register(rid)
    eng.step([(rid, prompts[rid], []) for rid in (3, 4, 5)])
    tok = rand_tokens(g, 1)[0]
    out = eng.step([(2, prompts[2], [tok])])
    ref = ref_logits(tiny_llama, prompts[2] + [tok])
    torch.testing.assert_close(out[0].float(), ref, rtol=1e-4, atol=1e-4)
