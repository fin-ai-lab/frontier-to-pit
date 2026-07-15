"""CPU ground-truth tests for the engine's PAGED-specific semantics.

The general engine behavior (lockstep/staggered ground truth, pairs, rewind,
window-crossing determinism, prompt-longer-than-window) is covered by
test_engine_cpu.py / test_engine_fused_cpu.py. This file covers what only the
paged design does: the re-prime past the window (verified against an explicit
visible-context reference at every step through two re-primes) and the page
pool lifecycle (recycling on unregister, mid-run growth without corrupting
live rows).
"""

from copy import deepcopy

import pytest
import torch

from ftp import AuxBatchedEngine

DEV = torch.device("cpu")
WINDOW = 32
KEEP = WINDOW - WINDOW // 4  # engine default DD_REPRIME_MARGIN=0.25 -> 24
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


class _VisTracker:
    """Mirror of the engine's visible-context rule, per request: (re)prime
    truncates to the window (or KEEP when the context exceeds it); decode
    appends. The reference forward runs on exactly this list."""

    def __init__(self) -> None:
        self.vis: dict[int, list[int]] = {}
        self.n_cached: dict[int, int] = {}

    def feed(self, rid: int, prompt: list[int], output: list[int]) -> None:
        ctx = list(prompt) + list(output)
        cached = self.n_cached.get(rid)
        if cached is None or cached + 1 > WINDOW:  # prime or re-prime
            keep = WINDOW if len(ctx) < WINDOW else KEEP
            self.vis[rid] = ctx[-keep:]
        else:
            self.vis[rid] = self.vis[rid] + [output[-1]]
        self.n_cached[rid] = len(self.vis[rid])


def test_reprime_crossing_window(tiny_llama, engine):
    """Generating past the window re-primes (truncate to KEEP, re-based
    positions). Verified against the engine-visible-context reference at every
    step through two re-primes."""
    g = torch.Generator().manual_seed(5)
    n, t0, steps = 3, 12, 3 * WINDOW
    prompts = [rand_tokens(g, t0) for _ in range(n)]
    conts = [rand_tokens(g, steps) for _ in range(n)]
    trk = _VisTracker()

    for i in range(n):
        engine.register(i)
    out = engine.step([(i, prompts[i], []) for i in range(n)])
    for i in range(n):
        trk.feed(i, prompts[i], [])
        torch.testing.assert_close(
            out[i].float(), ref_logits(tiny_llama, trk.vis[i]), rtol=1e-4, atol=1e-4
        )
    for s in range(steps):
        out = engine.step([(i, prompts[i], conts[i][: s + 1]) for i in range(n)])
        for i in range(n):
            trk.feed(i, prompts[i], conts[i][: s + 1])
            torch.testing.assert_close(
                out[i].float(), ref_logits(tiny_llama, trk.vis[i]), rtol=1e-4, atol=1e-4
            )


def test_window_sized_prompt_primes_at_keep(tiny_llama, engine):
    """A context of exactly WINDOW tokens primes on the KEEP tail directly.
    Priming the full window bought zero decode steps (the next step would
    immediately re-prime to KEEP anyway — a doubled prefill bill measured at
    input == window in serving); starting at KEEP is strictly cheaper and
    buys the whole re-prime margin. Exactness pinned against the reference
    forward at the prime and through the following decode steps."""
    g = torch.Generator().manual_seed(9)
    prompt = rand_tokens(g, WINDOW)
    cont = rand_tokens(g, WINDOW // 4)
    engine.register(0)

    out = engine.step([(0, prompt, [])])
    assert engine._states[0].seq_len == KEEP  # not WINDOW: no throwaway prime
    torch.testing.assert_close(
        out[0].float(), ref_logits(tiny_llama, prompt[-KEEP:]), rtol=1e-4, atol=1e-4
    )
    pool_pages_after_prime = len(engine._pool_full.free)
    for s in range(WINDOW // 4):  # decode straight through — no re-prime fires
        out = engine.step([(0, prompt, cont[: s + 1])])
        torch.testing.assert_close(
            out[0].float(),
            ref_logits(tiny_llama, prompt[-KEEP:] + cont[: s + 1]),
            rtol=1e-4, atol=1e-4,
        )
    # a re-prime would have freed and re-allocated this row's pages
    assert len(engine._pool_full.free) <= pool_pages_after_prime


def test_mixed_length_prompts_prime_in_one_forward(tiny_llama, engine, monkeypatch):
    """Contexts of different lengths share ONE padded prefill forward (real
    traffic never aligns lengths; per-length forwards would serialize the
    prime bill), exactly — including a beyond-window prompt truncated to
    KEEP. The 2x waste guard splits a group whose shortest row would pay
    more padding than content."""
    g = torch.Generator().manual_seed(12)
    prompts = {0: rand_tokens(g, 17), 1: rand_tokens(g, 25),
               2: rand_tokens(g, WINDOW + 9)}
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
    # 17, 25 and KEEP=24 (truncated from WINDOW+9) merge: max 25 < 2 * min 17
    assert calls == [[17, 24, 25]]
    for k, i in enumerate(prompts):
        vis = prompts[i] if len(prompts[i]) < WINDOW else prompts[i][-KEEP:]
        torch.testing.assert_close(
            out[k].float(), ref_logits(tiny_llama, vis), rtol=1e-4, atol=1e-4
        )
    # decode steps stay exact after the padded prime (pads never entered KV)
    cont = rand_tokens(g, 3)
    for s in range(3):
        out = engine.step([(i, prompts[i], cont[: s + 1]) for i in prompts])
        for k, i in enumerate(prompts):
            vis = prompts[i] if len(prompts[i]) < WINDOW else prompts[i][-KEEP:]
            torch.testing.assert_close(
                out[k].float(), ref_logits(tiny_llama, vis + cont[: s + 1]),
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
