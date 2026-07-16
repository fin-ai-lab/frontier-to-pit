"""CPU ground-truth tests for AuxBatchedEngine using a tiny random Llama.

These run the REAL engine code paths (batched prefill, uniform decode,
grouped staggered decode, per-request fallback, capacity enforcement, page/slot
recycling) against an unambiguous reference — re-running the model on the full
context from scratch at every step — in deterministic fp32 on CPU. This is the
CI net for the cache-management bug class (cursor desync, prefill copy
misalignment, page-table corruption).
"""

from copy import deepcopy

import pytest
import torch

from ftp import AuxBatchedEngine

DEV = torch.device("cpu")
WINDOW = 32
V = 512


class _Plane0Facade:
    """Expose plane 0 of a fused-pair engine through the single-engine API, so
    every ground-truth test in this file also runs against the fused path
    (plane-1 correctness is covered in test_engine_fused_cpu.py)."""

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
    """Engine factory: the plain single-model engine, or a fused pair seen
    through its plane 0 (whose weights are `tiny_llama`)."""

    def make():
        if request.param == "single":
            return AuxBatchedEngine(tiny_llama, DEV, torch.float32, WINDOW)
        return _Plane0Facade(
            AuxBatchedEngine(
                deepcopy(tiny_llama),
                DEV,
                torch.float32,
                WINDOW,
                model2=deepcopy(tiny_llama_q),
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


def assert_matches_reference(model, outs: dict, seqs: dict) -> None:
    """outs: {(step, rid): engine_logits}; seqs: {(step, rid): full token list}."""
    for key in outs:
        ref = ref_logits(model, seqs[key])
        torch.testing.assert_close(outs[key], ref, rtol=1e-4, atol=1e-4)


def test_lockstep_ground_truth(tiny_llama, engine):
    """Uniform batch through prefill + decode, exact vs full-context reference.

    Stays within the engine capacity: past it the engine raises (context is
    never truncated — see test_capacity_exceeded_raises)."""
    g = torch.Generator().manual_seed(1)
    n, t0 = 4, 12
    steps = WINDOW - t0 - 1  # last step decodes at T = WINDOW - 1: no shift yet
    prompts = [rand_tokens(g, t0) for _ in range(n)]
    conts = [rand_tokens(g, steps) for _ in range(n)]

    for i in range(n):
        engine.register(i)
    outs, seqs = {}, {}
    out = engine.step([(i, prompts[i], []) for i in range(n)])
    for i in range(n):
        outs[(0, i)] = out[i].float()
        seqs[(0, i)] = list(prompts[i])
    for s in range(steps):
        out = engine.step([(i, prompts[i], conts[i][: s + 1]) for i in range(n)])
        for i in range(n):
            outs[(s + 1, i)] = out[i].float()
            seqs[(s + 1, i)] = list(prompts[i]) + conts[i][: s + 1]

    assert_matches_reference(tiny_llama, outs, seqs)


def test_capacity_exceeded_raises(tiny_llama, make_engine):
    """The engine never truncates: a row that would exceed the capacity
    (``window``) raises instead of silently re-priming a truncated tail (the
    old sliding-window mechanic, removed 2026-07-16). Decode up to the cap is
    exact ground truth the whole way."""
    g = torch.Generator().manual_seed(6)
    t0 = 12
    prompt = rand_tokens(g, t0)
    cont = rand_tokens(g, WINDOW - t0 + 1)  # one token past the cap

    eng = make_engine()
    eng.register(0)
    eng.step([(0, prompt, [])])
    for s in range(WINDOW - t0):  # last step decodes at seq_len == WINDOW
        out = eng.step([(0, prompt, cont[: s + 1])])
    torch.testing.assert_close(
        out[0].float(),
        ref_logits(tiny_llama, prompt + cont[: WINDOW - t0]),
        rtol=1e-4, atol=1e-4,
    )
    with pytest.raises(RuntimeError, match="capacity"):
        eng.step([(0, prompt, cont)])


def test_staggered_ground_truth(tiny_llama, engine):
    """vLLM's leader/follower stagger: req 0 prefills one step early, then the
    batch decodes permanently one token apart → exercises the grouped decode
    path (and its column backup/restore) every step."""
    g = torch.Generator().manual_seed(2)
    n, t0, steps = 4, 10, 18  # stays below WINDOW → grouped path, never fallback
    prompt = rand_tokens(g, t0)
    conts = [rand_tokens(g, steps + 1) for _ in range(n)]

    outs, seqs = {}, {}
    engine.register(0)
    out = engine.step([(0, prompt, [])])
    outs[(0, 0)], seqs[(0, 0)] = out[0].float(), list(prompt)

    for i in range(1, n):
        engine.register(i)
    reqs = [(0, prompt, conts[0][:1])] + [(i, prompt, []) for i in range(1, n)]
    out = engine.step(reqs)
    for j, (rid, _, oids) in enumerate(reqs):
        outs[(1, rid)] = out[j].float()
        seqs[(1, rid)] = list(prompt) + list(oids)

    for s in range(2, steps):
        reqs = [(0, prompt, conts[0][:s])] + [(i, prompt, conts[i][: s - 1]) for i in range(1, n)]
        out = engine.step(reqs)
        for j, (rid, _, oids) in enumerate(reqs):
            outs[(s, rid)] = out[j].float()
            seqs[(s, rid)] = list(prompt) + list(oids)

    assert_matches_reference(tiny_llama, outs, seqs)


def test_fallback_ground_truth(tiny_llama, engine):
    """>4 distinct positions forces the per-request fallback path."""
    g = torch.Generator().manual_seed(3)
    n, t0 = 6, 8
    prompt = rand_tokens(g, t0)
    conts = [rand_tokens(g, 16) for _ in range(n)]

    outs, seqs = {}, {}
    # Prefill each request in its own step, so request i is i tokens ahead.
    for i in range(n):
        engine.register(i)
        reqs = [(j, prompt, conts[j][: i - j]) for j in range(i)] + [(i, prompt, [])]
        out = engine.step(reqs)
        for k, (rid, _, oids) in enumerate(reqs):
            outs[(i, rid)] = out[k].float()
            seqs[(i, rid)] = list(prompt) + list(oids)

    # Now 6 distinct seq lens → fallback every step.
    for s in range(4):
        reqs = [(j, prompt, conts[j][: n - j + s]) for j in range(n)]
        out = engine.step(reqs)
        for k, (rid, _, oids) in enumerate(reqs):
            outs[(n + s, rid)] = out[k].float()
            seqs[(n + s, rid)] = list(prompt) + list(oids)

    assert_matches_reference(tiny_llama, outs, seqs)


def test_slot_recycling(tiny_llama, engine):
    """Unregister/re-register must hand back a clean slot that still produces
    ground-truth logits (covers the zero_() cleanup + free-set reuse)."""
    g = torch.Generator().manual_seed(4)
    p1, p2 = rand_tokens(g, 9), rand_tokens(g, 13)

    engine.register(101)
    engine.step([(101, p1, [])])
    cont = rand_tokens(g, 5)
    for s in range(5):
        engine.step([(101, p1, cont[: s + 1])])
    engine.unregister(101)

    engine.register(202)  # reuses slot 0
    out = engine.step([(202, p2, [])])
    torch.testing.assert_close(out[0].float(), ref_logits(tiny_llama, p2), rtol=1e-4, atol=1e-4)
    c2 = rand_tokens(g, 3)
    for s in range(3):
        out = engine.step([(202, p2, c2[: s + 1])])
    torch.testing.assert_close(
        out[0].float(), ref_logits(tiny_llama, p2 + c2), rtol=1e-4, atol=1e-4
    )


def test_compile_flag_accepted_on_cpu(tiny_llama):
    """compile_model is accepted for signature compatibility and ignored (the
    paged engine runs eager), so a CPU engine with the flag still works."""
    from copy import deepcopy

    eng = AuxBatchedEngine(deepcopy(tiny_llama), DEV, torch.float32, WINDOW, compile_model=True)
    eng.register(0)
    out = eng.step([(0, [3, 4, 5], [])])
    assert torch.isfinite(out).all()


def test_prompt_longer_than_capacity_raises(tiny_llama, engine):
    """Prompts beyond the capacity raise — the engine never trims a prompt to
    a tail (the old KEEP-tail prime is gone with the sliding-window mechanic).
    A prompt of exactly the capacity primes in full and is exact."""
    g = torch.Generator().manual_seed(5)
    engine.register(7)
    with pytest.raises(ValueError, match="capacity"):
        engine.step([(7, rand_tokens(g, WINDOW + 10), [])])
    full_prompt = rand_tokens(g, WINDOW)
    engine.register(8)
    out = engine.step([(8, full_prompt, [])])
    torch.testing.assert_close(
        out[-1].float(), ref_logits(tiny_llama, full_prompt), rtol=1e-4, atol=1e-4
    )


def test_rewind_ground_truth(tiny_llama, engine):
    """rewind(k) below the window cap is exact: feeding an alternate
    continuation after the rewind matches the full-context reference on the
    alternate sequence (the universal-decoding retokenization case)."""
    g = torch.Generator().manual_seed(9)
    prompt, cont, alt = rand_tokens(g, 12), rand_tokens(g, 5), rand_tokens(g, 4)

    engine.register(11)
    engine.step([(11, prompt, [])])
    for s in range(5):
        engine.step([(11, prompt, cont[: s + 1])])

    engine.rewind(11, 2)  # drop cont[3:5]; cached context is prompt + cont[:3]
    for s in range(4):
        out = engine.step([(11, prompt, cont[:3] + alt[: s + 1])])
        torch.testing.assert_close(
            out[0].float(),
            ref_logits(tiny_llama, prompt + cont[:3] + alt[: s + 1]),
            rtol=1e-4,
            atol=1e-4,
        )


def test_rewind_at_cap_ground_truth(tiny_llama, engine):
    """rewind(k) on a row AT the capacity, then an alternate continuation back
    up to the cap, stays exact (no truncated tail anywhere: the cache holds the
    row's full context)."""
    g = torch.Generator().manual_seed(10)
    t0 = 12
    prompt = rand_tokens(g, t0)
    cont = rand_tokens(g, WINDOW - t0)  # fills the row exactly to the cap
    alt = rand_tokens(g, 2)

    engine.register(0)
    engine.step([(0, prompt, [])])
    for s in range(WINDOW - t0):
        engine.step([(0, prompt, cont[: s + 1])])
    engine.rewind(0, 2)  # seq_len WINDOW -> WINDOW - 2
    base = cont[: WINDOW - t0 - 2]
    for s in range(2):
        out = engine.step([(0, prompt, base + alt[: s + 1])])
        torch.testing.assert_close(
            out[0].float(),
            ref_logits(tiny_llama, prompt + base + alt[: s + 1]),
            rtol=1e-4, atol=1e-4,
        )


def test_subset_composition_change(tiny_llama, engine):
    """Decode-row subsets that change composition WITHOUT a register/unregister
    in between (vLLM recompute pauses; the universal bridge's drain rounds)
    must not replay another subset's cached slot indices. Regression: the
    decode index cache was keyed by output-index tuples only, so 'only request
    1' after 'only request 0' fed request 1's token into request 0's slot."""
    g = torch.Generator().manual_seed(12)
    prompts = [rand_tokens(g, 6), rand_tokens(g, 9)]
    conts = [rand_tokens(g, 6) for _ in range(2)]
    for i in range(2):
        engine.register(i)
    engine.step([(i, prompts[i], []) for i in range(2)])
    engine.step([(i, prompts[i], conts[i][:1]) for i in range(2)])

    # alternate single-row subsets at the same output index
    for s in range(1, 4):
        out0 = engine.step([(0, prompts[0], conts[0][: s + 1])])
        out1 = engine.step([(1, prompts[1], conts[1][: s + 1])])
        torch.testing.assert_close(
            out0[0].float(),
            ref_logits(tiny_llama, prompts[0] + conts[0][: s + 1]),
            rtol=1e-4,
            atol=1e-4,
        )
        torch.testing.assert_close(
            out1[0].float(),
            ref_logits(tiny_llama, prompts[1] + conts[1][: s + 1]),
            rtol=1e-4,
            atol=1e-4,
        )


def test_step_pairs_ground_truth(tiny_llama, engine):
    """step_pairs (1-2 tokens per row in one forward) below the window cap is
    exact: mixed k=1/k=2 batches match the full-context reference at the last
    fed token of every row."""
    g = torch.Generator().manual_seed(13)
    n = 3
    prompts = [rand_tokens(g, 5 + i) for i in range(n)]
    conts = [rand_tokens(g, 14) for _ in range(n)]
    for i in range(n):
        engine.register(i)
    engine.step([(i, prompts[i], []) for i in range(n)])

    fed = [0] * n
    for step in range(5):
        reqs = []
        for i in range(n):
            k = 2 if (step + i) % 2 == 0 else 1  # alternate 1- and 2-token rows
            reqs.append((i, conts[i][fed[i] : fed[i] + k]))
            fed[i] += k
        out = engine.step_pairs(reqs)
        for i in range(n):
            torch.testing.assert_close(
                out[i].float(),
                ref_logits(tiny_llama, prompts[i] + conts[i][: fed[i]]),
                rtol=1e-4,
                atol=1e-4,
            )


def test_step_pairs_interleaves_with_step(tiny_llama, engine):
    """Alternating step() and step_pairs() on the same rows stays on ground
    truth (the bridge mixes both paths per drain round)."""
    g = torch.Generator().manual_seed(14)
    prompt, cont = rand_tokens(g, 6), rand_tokens(g, 12)
    engine.register(0)
    engine.step([(0, prompt, [])])
    fed = 0
    for step in range(4):
        if step % 2 == 0:
            out = engine.step_pairs([(0, cont[fed : fed + 2])])
            fed += 2
        else:
            out = engine.step([(0, prompt, cont[: fed + 1])])
            fed += 1
        torch.testing.assert_close(
            out[0].float(),
            ref_logits(tiny_llama, prompt + cont[:fed]),
            rtol=1e-4,
            atol=1e-4,
        )


def test_step_pairs_after_rewind(tiny_llama, engine):
    """The bridge's burst pattern: rewind 1 then feed 2 replacement tokens in
    one pairs forward — exact below the cap."""
    g = torch.Generator().manual_seed(15)
    prompt, cont, alt = rand_tokens(g, 8), rand_tokens(g, 4), rand_tokens(g, 2)
    engine.register(0)
    engine.step([(0, prompt, [])])
    for s in range(4):
        engine.step([(0, prompt, cont[: s + 1])])
    engine.rewind(0, 1)
    out = engine.step_pairs([(0, alt)])
    torch.testing.assert_close(
        out[0].float(),
        ref_logits(tiny_llama, prompt + cont[:3] + alt),
        rtol=1e-4,
        atol=1e-4,
    )


def test_step_pairs_past_capacity_raises(tiny_llama, engine):
    """A pairs feed that would cross the capacity raises (page tables are
    sized to the capacity; nothing re-primes it away anymore)."""
    g = torch.Generator().manual_seed(16)
    prompt = rand_tokens(g, WINDOW - 1)
    engine.register(0)
    engine.step([(0, prompt, [])])
    with pytest.raises(RuntimeError, match="capacity"):
        engine.step_pairs([(0, rand_tokens(g, 2))])  # WINDOW - 1 + 2 > WINDOW
    out = engine.step_pairs([(0, rand_tokens(g, 1))])  # exactly at the cap: fine
    assert torch.isfinite(out).all()


def test_step_pairs_validation(tiny_llama, engine):
    g = torch.Generator().manual_seed(17)
    prompt = rand_tokens(g, 5)
    engine.register(0)
    with pytest.raises(ValueError, match="unprimed"):
        engine.step_pairs([(0, [1])])
    engine.step([(0, prompt, [])])
    with pytest.raises(ValueError, match="1-2 tokens"):
        engine.step_pairs([(0, [1, 2, 3])])


def test_rewind_validation(tiny_llama, engine):
    g = torch.Generator().manual_seed(11)
    prompt = rand_tokens(g, 6)
    engine.register(3)
    with pytest.raises(ValueError, match="unprimed"):
        engine.rewind(3, 1)
    engine.step([(3, prompt, [])])
    with pytest.raises(ValueError, match="exceeds"):
        engine.rewind(3, 7)
    engine.rewind(3, 6)  # rewinding the entire cached context is allowed


def test_register_mid_generation(tiny_llama, engine):
    """A request registered mid-generation (vLLM recompute gap) prefills from
    prompt + outputs and continues exactly on ground truth."""
    g = torch.Generator().manual_seed(8)
    prompt, cont = rand_tokens(g, 8), rand_tokens(g, 10)

    engine.register(5)
    out = engine.step([(5, prompt, cont[:6])])  # joins with 6 outputs already
    torch.testing.assert_close(
        out[0].float(), ref_logits(tiny_llama, prompt + cont[:6]), rtol=1e-4, atol=1e-4
    )
    for s in range(6, 9):
        out = engine.step([(5, prompt, cont[: s + 1])])
        torch.testing.assert_close(
            out[0].float(), ref_logits(tiny_llama, prompt + cont[: s + 1]), rtol=1e-4, atol=1e-4
        )
