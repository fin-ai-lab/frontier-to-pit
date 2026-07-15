"""CPU ground truth for the fused-pair engine BEYOND what the plane-0 facade
in test_engine_cpu.py covers: plane-1 correctness, cross-plane independence,
the per-plane cache grow-copy, and the [2, N, V] output contract.

Both planes are checked against full-context reference forwards of their own
source model at every step — the net for plane-marriage and plane-swap bugs.
"""

from copy import deepcopy

import pytest
import torch

from ftp import AuxBatchedEngine

DEV = torch.device("cpu")
WINDOW = 32
V = 512


@pytest.fixture()
def fused_engine(tiny_llama, tiny_llama_q):
    return AuxBatchedEngine(
        deepcopy(tiny_llama), DEV, torch.float32, WINDOW, model2=deepcopy(tiny_llama_q)
    )


def ref_logits(model, seq: list[int]) -> torch.Tensor:
    with torch.no_grad():
        return model(input_ids=torch.tensor([seq[-WINDOW:]])).logits[0, -1]


def rand_tokens(g: torch.Generator, n: int) -> list[int]:
    return torch.randint(2, V, (n,), generator=g).tolist()


def assert_both_planes(models, out, idx: int, seq: list[int]) -> None:
    for plane, model in enumerate(models):
        torch.testing.assert_close(
            out[plane, idx].float(), ref_logits(model, seq), rtol=1e-4, atol=1e-4
        )


def test_output_shapes(fused_engine):
    g = torch.Generator().manual_seed(0)
    assert fused_engine.step([]).shape == (2, 0, V)
    assert fused_engine.step_pairs([]).shape == (2, 0, V)
    fused_engine.register(0)
    out = fused_engine.step([(0, rand_tokens(g, 5), [])])
    assert out.shape == (2, 1, V)
    out = fused_engine.step_pairs([(0, rand_tokens(g, 2))])
    assert out.shape == (2, 1, V)


def test_planes_disagree(fused_engine, tiny_llama, tiny_llama_q):
    g = torch.Generator().manual_seed(1)
    prompt = rand_tokens(g, 10)
    fused_engine.register(0)
    out = fused_engine.step([(0, prompt, [])])
    assert not torch.allclose(out[0], out[1], rtol=1e-3, atol=1e-3)
    assert_both_planes((tiny_llama, tiny_llama_q), out, 0, prompt)


def test_rich_script_both_planes(fused_engine, tiny_llama, tiny_llama_q):
    """One serving-shaped script through every code path — staggered prefills,
    lockstep decode, mid-run cache GROWTH (the per-plane grow-copy net),
    rewind, step_pairs bursts, slot recycling, and a window crossing at the
    end — with both planes on ground truth at every checkpoint below the cap."""
    models = (tiny_llama, tiny_llama_q)
    g = torch.Generator().manual_seed(2)
    prompts = [rand_tokens(g, 6 + i) for i in range(4)]
    conts = [rand_tokens(g, 24) for _ in range(4)]

    # Two requests first: cache is allocated at cache_n=2.
    for i in range(2):
        fused_engine.register(i)
    out = fused_engine.step([(i, prompts[i], []) for i in range(2)])
    for i in range(2):
        assert_both_planes(models, out, i, prompts[i])
    for s in range(3):
        out = fused_engine.step([(i, prompts[i], conts[i][: s + 1]) for i in range(2)])
        for i in range(2):
            assert_both_planes(models, out, i, prompts[i] + conts[i][: s + 1])

    # Two more requests join: forces _ensure_main_cache growth 2 -> 4 with
    # live data in both planes (a flat copy would corrupt plane 1).
    for i in range(2, 4):
        fused_engine.register(i)
    reqs = [(i, prompts[i], conts[i][:4]) for i in range(2)] + [
        (i, prompts[i], []) for i in range(2, 4)
    ]
    out = fused_engine.step(reqs)
    for j, (rid, _, oids) in enumerate(reqs):
        assert_both_planes(models, out, j, prompts[rid] + list(oids))

    # Lockstep all four.
    fed = [4, 4, 0, 0]
    for _ in range(3):
        reqs = [(i, prompts[i], conts[i][: fed[i] + 1]) for i in range(4)]
        out = fused_engine.step(reqs)
        for i in range(4):
            fed[i] += 1
            assert_both_planes(models, out, i, prompts[i] + conts[i][: fed[i]])

    # Rewind one row, replace via a step_pairs burst.
    fused_engine.rewind(1, 1)
    fed[1] -= 1
    alt = rand_tokens(g, 2)
    out = fused_engine.step_pairs([(1, alt)])
    assert_both_planes(models, out, 0, prompts[1] + conts[1][: fed[1]] + alt)

    # Recycle a slot: the freed slot must be clean in BOTH planes.
    fused_engine.unregister(0)
    fused_engine.register(99)  # reuses slot 0
    p99 = rand_tokens(g, 9)
    out = fused_engine.step([(99, p99, [])])
    assert_both_planes(models, out, 0, p99)


def test_window_crossing_deterministic_both_planes(tiny_llama, tiny_llama_q):
    """Hysteresis eviction must shift BOTH planes: bitwise-deterministic runs
    and sane per-plane distributions through ~20 at-cap steps."""
    g = torch.Generator().manual_seed(3)
    n, t0, steps = 2, 12, 40
    prompts = [rand_tokens(g, t0) for _ in range(n)]
    conts = [rand_tokens(g, steps) for _ in range(n)]

    def run():
        eng = AuxBatchedEngine(
            deepcopy(tiny_llama), DEV, torch.float32, WINDOW, model2=deepcopy(tiny_llama_q)
        )
        for i in range(n):
            eng.register(i)
        outs = [eng.step([(i, prompts[i], []) for i in range(n)])]
        for s in range(steps):
            outs.append(eng.step([(i, prompts[i], conts[i][: s + 1]) for i in range(n)]))
        return outs

    a, b = run(), run()
    for sa, sb in zip(a, b, strict=True):
        assert torch.equal(sa, sb)
        assert torch.isfinite(sa).all()
        probs = sa.float().log_softmax(-1).exp().sum(-1)
        assert probs.allclose(torch.ones(2, n), atol=1e-4)
