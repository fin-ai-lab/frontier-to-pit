"""CPU ground-truth tests for HYBRID (sliding + full attention) aux models —
the FlexLlama v4 arch — through the paged engine.

The reference is the model's own full-context forward (the modeling code
defines the sliding semantics), recomputed from scratch at every step in
deterministic fp32. Covers: prefill below/above the sliding window (the
16-aligned tail scatter), decode through multiple ring evictions, the fused
pair with GENUINELY DIFFERENT precomputed rope caches per plane (the
PairedLookupRotary path — the real flexdoc cooldowns differ exactly like
this), and rewind guards at the ring boundary.
"""

from copy import deepcopy

import pytest
import torch

from ftp import AuxBatchedEngine

DEV = torch.device("cpu")
WINDOW = 128  # engine context budget: large enough that re-prime never fires here
SW = 16  # fixture sliding_window
V = 512


def ref_logits(model, seq: list[int]) -> torch.Tensor:
    with torch.no_grad():
        return model(input_ids=torch.tensor([seq])).logits[0, -1]


def rand_tokens(g: torch.Generator, n: int) -> list[int]:
    return torch.randint(2, V, (n,), generator=g).tolist()


@pytest.fixture()
def flex_pair(tiny_flexllama_factory):
    return tiny_flexllama_factory(0), tiny_flexllama_factory(1)


def test_hybrid_ground_truth_through_evictions(tiny_flexllama_factory):
    """Single hybrid model: short and long prompts, then decode far past the
    sliding window (multiple ring evictions), exact at every step."""
    model = tiny_flexllama_factory(0)
    eng = AuxBatchedEngine(deepcopy(model), DEV, torch.float32, WINDOW)
    g = torch.Generator().manual_seed(1)
    prompts = {0: rand_tokens(g, 9), 1: rand_tokens(g, 40)}  # below / above SW
    conts = {i: rand_tokens(g, 3 * SW) for i in prompts}  # crosses eviction 3x

    for i in prompts:
        eng.register(i)
    out = eng.step([(i, prompts[i], []) for i in prompts])
    for k, i in enumerate(prompts):
        torch.testing.assert_close(
            out[k].float(), ref_logits(model, prompts[i]), rtol=1e-4, atol=1e-4
        )
    for s in range(3 * SW):
        out = eng.step([(i, prompts[i], conts[i][: s + 1]) for i in prompts])
        for k, i in enumerate(prompts):
            ref = ref_logits(model, prompts[i] + conts[i][: s + 1])
            torch.testing.assert_close(out[k].float(), ref, rtol=1e-4, atol=1e-4)


def test_hybrid_fused_planes_keep_their_own_rope(flex_pair):
    """Fused pair whose PRECOMPUTED rope caches differ per plane: each plane
    must match ITS OWN model's full-context reference — this is the
    PairedLookupRotary correctness net (plane 1 on plane 0's rope caches is
    the failure mode the real flexdoc checkpoints forbid)."""
    m0, m1 = flex_pair
    eng = AuxBatchedEngine(deepcopy(m0), DEV, torch.float32, WINDOW, model2=deepcopy(m1))
    g = torch.Generator().manual_seed(2)
    prompts = [rand_tokens(g, 24), rand_tokens(g, 7)]  # one beyond SW, one below
    conts = [rand_tokens(g, 2 * SW) for _ in prompts]

    for i in range(2):
        eng.register(i)
    out = eng.step([(i, prompts[i], []) for i in range(2)])
    for i in range(2):
        torch.testing.assert_close(
            out[0, i].float(), ref_logits(m0, prompts[i]), rtol=1e-4, atol=1e-4
        )
        torch.testing.assert_close(
            out[1, i].float(), ref_logits(m1, prompts[i]), rtol=1e-4, atol=1e-4
        )
    for s in range(2 * SW):
        out = eng.step([(i, prompts[i], conts[i][: s + 1]) for i in range(2)])
        for i in range(2):
            seq = prompts[i] + conts[i][: s + 1]
            torch.testing.assert_close(
                out[0, i].float(), ref_logits(m0, seq), rtol=1e-4, atol=1e-4
            )
            torch.testing.assert_close(
                out[1, i].float(), ref_logits(m1, seq), rtol=1e-4, atol=1e-4
            )


def test_hybrid_pairs_and_rewind(tiny_flexllama_factory):
    """step_pairs bursts and an in-ring rewind stay exact on the hybrid arch;
    a rewind crossing the evicted ring raises."""
    model = tiny_flexllama_factory(0)
    eng = AuxBatchedEngine(deepcopy(model), DEV, torch.float32, WINDOW)
    g = torch.Generator().manual_seed(3)
    prompt = rand_tokens(g, 20)
    eng.register(0)
    eng.step([(0, prompt, [])])

    burst = rand_tokens(g, 2)
    out = eng.step_pairs([(0, burst)])
    torch.testing.assert_close(
        out[0].float(), ref_logits(model, prompt + burst), rtol=1e-4, atol=1e-4
    )

    eng.rewind(0, 2)
    c, d = rand_tokens(g, 2)
    out = eng.step_pairs([(0, [c, d])])
    torch.testing.assert_close(
        out[0].float(), ref_logits(model, prompt + [c, d]), rtol=1e-4, atol=1e-4
    )

    # Push far past the window so the ring evicts, then rewind beyond it.
    outs: list[int] = [c, d]
    for _ in range(3 * SW):
        outs.append(rand_tokens(g, 1)[0])
        eng.step([(0, prompt, outs)])
    st = eng._states[0]
    assert st.sw_drop > 0  # evictions actually happened
    too_far = st.seq_len - st.sw_drop + 1
    with pytest.raises(ValueError, match="sliding window"):
        eng.rewind(0, too_far)


def test_hybrid_page_accounting(tiny_flexllama_factory):
    """The sliding pool stays bounded (ring) while the full pool grows with
    context; unregister returns pages to both pools."""
    model = tiny_flexllama_factory(0)
    eng = AuxBatchedEngine(deepcopy(model), DEV, torch.float32, WINDOW)
    g = torch.Generator().manual_seed(4)
    prompt = rand_tokens(g, 8)
    eng.register(0)
    eng.step([(0, prompt, [])])
    outs: list[int] = []
    for _ in range(4 * SW):
        outs.append(rand_tokens(g, 1)[0])
        eng.step([(0, prompt, outs)])
    st = eng._states[0]
    ring_pages = len(st.pages_sw[0])
    assert ring_pages <= (SW + 16) // 16 + 1  # bounded ring
    assert len(st.pages_full[0]) >= (8 + 4 * SW) // 16  # full retains all
    free_full = len(eng._pool_full.free)
    free_sw = len(eng._pool_sw.free)
    eng.unregister(0)
    assert len(eng._pool_full.free) > free_full
    assert len(eng._pool_sw.free) > free_sw
