"""End-to-end CPU ground truth for the universal-decoding bridge.

Two genuinely different tokenizers (P side vs aux side) and two different tiny
aux models drive UniversalBridge + two real AuxBatchedEngines; every P step's
mapped (l_p, l_q) is compared against an offline reference that re-encodes the
full text from scratch, runs full-context forwards, and maps through the same
VocabMapper. This pins the whole chain: incremental retokenization, engine
rewinds, the drain loop, d=0 logits reuse, and the log-prob mapping.
"""

from copy import deepcopy

import pytest
import torch

from ftp import AuxBatchedEngine
from ftp.translate import (
    ChatTemplateAdapter,
    StreamRetokenizer,
    TokenTextTable,
    UniversalBridge,
    VocabMapper,
    _utf8_complete_len,
)

DEV = torch.device("cpu")
WINDOW = 64
V_P = 448  # >= P tokenizer width; stands in for vLLM's padded logits width


class _NoopEngine:
    """Stands in the aux_q seat when the rig runs a FUSED engine: the single
    engine in the aux_p seat owns registration for both planes, so the tests'
    aux_q.register/unregister calls become no-ops."""

    def register(self, rid):
        pass

    def unregister(self, rid):
        pass


def bridge_aux_q(aux_p, aux_q):
    """The aux_q constructor argument for a second bridge over rig engines:
    None when aux_p is a fused pair."""
    return None if getattr(aux_p, "_replicas", 1) == 2 else aux_q


@pytest.fixture(params=["pair", "fused"])
def rig(request, tiny_tok_p, tiny_tok_aux, tiny_llama_factory):
    table = TokenTextTable(tiny_tok_p)
    mapper = VocabMapper(tiny_tok_aux, table, DEV)
    model_p, model_q = tiny_llama_factory(1), tiny_llama_factory(2)
    if request.param == "fused":
        aux_p = AuxBatchedEngine(
            deepcopy(model_p), DEV, torch.float32, WINDOW, model2=deepcopy(model_q)
        )
        aux_q = _NoopEngine()
    else:
        aux_p = AuxBatchedEngine(model_p, DEV, torch.float32, WINDOW)
        aux_q = AuxBatchedEngine(model_q, DEV, torch.float32, WINDOW)
    bridge = UniversalBridge(
        tiny_tok_aux,
        table,
        mapper,
        aux_p,
        bridge_aux_q(aux_p, aux_q),
        max_feeds_per_step=8,
        window=WINDOW,
    )
    return table, mapper, model_p, model_q, aux_p, aux_q, bridge


def ref_row(model, tiny_tok_aux, table, mapper, p_ids) -> torch.Tensor:
    """Offline reference: full re-encode + full-context forward + mapping."""
    raw = b"".join(table.bytes_for(i) for i in p_ids)
    text = raw[: _utf8_complete_len(raw)].decode("utf-8", "replace")
    ids = [tiny_tok_aux.bos_token_id] + tiny_tok_aux(text, add_special_tokens=False)["input_ids"]
    ids = ids[-WINDOW:]
    with torch.no_grad():
        logits = model(input_ids=torch.tensor([ids])).logits[0, -1]
    return mapper.map_logits(logits.unsqueeze(0), V_P)[0]


def check(rig_tuple, reqs):
    table, mapper, model_p, model_q, _, _, bridge = rig_tuple
    lp, lq = bridge.step(reqs, V_P)
    assert lp.shape == lq.shape == (len(reqs), V_P)
    for j, (_rid, p, o) in enumerate(reqs):
        full = list(p or []) + list(o)
        ref_p = ref_row(model_p, bridge._aux_tok, table, mapper, full)
        ref_q = ref_row(model_q, bridge._aux_tok, table, mapper, full)
        torch.testing.assert_close(lp[j], ref_p, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(lq[j], ref_q, rtol=1e-4, atol=1e-4)
    return lp, lq


def p_stream(tok_p, text):
    return tok_p(text, add_special_tokens=False)["input_ids"]


TEXTS = [
    "The quick brown fox jumps over the lazy dog. Pack my box.",
    "In 2024, prices rose by 12.5% across 1,234 categories ====",
    "naïve café 🚀 strawberry fields def main(): return 42",
]


def test_bridge_ground_truth_staggered(tiny_tok_p, rig):
    """Three requests with different texts, one joining two steps late; every
    step's mapped distributions match the offline reference exactly."""
    _, _, _, _, aux_p, aux_q, bridge = rig
    streams = [p_stream(tiny_tok_p, t) for t in TEXTS]
    prompts = [s[:4] for s in streams]
    conts = [s[4:] for s in streams]

    for rid in (0, 1):
        aux_p.register(rid)
        aux_q.register(rid)
    check(rig, [(0, prompts[0], []), (1, prompts[1], [])])

    n_steps = min(len(c) for c in conts)
    for s in range(n_steps):
        reqs = [(0, prompts[0], conts[0][: s + 1]), (1, prompts[1], conts[1][: s + 1])]
        if s == 2:  # late joiner: fresh registration mid-run, prefills via bridge
            aux_p.register(2)
            aux_q.register(2)
        if s >= 2:
            reqs.append((2, prompts[2], conts[2][: s - 2] if s > 2 else []))
        check(rig, reqs)


def test_bridge_d0_and_multibyte(tiny_tok_p, rig):
    """Emoji split across P tokens: steps with no new decodable text must
    reuse the buffered logits and still match the reference (whose text also
    hasn't changed)."""
    _, _, _, _, aux_p, aux_q, bridge = rig
    stream = p_stream(tiny_tok_p, "ab 🚀🔥 cd the fox")
    aux_p.register(0)
    aux_q.register(0)
    check(rig, [(0, stream[:2], [])])
    for s in range(len(stream) - 2):
        check(rig, [(0, stream[:2], stream[2 : s + 3])])


def test_bridge_backlog_drains(tiny_tok_p, tiny_tok_aux, rig):
    """With max_feeds_per_step=1, a multi-aux-token push leaves a backlog
    (lagging logits); a follow-up step with no new P token drains it back to
    ground truth."""
    table, mapper, model_p, model_q, aux_p, aux_q, _ = rig
    bridge = UniversalBridge(
        tiny_tok_aux,
        table,
        mapper,
        aux_p,
        bridge_aux_q(aux_p, aux_q),
        max_feeds_per_step=1,
        window=WINDOW,
    )
    # find a P token whose push yields >= 2 aux tokens
    stream = p_stream(tiny_tok_p, TEXTS[0])
    probe = StreamRetokenizer(tiny_tok_aux, table)
    probe.reset(stream[:4])
    burst = None
    for k, pid in enumerate(stream[4:]):
        _, new = probe.push(pid)
        if len(new) >= 2:
            burst = k
            break
    assert burst is not None

    aux_p.register(9)
    aux_q.register(9)
    bridge.step([(9, stream[:4], [])], V_P)
    for s in range(burst):
        bridge.step([(9, stream[:4], stream[4 : 5 + s])], V_P)

    reqs = [(9, stream[:4], stream[4 : 5 + burst])]
    bridge.step(reqs, V_P)
    assert bridge._states[9].pending  # lagging by design
    # same reqs again (no P growth) → backlog drains → ground truth
    lp, lq = bridge.step(reqs, V_P)
    assert not bridge._states[9].pending
    full = stream[: 5 + burst]
    torch.testing.assert_close(
        lp[0], ref_row(model_p, tiny_tok_aux, table, mapper, full), rtol=1e-4, atol=1e-4
    )
    torch.testing.assert_close(
        lq[0], ref_row(model_q, tiny_tok_aux, table, mapper, full), rtol=1e-4, atol=1e-4
    )


def test_bridge_pause_and_resume(tiny_tok_p, rig):
    """A row absent from a step (vLLM recompute pause) is untouched and
    resumes on ground truth."""
    _, _, _, _, aux_p, aux_q, bridge = rig
    s0, s1 = p_stream(tiny_tok_p, TEXTS[0]), p_stream(tiny_tok_p, TEXTS[1])
    for rid in (0, 1):
        aux_p.register(rid)
        aux_q.register(rid)
    check(rig, [(0, s0[:4], []), (1, s1[:4], [])])
    check(rig, [(0, s0[:4], s0[4:5]), (1, s1[:4], s1[4:5])])
    for s in range(1, 4):  # rid 1 paused
        check(rig, [(0, s0[:4], s0[4 : 5 + s])])
    check(rig, [(0, s0[:4], s0[4:9]), (1, s1[:4], s1[4:6])])


def test_bridge_mark_reset(tiny_tok_p, rig):
    """mark_reset + engine re-registration (the vLLM gap path) rebuilds the
    stream from full context and stays on ground truth."""
    _, _, _, _, aux_p, aux_q, bridge = rig
    stream = p_stream(tiny_tok_p, TEXTS[2])
    aux_p.register(0)
    aux_q.register(0)
    check(rig, [(0, stream[:4], [])])
    for s in range(3):
        check(rig, [(0, stream[:4], stream[4 : 5 + s])])
    # the processor's gap branch: re-register engines + mark_reset, and the
    # output may have jumped several tokens
    for eng in (aux_p, aux_q):
        eng.unregister(0)
        eng.register(0)
    bridge.mark_reset(0)
    check(rig, [(0, stream[:4], stream[4:12])])
    check(rig, [(0, stream[:4], stream[4:13])])


@pytest.mark.parametrize("flavor", ["pair", "fused"])
def test_bridge_identical_engines_zero_shift(flavor, tiny_tok_p, tiny_tok_aux, tiny_llama_factory):
    """aux_p == aux_q ⇒ the mapped shift l_q − l_p is exactly zero — the same
    assertion the GPU smoke test uses end-to-end through vLLM."""
    table = TokenTextTable(tiny_tok_p)
    mapper = VocabMapper(tiny_tok_aux, table, DEV)
    model = tiny_llama_factory(3)
    if flavor == "fused":
        aux_p = AuxBatchedEngine(
            deepcopy(model), DEV, torch.float32, WINDOW, model2=deepcopy(model)
        )
        aux_q, aux_q_arg = _NoopEngine(), None
    else:
        aux_p = AuxBatchedEngine(model, DEV, torch.float32, WINDOW)
        aux_q = aux_q_arg = AuxBatchedEngine(model, DEV, torch.float32, WINDOW)
    bridge = UniversalBridge(
        tiny_tok_aux, table, mapper, aux_p, aux_q_arg, max_feeds_per_step=8, window=WINDOW
    )
    stream = p_stream(tiny_tok_p, TEXTS[0])
    aux_p.register(0)
    aux_q.register(0)

    # Two separate engines run the same forward twice → bitwise equal. The fused
    # engine computes both planes in ONE batched bmm, where bitwise plane equality
    # is a BLAS implementation detail (plane 1's alignment can change accumulation
    # order on some CPUs) — that flavor asserts equality only up to float noise.
    def check(lp, lq):
        if flavor == "fused":
            assert torch.allclose(lp, lq, rtol=0, atol=1e-5)
        else:
            assert torch.equal(lp, lq)

    lp, lq = bridge.step([(0, stream[:4], [])], V_P)
    check(lp, lq)
    for s in range(6):
        lp, lq = bridge.step([(0, stream[:4], stream[4 : 5 + s])], V_P)
        check(lp, lq)


def ref_row_prefix(model, tiny_tok_aux, table, mapper, prefix_ids, gen_p_ids) -> torch.Tensor:
    """Adapter reference: a FROZEN aux prefix + retokenized generation, run as a
    full-context forward (clipped to the aux window, exactly as the engine
    slides). Mirrors reset_with_prefix's prefix + aux_tok(gen) invariant."""
    raw = b"".join(table.bytes_for(i) for i in gen_p_ids)
    gen_text = raw[: _utf8_complete_len(raw)].decode("utf-8", "replace")
    ids = (prefix_ids + tiny_tok_aux(gen_text, add_special_tokens=False)["input_ids"])[-WINDOW:]
    with torch.no_grad():
        logits = model(input_ids=torch.tensor([ids])).logits[0, -1]
    return mapper.map_logits(logits.unsqueeze(0), V_P)[0]


def test_bridge_adapter_retemplates_prompt(tiny_tok_p, tiny_tok_aux, tiny_llama_factory):
    """End-to-end adapter path: the bridge re-renders P's prompt under the aux
    chat template, freezes it, and retokenizes only P's generation — matching a
    reference forward over prefix + aux_tok(gen) at every step (and sliding the
    prefix out of the aux window identically once gen overruns it)."""
    table = TokenTextTable(tiny_tok_p)
    mapper = VocabMapper(tiny_tok_aux, table, DEV)
    model_p, model_q = tiny_llama_factory(1), tiny_llama_factory(2)
    aux_p = AuxBatchedEngine(model_p, DEV, torch.float32, WINDOW)
    aux_q = AuxBatchedEngine(model_q, DEV, torch.float32, WINDOW)
    adapter = ChatTemplateAdapter(user_open="<start_of_turn>user\n", user_close="<end_of_turn>")
    bridge = UniversalBridge(
        tiny_tok_aux, table, mapper, aux_p, aux_q,
        max_feeds_per_step=8, window=WINDOW, adapter=adapter,
    )
    old = getattr(tiny_tok_aux, "chat_template", None)
    tiny_tok_aux.chat_template = (
        "{% for m in messages %}<|aux|>{{ m['role'] }}\n{{ m['content'] }}<|end|>\n"
        "{% endfor %}{% if add_generation_prompt %}<|aux|>assistant\n{% endif %}"
    )
    try:
        p_prompt = "<start_of_turn>user\nThe quick brown fox?<end_of_turn>\n<start_of_turn>model\n"
        p_ids = p_stream(tiny_tok_p, p_prompt)
        gen = p_stream(tiny_tok_p, "Jumps over the lazy dog.")
        prefix_text = adapter.aux_prefix_text(p_prompt, tiny_tok_aux)
        # the bridge sees P's MARKUP, never the structured turn -> it must
        # reconstruct exactly this aux prefix from the detokenized prompt
        assert prefix_text == "<|aux|>user\nThe quick brown fox?<|end|>\n<|aux|>assistant\n"
        prefix_ids = tiny_tok_aux(prefix_text, add_special_tokens=False)["input_ids"]

        aux_p.register(0)
        aux_q.register(0)
        bridge.step([(0, p_ids, [])], V_P)
        # the frozen aux prefix is the re-rendered prompt, NOT P's markup
        assert bridge._states[0].retok.aux_ids[: len(prefix_ids)] == prefix_ids
        for s in range(len(gen)):
            lp, lq = bridge.step([(0, p_ids, gen[: s + 1])], V_P)
            ref_p = ref_row_prefix(model_p, tiny_tok_aux, table, mapper, prefix_ids, gen[: s + 1])
            ref_q = ref_row_prefix(model_q, tiny_tok_aux, table, mapper, prefix_ids, gen[: s + 1])
            torch.testing.assert_close(lp[0], ref_p, rtol=1e-4, atol=1e-4)
            torch.testing.assert_close(lq[0], ref_q, rtol=1e-4, atol=1e-4)
    finally:
        tiny_tok_aux.chat_template = old


def test_bridge_adapter_reasoning_rebase(tiny_tok_p, tiny_tok_aux, tiny_llama_factory):
    """Reasoning P: the CoT streams verbatim, then at P's end-of-think the bridge
    re-bases ONCE so the aux read a Qwen-native think trace (P's delimiters ->
    <think>/</think>, CoT kept). Final aux stream must equal
    prefix + aux_tok(translate_reasoning(full output)), and the mapped logits a
    reference forward over those ids."""
    table = TokenTextTable(tiny_tok_p)
    mapper = VocabMapper(tiny_tok_aux, table, DEV)
    model_p, model_q = tiny_llama_factory(1), tiny_llama_factory(2)
    aux_p = AuxBatchedEngine(model_p, DEV, torch.float32, WINDOW)
    aux_q = AuxBatchedEngine(model_q, DEV, torch.float32, WINDOW)
    adapter = ChatTemplateAdapter(
        user_open="<U>", user_close="</U>", reason_open="<A>", reason_close="<B>",
    )
    bridge = UniversalBridge(
        tiny_tok_aux, table, mapper, aux_p, aux_q,
        max_feeds_per_step=8, window=WINDOW, adapter=adapter,
    )
    old = getattr(tiny_tok_aux, "chat_template", None)
    tiny_tok_aux.chat_template = (
        "{% for m in messages %}<|aux|>{{ m['role'] }}\n{{ m['content'] }}<|end|>\n"
        "{% endfor %}{% if add_generation_prompt %}<|aux|>assistant\n{% endif %}"
    )
    try:
        p_prompt = "<U>solve</U>"
        p_ids = p_stream(tiny_tok_p, p_prompt)
        gen = p_stream(tiny_tok_p, "<A>reason<B>answer")
        prefix_ids = tiny_tok_aux(
            adapter.aux_prefix_text(p_prompt, tiny_tok_aux), add_special_tokens=False
        )["input_ids"]
        aux_p.register(0)
        aux_q.register(0)
        bridge.step([(0, p_ids, [])], V_P)
        lp = lq = None
        for s in range(len(gen)):
            lp, lq = bridge.step([(0, p_ids, gen[: s + 1])], V_P)

        st = bridge._states[0]
        assert st.post_reason  # crossed end-of-think and re-based exactly once
        full = b"".join(table.bytes_for(i) for i in gen).decode("utf-8", "replace")
        translated = adapter.translate_reasoning(full)
        assert translated == "<think>\nreason\n</think>\n\nanswer"  # CoT kept, delims swapped
        ref_ids = prefix_ids + tiny_tok_aux(translated, add_special_tokens=False)["input_ids"]
        assert st.retok.aux_ids == ref_ids
        assert len(ref_ids) <= WINDOW  # keep the exact-logits check off the sliding path
        with torch.no_grad():
            rp = model_p(input_ids=torch.tensor([ref_ids[-WINDOW:]])).logits[0, -1]
        torch.testing.assert_close(lp[0], mapper.map_logits(rp.unsqueeze(0), V_P)[0],
                                   rtol=1e-4, atol=1e-4)
    finally:
        tiny_tok_aux.chat_template = old


def test_bridge_drop_recycles_slots(tiny_tok_p, rig):
    _, _, _, _, aux_p, aux_q, bridge = rig
    stream = p_stream(tiny_tok_p, TEXTS[0])
    for rid in (0, 1):
        aux_p.register(rid)
        aux_q.register(rid)
    check(rig, [(0, stream[:4], []), (1, stream[:6], [])])
    slot0 = bridge._slot[0]
    aux_p.unregister(0)
    aux_q.unregister(0)
    bridge.drop(0)
    aux_p.register(7)
    aux_q.register(7)
    check(rig, [(7, stream[:5], []), (1, stream[:6], stream[6:7])])
    assert bridge._slot[7] == slot0  # lowest free slot reused


def test_bridge_two_phase_with_late_rows(tiny_tok_p, rig):
    """Overlap mode contract: prefetch(rows) ... prefetch(late) then
    finalize(V, rids=combined) must equal a single step over all rows."""
    table, mapper, model_p, model_q, aux_p, aux_q, bridge = rig
    s0, s1 = p_stream(tiny_tok_p, TEXTS[0]), p_stream(tiny_tok_p, TEXTS[1])
    for rid in (0, 1):
        aux_p.register(rid)
        aux_q.register(rid)
    # both rows prefill via one normal step
    check(rig, [(0, s0[:4], []), (1, s1[:4], [])])
    # next P step: row 0 known at "update_state", row 1 arrives late
    bridge.prefetch([(0, s0[:4], s0[4:5])])
    bridge.prefetch([(1, s1[:4], s1[4:5])])
    lp, lq = bridge.finalize(V_P, rids=[0, 1])
    torch.testing.assert_close(
        lp[0], ref_row(model_p, bridge._aux_tok, table, mapper, s0[:5]), rtol=1e-4, atol=1e-4
    )
    torch.testing.assert_close(
        lp[1], ref_row(model_p, bridge._aux_tok, table, mapper, s1[:5]), rtol=1e-4, atol=1e-4
    )
    torch.testing.assert_close(
        lq[1], ref_row(model_q, bridge._aux_tok, table, mapper, s1[:5]), rtol=1e-4, atol=1e-4
    )
