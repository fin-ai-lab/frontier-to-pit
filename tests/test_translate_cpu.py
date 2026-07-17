"""CPU tests for the universal-decoding translation layer.

The two tiny BPE tokenizers (conftest) are trained on different corpora with
different vocab sizes and specials, so they genuinely segment the same text
differently — every push exercises real retokenization boundaries.
"""

import pytest
import torch

from ftp.translate import (
    ChatTemplateAdapter,
    StreamRetokenizer,
    TokenTextTable,
    VocabMapper,
    _lcp,
    _utf8_complete_len,
    make_chat_adapter,
)

SAMPLES = [
    "The quick brown fox jumps over the lazy dog.",
    "In 2024, prices rose by 12.5% across 1,234 categories.",
    "def main():\n    return [i**2 for i in range(10)]\n",
    "Strawberries grow in strawberry fields ====---- 0000",
    "naïve café — déjà vu 🚀 日本語",
]


# ── TokenTextTable ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("text", SAMPLES)
def test_table_byte_exact_byte_level(tiny_tok_p, text):
    """Concatenated per-id bytes must reproduce the encoded text exactly."""
    table = TokenTextTable(tiny_tok_p)
    ids = tiny_tok_p(text, add_special_tokens=False)["input_ids"]
    assert b"".join(table.bytes_for(i) for i in ids) == text.encode("utf-8")


def test_table_sp_pieces(tiny_tok_sp):
    table = TokenTextTable(tiny_tok_sp)
    vocab = tiny_tok_sp.get_vocab()
    assert table.bytes_of[vocab["▁strawberry"]] == b" strawberry"
    assert table.bytes_of[vocab["berry"]] == b"berry"
    assert table.bytes_of[vocab["<0x41>"]] == b"A"
    assert table.bytes_of[vocab["<0xE2>"]] == b"\xe2"


def test_table_specials_surface_text(tiny_tok_p):
    table = TokenTextTable(tiny_tok_p)
    vocab = tiny_tok_p.get_vocab()
    sid = vocab["<|special|>"]
    assert sid in table.special_ids
    assert table.bytes_of[sid] == b"<|special|>"


def test_table_out_of_range(tiny_tok_p):
    table = TokenTextTable(tiny_tok_p)
    assert table.bytes_for(-1) == b""
    assert table.bytes_for(10**6) == b""


# ── helpers ──────────────────────────────────────────────────────────────────


def test_utf8_complete_len():
    rocket = "🚀".encode()  # 4 bytes
    assert _utf8_complete_len(b"abc") == 3
    for cut in range(1, 4):
        assert _utf8_complete_len(b"ab" + rocket[:cut]) == 2
    assert _utf8_complete_len(b"ab" + rocket) == 6
    accent = "é".encode()  # 2 bytes
    assert _utf8_complete_len(accent[:1]) == 0
    assert _utf8_complete_len(accent) == 2
    assert _utf8_complete_len(b"") == 0


@pytest.mark.parametrize(
    ("old", "new", "want"),
    [
        ([1, 2, 3], [1, 2, 3], 3),
        ([1, 2, 3], [1, 2, 4, 5], 2),
        ([1, 2, 3], [9], 0),
        ([], [1], 0),
        (list(range(1000)) + [7], list(range(1000)) + [8, 9], 1000),
        ([5] + list(range(1000)), [6] + list(range(1000)), 0),
    ],
)
def test_lcp(old, new, want):
    assert _lcp(old, new, 4) == want


# ── StreamRetokenizer ────────────────────────────────────────────────────────


def drive(retok, deltas_target, p_ids):
    """Push p_ids one at a time, applying each (rewind, new) delta to a shadow
    list — the exact contract the engine bridge relies on."""
    shadow = list(retok.aux_ids)
    max_rewind = 0
    for pid in p_ids:
        rewind, new = retok.push(pid)
        max_rewind = max(max_rewind, rewind)
        if rewind:
            del shadow[len(shadow) - rewind :]
        shadow.extend(new)
        assert shadow == retok.aux_ids
        deltas_target.append((rewind, new))
    return max_rewind


@pytest.mark.parametrize("text", SAMPLES)
def test_retokenizer_invariant(tiny_tok_p, tiny_tok_aux, text):
    """At every push: committed ids == aux BOS + aux_tok(full decoded text),
    and the (rewind, new) deltas reconstruct the committed list exactly."""
    table = TokenTextTable(tiny_tok_p)
    retok = StreamRetokenizer(tiny_tok_aux, table)
    p_ids = tiny_tok_p(text, add_special_tokens=False)["input_ids"]
    prompt, stream = p_ids[:3], p_ids[3:]
    retok.reset(prompt)

    bos = [tiny_tok_aux.bos_token_id]
    shadow = list(retok.aux_ids)
    for s, pid in enumerate(stream):
        rewind, new = retok.push(pid)
        if rewind:
            del shadow[len(shadow) - rewind :]
        shadow.extend(new)
        assert shadow == retok.aux_ids
        # ground truth: full re-encode of the decodable text so far
        decodable = b"".join(table.bytes_for(i) for i in prompt + stream[: s + 1])
        n = _utf8_complete_len(decodable)
        full_text = decodable[:n].decode("utf-8", "replace")
        ref = bos + tiny_tok_aux(full_text, add_special_tokens=False)["input_ids"]
        assert retok.aux_ids == ref


@pytest.mark.parametrize("reenc", [4, 8, 48])
def test_retokenizer_bounded_long_context(tiny_tok_p, tiny_tok_aux, reenc):
    """Bounded-suffix re-encode: on a stream far longer than the freeze
    window, the frozen-prefix splice must keep committed == aux BOS +
    aux_tok(full_text) at EVERY push. A small window stresses the freeze
    boundary hard (this is the path the short SAMPLES never reach)."""
    table = TokenTextTable(tiny_tok_p)
    long_text = " ".join(s.strip() for s in SAMPLES) * 8  # >> any window
    p_ids = tiny_tok_p(long_text, add_special_tokens=False)["input_ids"]
    retok = StreamRetokenizer(tiny_tok_aux, table, reenc_window=reenc)
    retok.reset([])
    bos = [tiny_tok_aux.bos_token_id]
    froze = False
    for s, pid in enumerate(p_ids):
        retok.push(pid)
        froze = froze or retok._frozen > len(retok._prefix)
        decodable = b"".join(table.bytes_for(i) for i in p_ids[: s + 1])
        n = _utf8_complete_len(decodable)
        full_text = decodable[:n].decode("utf-8", "replace")
        ref = bos + tiny_tok_aux(full_text, add_special_tokens=False)["input_ids"]
        assert retok.aux_ids == ref, f"diverged at push {s} (reenc={reenc})"
    assert froze, "freeze boundary never advanced — bounded path not exercised"


def test_retokenizer_exact_mode_fallback(tiny_tok_p, tiny_tok_aux):
    """A non-fast tokenizer (no char offsets) keeps the exact full-text path;
    the invariant must still hold and freezing must stay disabled."""
    table = TokenTextTable(tiny_tok_p)
    retok = StreamRetokenizer(tiny_tok_aux, table, reenc_window=4)
    retok._bounded = False  # simulate a tokenizer without offset mapping
    long_text = " ".join(s.strip() for s in SAMPLES) * 8
    p_ids = tiny_tok_p(long_text, add_special_tokens=False)["input_ids"]
    retok.reset([])
    bos = [tiny_tok_aux.bos_token_id]
    for s, pid in enumerate(p_ids):
        retok.push(pid)
        assert retok._frozen == len(retok._prefix)  # never freezes
        decodable = b"".join(table.bytes_for(i) for i in p_ids[: s + 1])
        n = _utf8_complete_len(decodable)
        full_text = decodable[:n].decode("utf-8", "replace")
        ref = bos + tiny_tok_aux(full_text, add_special_tokens=False)["input_ids"]
        assert retok.aux_ids == ref


def test_retokenizer_rewind_happens(tiny_tok_p, tiny_tok_aux):
    """The fixtures must actually produce non-causal retokenization (rewinds);
    otherwise the suite isn't testing the hard case."""
    table = TokenTextTable(tiny_tok_p)
    total_rewinds = 0
    for text in SAMPLES:
        retok = StreamRetokenizer(tiny_tok_aux, table)
        retok.reset([])
        p_ids = tiny_tok_p(text, add_special_tokens=False)["input_ids"]
        deltas = []
        drive(retok, deltas, p_ids)
        total_rewinds += sum(1 for r, _ in deltas if r > 0)
    assert total_rewinds > 0


def test_retokenizer_multibyte_holdback(tiny_tok_p, tiny_tok_aux):
    """A P id stream that splits a multi-byte char mid-sequence must yield a
    d=0 push (held-back bytes), then catch up exactly."""
    table = TokenTextTable(tiny_tok_p)
    retok = StreamRetokenizer(tiny_tok_aux, table)
    retok.reset([])

    text = "ab 🚀 cd"
    p_ids = tiny_tok_p(text, add_special_tokens=False)["input_ids"]
    # The 4-byte rocket can't be a single trained token of the tiny vocab, so
    # some push mid-emoji must produce no new decodable text.
    saw_d0 = False
    for pid in p_ids:
        _, new = retok.push(pid)
        if not new and table.bytes_for(pid):
            saw_d0 = True
    assert saw_d0
    ref = [tiny_tok_aux.bos_token_id] + tiny_tok_aux(text, add_special_tokens=False)["input_ids"]
    assert retok.aux_ids == ref


def test_retokenizer_specials_as_text(tiny_tok_p, tiny_tok_aux):
    """P's special tokens stream through as their surface text."""
    table = TokenTextTable(tiny_tok_p)
    retok = StreamRetokenizer(tiny_tok_aux, table)
    retok.reset([])
    sid = tiny_tok_p.get_vocab()["<|special|>"]
    retok.push(sid)
    ref = [tiny_tok_aux.bos_token_id] + tiny_tok_aux("<|special|>", add_special_tokens=False)[
        "input_ids"
    ]
    assert retok.aux_ids == ref


def test_retokenizer_reset_mid_stream(tiny_tok_p, tiny_tok_aux):
    """reset() with prompt+outputs (the vLLM recompute-gap case) must agree
    with a stream that pushed the same ids."""
    table = TokenTextTable(tiny_tok_p)
    text = SAMPLES[0]
    p_ids = tiny_tok_p(text, add_special_tokens=False)["input_ids"]

    streamed = StreamRetokenizer(tiny_tok_aux, table)
    streamed.reset(p_ids[:4])
    for pid in p_ids[4:]:
        streamed.push(pid)

    fresh = StreamRetokenizer(tiny_tok_aux, table)
    ids = fresh.reset(p_ids)
    assert ids == streamed.aux_ids


# ── reset_with_prefix (chat-template swap) ───────────────────────────────────


def test_reset_with_prefix_invariant(tiny_tok_p, tiny_tok_aux):
    """reset_with_prefix freezes an externally supplied aux prefix and only
    retokenizes P's generated tokens. The maintained invariant is
    committed == prefix + aux_tok(generated_text) — NOT aux_tok(prefix+gen):
    the prompt/response seam is tokenized separately (matching the aux SFT)."""
    table = TokenTextTable(tiny_tok_p)
    gen_text = SAMPLES[1]
    gen_ids = tiny_tok_p(gen_text, add_special_tokens=False)["input_ids"]
    prefix = tiny_tok_aux("a rendered prompt prefix\n", add_special_tokens=False)["input_ids"]

    retok = StreamRetokenizer(tiny_tok_aux, table)
    assert retok.reset_with_prefix(prefix, []) == prefix  # no generation yet

    for s, pid in enumerate(gen_ids):
        retok.push(pid)
        decodable = b"".join(table.bytes_for(i) for i in gen_ids[: s + 1])
        n = _utf8_complete_len(decodable)
        gen_so_far = decodable[:n].decode("utf-8", "replace")
        ref = prefix + tiny_tok_aux(gen_so_far, add_special_tokens=False)["input_ids"]
        assert retok.aux_ids == ref
        assert retok.aux_ids[: len(prefix)] == prefix  # prefix never rewinds


@pytest.mark.parametrize("reenc", [4, 8, 48])
def test_reset_with_prefix_bounded_long(tiny_tok_p, tiny_tok_aux, reenc):
    """Long generation after a frozen prefix — the bounded-suffix re-encode that
    feeds the sliding-window KV cache its (rewind, new) deltas. committed must
    stay prefix + aux_tok(full_generation) at EVERY push, and the frozen prefix
    must advance INTO the generated region (never before it)."""
    table = TokenTextTable(tiny_tok_p)
    prefix = tiny_tok_aux("system + user turn\n", add_special_tokens=False)["input_ids"]
    long_gen = " ".join(s.strip() for s in SAMPLES) * 8  # >> any window
    gen_ids = tiny_tok_p(long_gen, add_special_tokens=False)["input_ids"]
    retok = StreamRetokenizer(tiny_tok_aux, table, reenc_window=reenc)
    retok.reset_with_prefix(prefix, [])
    froze = False
    for s, pid in enumerate(gen_ids):
        retok.push(pid)
        froze = froze or retok._frozen > len(prefix)
        decodable = b"".join(table.bytes_for(i) for i in gen_ids[: s + 1])
        n = _utf8_complete_len(decodable)
        gen_so_far = decodable[:n].decode("utf-8", "replace")
        ref = prefix + tiny_tok_aux(gen_so_far, add_special_tokens=False)["input_ids"]
        assert retok.aux_ids == ref, f"diverged at push {s} (reenc={reenc})"
        assert retok.aux_ids[: len(prefix)] == prefix
    assert froze, "freeze boundary never advanced past the prefix"


def test_reset_with_prefix_resume_matches_stream(tiny_tok_p, tiny_tok_aux):
    """A mid-stream rebuild (the vLLM gap / id-collision path) via
    reset_with_prefix(prefix, generated_ids) must agree with a stream that
    pushed the same generated tokens after the same prefix."""
    table = TokenTextTable(tiny_tok_p)
    gen_ids = tiny_tok_p(SAMPLES[0], add_special_tokens=False)["input_ids"]
    prefix = tiny_tok_aux("prompt\n", add_special_tokens=False)["input_ids"]

    streamed = StreamRetokenizer(tiny_tok_aux, table)
    streamed.reset_with_prefix(prefix, [])
    for pid in gen_ids:
        streamed.push(pid)

    fresh = StreamRetokenizer(tiny_tok_aux, table)
    assert fresh.reset_with_prefix(prefix, gen_ids) == streamed.aux_ids


# ── ChatTemplateAdapter ──────────────────────────────────────────────────────


def test_chat_adapter_extracts_user_turn():
    adapter = ChatTemplateAdapter(user_open="<start_of_turn>user\n", user_close="<end_of_turn>")
    p_prompt = "<bos><start_of_turn>user\nWhat is 2+2?<end_of_turn>\n<start_of_turn>model\n"
    assert adapter.user_content(p_prompt) == "What is 2+2?"
    assert adapter.user_content("no chat markup at all") is None  # caller falls back


def test_chat_adapter_rerenders_under_aux_template(tiny_tok_aux):
    adapter = ChatTemplateAdapter(user_open="<start_of_turn>user\n", user_close="<end_of_turn>")
    p_prompt = "<bos><start_of_turn>user\nhello there<end_of_turn>\n<start_of_turn>model\n"
    old = getattr(tiny_tok_aux, "chat_template", None)
    tiny_tok_aux.chat_template = (
        "{% for m in messages %}<|aux|>{{ m['role'] }}\n{{ m['content'] }}<|end|>\n"
        "{% endfor %}{% if add_generation_prompt %}<|aux|>assistant\n{% endif %}"
    )
    try:
        text = adapter.aux_prefix_text(p_prompt, tiny_tok_aux)
        assert text == "<|aux|>user\nhello there<|end|>\n<|aux|>assistant\n"
        assert adapter.aux_prefix_text("no markup", tiny_tok_aux) is None
    finally:
        tiny_tok_aux.chat_template = old


def test_make_chat_adapter_presets():
    assert make_chat_adapter(None) is None
    assert make_chat_adapter("") is None
    gemma = make_chat_adapter("gemma")
    assert gemma.user_open == "<start_of_turn>user\n"
    assert gemma.template_kwargs == {"enable_thinking": False}
    with pytest.raises(ValueError):
        make_chat_adapter("not-a-preset")


# Each preset must slice the user turn out of its P family's rendered prompt.
_PRESET_PROMPTS = {
    "gemma": "<bos><start_of_turn>user\nHELLO<end_of_turn>\n<start_of_turn>model\n",
    "llama3": (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nSys<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\nHELLO<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    ),
    "gptoss": (
        "<|start|>system<|message|>Sys<|end|>"
        "<|start|>developer<|message|>Dev<|end|>"
        "<|start|>user<|message|>HELLO<|end|><|start|>assistant"
    ),
}


@pytest.mark.parametrize("name", ["gemma", "llama3", "gptoss"])
def test_preset_extracts_user_turn(name):
    adapter = make_chat_adapter(name)
    assert adapter.user_content(_PRESET_PROMPTS[name]) == "HELLO"


def test_translate_reasoning_rewrites_delimiters():
    a = ChatTemplateAdapter(
        user_open="<|start|>user<|message|>", user_close="<|end|>",
        reason_open="<A>", reason_close="<B>",
    )
    assert a.translate_reasoning("<A>chain of thought so far") is None  # still thinking
    out = a.translate_reasoning("<A>chain of thought<B>the answer")
    assert out == "<think>\nchain of thought\n</think>\n\nthe answer"  # CoT kept, delims swapped


def test_translate_reasoning_noop_for_nonreasoning_adapter():
    a = make_chat_adapter("gemma")  # reason_close is None
    assert a.translate_reasoning("anything <B> here") is None


# ── VocabMapper ──────────────────────────────────────────────────────────────


def aux_width(tok) -> int:
    return max(tok.get_vocab().values()) + 1


@pytest.mark.parametrize("aux_fixture", ["tiny_tok_aux", "tiny_tok_sp"])
def test_mapper_first_token_prefix_property(tiny_tok_p, aux_fixture, request):
    """Every covered P token must map to an aux token whose bytes are a
    non-empty prefix of the P token's bytes (the defining property of a
    first-token map) — checked for both byte-level and SP-style aux."""
    aux_tok = request.getfixturevalue(aux_fixture)
    table = TokenTextTable(tiny_tok_p)
    aux_table = TokenTextTable(aux_tok)
    mapper = VocabMapper(aux_tok, table, "cpu")

    n_checked = 0
    for i, b in enumerate(table.bytes_of):
        if not mapper.coverage[i]:
            continue
        ab = aux_table.bytes_for(int(mapper.map_idx[i]))
        assert ab, f"P id {i} mapped to an empty-byte aux token"
        assert b.startswith(ab), f"P id {i} bytes {b!r} not prefixed by aux bytes {ab!r}"
        n_checked += 1
    assert n_checked > 100  # the map must actually cover the bulk of the vocab


def test_mapper_specials_and_padding_uncovered(tiny_tok_p, tiny_tok_aux):
    table = TokenTextTable(tiny_tok_p)
    mapper = VocabMapper(tiny_tok_aux, table, "cpu")
    for sid in table.special_ids:
        assert not mapper.coverage[sid]
    n = len(table.bytes_of)
    mapper.ensure(n + 7)  # simulate vLLM vocab padding
    assert mapper.coverage.shape[0] == n + 7
    assert not mapper.coverage[n:].any()


def test_mapper_byte_fallback(tiny_tok_p, tiny_tok_aux):
    """P tokens carrying partial UTF-8 (e.g. pieces of an emoji) map to the
    aux token for their first raw byte."""
    table = TokenTextTable(tiny_tok_p)
    mapper = VocabMapper(tiny_tok_aux, table, "cpu")
    partial = [
        i
        for i in tiny_tok_p("🚀", add_special_tokens=False)["input_ids"]
        if _is_partial(table.bytes_for(i))
    ]
    assert partial  # the tiny vocab can't hold the emoji as one token
    aux_table = TokenTextTable(tiny_tok_aux)
    for i in partial:
        assert mapper.coverage[i]
        assert aux_table.bytes_for(int(mapper.map_idx[i])) == table.bytes_for(i)[:1]


def _is_partial(b: bytes) -> bool:
    try:
        b.decode("utf-8")
        return False
    except UnicodeDecodeError:
        return True


def test_mapper_logprob_invariance(tiny_tok_p, tiny_tok_aux):
    """Adding a per-row constant to the raw aux logits must not change the
    mapped output — the regression net for gathering log-probs, not logits."""
    table = TokenTextTable(tiny_tok_p)
    mapper = VocabMapper(tiny_tok_aux, table, "cpu")
    V = len(table.bytes_of)
    g = torch.Generator().manual_seed(0)
    l_aux = torch.randn(3, aux_width(tiny_tok_aux), generator=g)
    shift = torch.tensor([[3.7], [-12.0], [100.0]])
    a = mapper.map_logits(l_aux, V)
    b = mapper.map_logits(l_aux + shift, V)
    torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-4)
    # uncovered entries are exactly zero
    assert (a[:, ~mapper.coverage[:V]] == 0.0).all()
    # covered entries are genuine log-probs
    assert (a[:, mapper.coverage[:V]] < 0.0).all()


def test_mapper_gather_in_range(tiny_tok_p, tiny_tok_aux):
    table = TokenTextTable(tiny_tok_p)
    mapper = VocabMapper(tiny_tok_aux, table, "cpu")
    assert int(mapper.map_idx.max()) < aux_width(tiny_tok_aux)
    assert int(mapper.map_idx.min()) >= 0


def test_mapper_disk_cache(tiny_tok_p, tiny_tok_aux, tmp_path):
    table = TokenTextTable(tiny_tok_p)
    m1 = VocabMapper(tiny_tok_aux, table, "cpu", cache_dir=tmp_path)
    files = list(tmp_path.glob("vocab_map_*.pt"))
    assert len(files) == 1
    m2 = VocabMapper(tiny_tok_aux, table, "cpu", cache_dir=tmp_path)
    assert torch.equal(m1.map_idx, m2.map_idx)
    assert torch.equal(m1.coverage, m2.coverage)
