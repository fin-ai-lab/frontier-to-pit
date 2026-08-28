"""CPU tests for the live degeneration guard (ftp.guard) — no vLLM, no GPU.

The engine half (GuardLogitsProcessor) needs vLLM and a judge model, so it is
covered by GPU smoke runs; everything here exercises the vLLM-free pieces: the
config round-trip, the judge window, marker resolution, and — most importantly —
the client rollback loop's policy (rewind, escalation, visible failure, budget
and round backstops) against a scripted fake engine.
"""

import pytest

from ftp.guard import (
    FAIL_TEXT,
    GuardConfig,
    resolve_marker_id,
    rollback_generate,
    window_ids,
)

MARKER = 999


# ─────────────────────────────── config ───────────────────────────────


def test_config_env_round_trip():
    cfg = GuardConfig(model="x/y", device="cuda:1", interval=10, backtrack=20,
                      threshold=0.5, marker="<|m|>", dtype="float32", tries=4,
                      max_rounds=7, min_tokens=9)
    assert GuardConfig.from_env(cfg.to_env()) == cfg


def test_config_env_round_trip_null_device():
    cfg = GuardConfig(device=None)
    assert GuardConfig.from_env(cfg.to_env()).device is None


def test_config_rejects_backtrack_smaller_than_interval():
    # A rewind must discard at least everything generated since the last clean
    # check, or the surviving prefix can end in judged-bad tokens.
    with pytest.raises(ValueError, match="backtrack"):
        GuardConfig(interval=25, backtrack=10)


@pytest.mark.parametrize("kw", [{"threshold": 0.0}, {"threshold": 1.5},
                                {"tries": 0}, {"max_rounds": 0},
                                {"interval": 0}, {"min_tokens": 0}])
def test_config_rejects_bad_values(kw):
    with pytest.raises(ValueError):
        GuardConfig(**kw)


# ─────────────────────────────── window ───────────────────────────────


def test_window_uses_output_tail():
    assert window_ids([1, 2, 3], list(range(100, 200)), 10) == list(range(190, 200))


def test_window_crosses_into_prompt_tail():
    # 3 output tokens, backtrack 10 -> the last 7 prompt tokens fill the window.
    assert window_ids([1, 2, 3, 4, 5, 6, 7, 8, 9], [50, 51, 52], 10) == \
        [3, 4, 5, 6, 7, 8, 9, 50, 51, 52]


def test_window_short_prompt_and_output():
    assert window_ids([1], [2], 10) == [1, 2]


def test_window_no_prompt():
    assert window_ids(None, [1, 2, 3], 2) == [2, 3]


# ─────────────────────────── marker resolution ───────────────────────────


class _Tok:
    def __init__(self, vocab, unk=None):
        self.vocab = vocab
        self.unk_token_id = unk

    def convert_tokens_to_ids(self, t):
        return self.vocab.get(t, self.unk_token_id)


def test_marker_resolves_configured_token():
    assert resolve_marker_id(_Tok({"<|m|>": 7}), "<|m|>") == 7


def test_marker_falls_back_through_candidates():
    # configured marker absent -> first reserved candidate that resolves wins
    assert resolve_marker_id(_Tok({"<|fim_prefix|>": 11}, unk=0), "<|m|>") == 11


def test_marker_raises_when_nothing_resolves():
    with pytest.raises(ValueError, match="DD_GUARD_MARKER"):
        resolve_marker_id(_Tok({}, unk=0), "<|m|>")


# ─────────────────────────── rollback loop ───────────────────────────


class _SP:
    """Duck-typed SamplingParams: only what rollback_generate touches."""

    def __init__(self, max_tokens=100, stop_token_ids=None, extra_args=None):
        self.max_tokens = max_tokens
        self.stop_token_ids = stop_token_ids
        self.extra_args = extra_args


class _Comp:
    def __init__(self, token_ids, stop_reason=None):
        self.token_ids = list(token_ids)
        self.stop_reason = stop_reason
        self.text = ""


class _Out:
    def __init__(self, token_ids, stop_reason=None):
        self.outputs = [_Comp(token_ids, stop_reason)]


class _FakeLLM:
    """Scripted engine: pops one round of (ids, stop_reason) per generate call,
    matched positionally to the still-pending requests."""

    def __init__(self, rounds):
        self.rounds = [list(r) for r in rounds]
        self.calls = []  # (prompt_token_ids per request, sampling params per request)

    def generate(self, prompts, sps, use_tqdm=False):
        self.calls.append(([list(p["prompt_token_ids"]) for p in prompts], list(sps)))
        spec = self.rounds.pop(0)
        assert len(spec) == len(prompts), "script/pending mismatch"
        return [_Out(ids, sr) for ids, sr in spec]


def _decode(ids):
    return ",".join(map(str, ids))


def _run(llm, prompts, sps, **kw):
    kw.setdefault("marker_id", MARKER)
    kw.setdefault("backtrack", 5)
    kw.setdefault("tries", 2)
    kw.setdefault("max_rounds", 12)
    kw.setdefault("decode", _decode)
    return rollback_generate(llm, prompts, sps, **kw)


def test_clean_passthrough():
    llm = _FakeLLM([[(list(range(10)), None)]])
    outs, stats = _run(llm, [[1, 2, 3]], _SP(max_tokens=50))
    assert outs[0].outputs[0].token_ids == list(range(10))
    assert outs[0].outputs[0].text == _decode(list(range(10)))
    assert stats == {"rollbacks": 0, "escalations": 0, "failed": 0}


def test_marker_registered_and_caller_sp_untouched():
    caller_sp = _SP(max_tokens=50, stop_token_ids=[7])
    llm = _FakeLLM([[([1], None)]])
    _run(llm, [[9]], caller_sp)
    used = llm.calls[0][1][0]
    assert MARKER in used.stop_token_ids and 7 in used.stop_token_ids
    assert caller_sp.stop_token_ids == [7]  # caller's object never mutated


def test_single_rollback_splices_and_rebudgets():
    # round 1: 20 tokens then trip -> keep 15 (backtrack 5); round 2: clean 3.
    r1 = list(range(100, 120)) + [MARKER]
    llm = _FakeLLM([[(r1, MARKER)], [([7, 8, 9], None)]])
    outs, stats = _run(llm, [[1]], _SP(max_tokens=50))
    merged = list(range(100, 115)) + [7, 8, 9]
    assert outs[0].outputs[0].token_ids == merged
    assert outs[0].outputs[0].text == _decode(merged)
    assert stats["rollbacks"] == 1 and stats["failed"] == 0
    # round-2 resubmission: prompt grew by the accepted prefix, budget shrank by it
    prompts2, sps2 = llm.calls[1]
    assert prompts2[0] == [1] + list(range(100, 115))
    assert sps2[0].max_tokens == 50 - 15


def test_no_progress_trips_escalate_into_accepted_text():
    # Round 1 accepts 15 tokens. Rounds 2 and 3 trip with tiny outputs (rewind eats
    # them whole -> no progress). tries=2 -> escalation walks 5 tokens back into the
    # accepted prefix. Round 4 finishes clean.
    llm = _FakeLLM([
        [(list(range(100, 120)) + [MARKER], MARKER)],  # accept 100..114
        [([55, MARKER], MARKER)],                      # no progress (1 of 2)
        [([66, MARKER], MARKER)],                      # no progress (2 of 2) -> escalate
        [([7], None)],
    ])
    outs, stats = _run(llm, [[1]], _SP(max_tokens=50))
    # escalation dropped 110..114 from the accepted prefix
    assert outs[0].outputs[0].token_ids == list(range(100, 110)) + [7]
    assert stats["escalations"] == 1 and stats["rollbacks"] == 3
    assert llm.calls[3][0][0] == [1] + list(range(100, 110))


def test_fails_visibly_when_breaking_at_the_start():
    # Nothing ever accepted; two no-progress trips (tries=2) with an empty prefix
    # -> the visible failure string, not garbage.
    llm = _FakeLLM([
        [([55, 56, MARKER], MARKER)],
        [([57, MARKER], MARKER)],
    ])
    outs, stats = _run(llm, [[1]], _SP(max_tokens=50))
    assert outs[0].outputs[0].text == FAIL_TEXT
    assert outs[0].outputs[0].token_ids == []
    assert stats["failed"] == 1


def test_max_rounds_returns_what_we_have():
    # Every round trips but makes progress (6 clean tokens kept per round), so
    # escalation never fires; max_rounds=3 stops the rescue and returns the
    # accepted prefix.
    rounds = [[(list(range(i * 10, i * 10 + 11)) + [MARKER], MARKER)] for i in range(3)]
    llm = _FakeLLM(rounds)
    outs, stats = _run(llm, [[1]], _SP(max_tokens=500), max_rounds=3)
    assert stats["rollbacks"] == 3
    kept = []
    for i in range(3):
        kept += list(range(i * 10, i * 10 + 6))  # 11 emitted - 5 backtrack
    assert outs[0].outputs[0].token_ids == kept


def test_budget_exhausted_mid_rescue_returns_prefix():
    # max_tokens=16: round 1 keeps 15 -> remaining budget 1; round 2 trips with no
    # progress -> remaining budget still 1 >= 1, round 3 would run, but make round 2
    # eat the budget instead: keep 15 + trip keeps 1 more -> budget 0 -> stop.
    llm = _FakeLLM([
        [(list(range(100, 120)) + [MARKER], MARKER)],   # keep 15, budget left 1
        [([55, 56, 57, 58, 59, 60, MARKER], MARKER)],   # keep 1 (6 - 5), budget 0
    ])
    outs, stats = _run(llm, [[1]], _SP(max_tokens=16))
    assert outs[0].outputs[0].token_ids == list(range(100, 115)) + [55]
    assert stats["rollbacks"] == 2 and stats["failed"] == 0


def test_batch_mixed_outcomes_keep_input_order():
    # request 0 clean; request 1 trips once then finishes.
    llm = _FakeLLM([
        [([11, 12], None), (list(range(200, 210)) + [MARKER], MARKER)],
        [([31], None)],  # only request 1 pending
    ])
    outs, stats = _run(llm, [[1], [2]], [_SP(max_tokens=50), _SP(max_tokens=50)])
    assert outs[0].outputs[0].token_ids == [11, 12]
    assert outs[1].outputs[0].token_ids == list(range(200, 205)) + [31]
    assert stats["rollbacks"] == 1
