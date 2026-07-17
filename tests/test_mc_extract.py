"""Unit tests for the robust multiple-choice answer extractor (evals/scoring/mc_extract).

Covers the real failure modes that pushed the point-in-time baselines below random with the
harness's narrow regex: markdown/LaTeX-wrapped answers, restated answers (take the last),
lone-letter sign-offs, and out-of-range letters that must NOT be mistaken for a choice.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "evals", "scoring"))
from mc_extract import extract_choice, gold_letter  # noqa: E402


@pytest.mark.parametrize("text,expected", [
    # plain harness-style phrasing
    ("... therefore the answer is (D).", "D"),
    ("The answer is D", "D"),
    ("Answer: C", "C"),
    # markdown / LaTeX wrappers that broke the strict `answer is \(?([A-J])\)?` regex
    ("The correct answer is **(D)**", "D"),
    ("So the answer is **E**.", "E"),
    (r"Hence \boxed{B}.", "B"),
    (r"Thus the final answer is $\text{F}$.", "F"),
    # restated answer -> take the LAST commitment
    ("First I thought the answer is A, but the answer is C.", "C"),
    # lone-letter sign-off on its own line
    ("Reasoning about the problem...\n\n**D**", "D"),
    ("...long chain of thought.\nH.", "H"),
])
def test_extracts_committed_answer_mmlu_pro(text, expected):
    assert extract_choice(text, 10) == expected


def test_out_of_range_letter_not_matched():
    # 'Z' is beyond the A-D range for a 4-option task and must not be returned; the real
    # committed answer 'B' should win.
    assert extract_choice("Option Z is a trap. The answer is B.", 4) == "B"


def test_reasoning_letters_do_not_beat_explicit_answer():
    # 'A protein' / 'B cells' litter the reasoning, but the explicit final answer is C.
    txt = "A protein binds first. B cells respond later. Therefore the answer is C."
    assert extract_choice(txt, 4) == "C"


def test_bare_first_letter_last_resort():
    # No explicit commitment: fall back to the first bare in-range capital (old Redux rule).
    assert extract_choice("D", 4) == "D"
    assert extract_choice("Hmm, maybe B then.", 4) == "B"
    # ...but the last-resort tier can be disabled.
    assert extract_choice("Hmm, maybe B then.", 4, allow_bare=False) is None


def test_empty_and_degenerate_return_none():
    assert extract_choice("", 10) is None
    assert extract_choice("\n", 10) is None
    assert extract_choice("### Response:### Response:### Response:", 10) is None


def test_gold_letter_normalization():
    assert gold_letter("B") == "B"
    assert gold_letter("(D)") == "D"
    assert gold_letter(" c ") == "C"
    assert gold_letter(None) is None
