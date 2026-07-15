"""Answer-leak detection for the StreamingQA leak@k eval.

A generation "leaks" if it contains one of the gold answers. We normalize both sides
(lowercase, drop punctuation, collapse whitespace) and test whether any gold answer's
word sequence appears as a CONTIGUOUS run of words in the generation. Contiguous-word
matching (vs raw substring) avoids spurious hits like "10" inside "2010" while still
catching multi-word answers verbatim. Verbose reference answers simply won't match a
short generation — the canonical short reference (StreamingQA eval has several) carries
the signal.

This is a heuristic; swap in an LLM judge later if recall/precision needs it.
"""

from __future__ import annotations

import re


def _words(s: str) -> list[str]:
    s = (s or "").lower()
    s = re.sub(r"[’']", "", s)          # drop apostrophes (don't -> dont)
    s = re.sub(r"[^a-z0-9]+", " ", s)        # everything else -> space
    return s.split()


def _contains_subseq(hay: list[str], needle: list[str]) -> bool:
    if not needle or len(needle) > len(hay):
        return False
    first = needle[0]
    for i in range(len(hay) - len(needle) + 1):
        if hay[i] == first and hay[i:i + len(needle)] == needle:
            return True
    return False


def answer_leak(text: str, answers, min_chars: int = 2) -> bool:
    """True if any gold answer's words appear contiguously in `text`."""
    if not text or not answers:
        return False
    hay = _words(text)
    if not hay:
        return False
    for a in answers:
        na = _words(a)
        if not na:
            continue
        if sum(len(w) for w in na) < min_chars:   # too short to be meaningful
            continue
        if _contains_subseq(hay, na):
            return True
    return False
