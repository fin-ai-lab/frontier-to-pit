"""Firm-name normalization + look-ahead mention check for the M&A eval.

Ported verbatim (logic-wise) from forecasting-thoughts
``src/fcst_thoughts/ma_match.py``. Firms carry corporate suffixes (Inc, Corp,
Holdings, ...) and sometimes stack multiples; a model's free-form completion may
use a different suffix combination or none. We strip these tokens from both sides
before substring-matching to detect "the model mentioned the target" robustly.
"""

from __future__ import annotations

import re

# Corporate suffix tokens stripped from the end of a firm name. Matched
# case-insensitively against whitespace tokens. Applied iteratively because firms
# stack them (e.g. "Holdings Inc", "Co Inc", "Group LP").
SUFFIX_TOKENS: frozenset[str] = frozenset(
    {
        # US
        "inc",
        "incorporated",
        "corp",
        "corporation",
        "co",
        "company",
        "ltd",
        "limited",
        "llc",
        "lp",
        "llp",
        "plc",
        "holdings",
        "holding",
        "group",
        "partners",
        "trust",
        "bancorp",
        "bankshares",
        "industries",
        "international",
        "enterprises",
        "ventures",
        "financial",
        "financials",
        # Non-US (sample-driven from data/ma_data_all.csv)
        "ag",
        "sa",
        "nv",
        "asa",
        "ab",
        "oy",
        "gmbh",
        "sarl",
        "spa",
        "kk",
    }
)

# Words dropped from the front of a firm name.
PREFIX_TOKENS: frozenset[str] = frozenset({"the"})


def _tokenize(name: str) -> list[str]:
    """Lowercase, split on whitespace, strip edge punctuation per token.

    Keeps ``&`` and ``-`` inside tokens so "BB&T" and "Bristol-Myers" survive,
    but drops standalone-punctuation tokens.
    """
    s = name.lower()
    s = re.sub(r"[’']", "", s)
    s = re.sub(r"[.,;:/]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    out: list[str] = []
    for t in s.split(" "):
        t = t.strip(".,;:&")
        if not t or t in {"&", "-"}:
            continue
        out.append(t)
    return out


def normalize_firm_name(name: str) -> str:
    """Strip leading articles + trailing corporate suffixes; lowercase the rest.

    Iterative on the suffix side ("Foo Holdings Inc" -> "foo"). Returns "" if the
    name collapses to nothing after stripping.
    """
    toks = _tokenize(name or "")
    while toks and toks[0] in PREFIX_TOKENS:
        toks.pop(0)
    while toks and toks[-1] in SUFFIX_TOKENS:
        toks.pop()
    return " ".join(toks)


def mentions_firm(response: str, firm_name: str, min_chars: int = 4) -> bool:
    """True if the (normalized) firm name appears in the (normalized) response.

    ``min_chars`` guards against trivially-short normalized names matching common
    words; a name normalizing to fewer than ``min_chars`` chars is unmatchable.
    """
    if not response or not firm_name:
        return False
    norm_firm = normalize_firm_name(firm_name)
    if len(norm_firm) < min_chars:
        return False
    norm_resp = " ".join(_tokenize(response))
    return norm_firm in norm_resp
