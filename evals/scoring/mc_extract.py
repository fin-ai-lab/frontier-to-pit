"""Robust multiple-choice answer extraction for the utility benchmarks.

The lm-eval-harness ships one narrow regex per task:

  * MMLU-Pro   ``custom-extract``:   ``answer is \\(?([A-J])\\)?``  — only matches the
    exact closing phrase, so a model that writes ``the answer is **(D)**`` (markdown
    bold), ``\\boxed{D}``, ``Answer: D`` or just ends on a lone ``D`` scores ``[invalid]``
    and is graded WRONG — dropping it *below* the random-guess floor.
  * MMLU-Redux ``default``:          ``([ABCD])`` take-first — grabs the FIRST capital
    A–D anywhere, which for a rambling model is usually a stray letter in the reasoning
    (noise), not the committed answer.
  * GPQA       ``flexible-extract``: last ``(\\([A-Z]\\))`` — already fairly robust.

`extract_choice` replaces all three with a single priority cascade that reads the
model's *committed* answer while staying conservative enough not to invent one:

  Tier 1  explicit final-answer phrases      "the answer is (D)", "answer: D",
          (take the LAST — models restate)   "final answer = D", "correct option is D"
  Tier 2  \\boxed{D}                          LaTeX box (take LAST)
  Tier 3  a line that is ONLY the letter      "**D**", "(D).", "D" on its own line (LAST)
  Tier 4  a parenthesized letter (D)          gpqa-style (take LAST)
  Tier 5  "option D" / "choice D"             weak commit (take LAST)
  Tier 6  first bare capital letter in range  the old Redux behaviour — noisy, so it is
          (take FIRST)                        LAST-RESORT and only fires when nothing
                                              above matched; disabled with allow_bare=False

Markdown (``**``, ``__``, backticks), LaTeX (``$``, ``\\(``, ``\\)``, ``\\text{}``) and
surrounding punctuation are tolerated everywhere. Matching is case-insensitive; the
returned letter is upper-cased. Only letters within the task's option range are
considered (``A-J`` for MMLU-Pro, ``A-D`` for Redux/GPQA), so an out-of-range capital
never masquerades as an answer.
"""

from __future__ import annotations

import re
import string

# Wrapper cruft allowed to sit immediately before the answer letter: markdown emphasis,
# LaTeX inline-math / \( \) / \text{ / \boxed{, an opening paren/brace/bracket, quotes.
_PRE = r"(?:\*|_|`|\$|\\\(|\\text\{|\\mathrm\{|\\boxed\{|\(|\[|\{|\"|'|\s)*"


def _letters(n_options: int) -> str:
    if not 1 <= n_options <= 26:
        raise ValueError(f"n_options out of range: {n_options}")
    return string.ascii_uppercase[:n_options]


def _last(pat: str, text: str):
    ms = list(re.finditer(pat, text, re.IGNORECASE | re.MULTILINE))
    return ms[-1].group(1).upper() if ms else None


def _first(pat: str, text: str):
    m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
    return m.group(1).upper() if m else None


def extract_choice(text: str, n_options: int, allow_bare: bool = True):
    """Return the committed choice letter (upper-case) or None if none is discernible.

    `n_options` fixes the valid letter range (10 -> A-J, 4 -> A-D). `allow_bare=False`
    disables the noisy last-resort "first capital letter" tier (Tier 6).
    """
    if not text:
        return None
    L = _letters(n_options)
    cls = f"[{L}]"

    # Tier 1 — explicit final-answer declarations. The trigger phrase, an optional
    # connector (is / : / = / -), optional wrapper cruft, then the letter with a
    # trailing boundary so "A" in "Answer: A protein" would need the boundary — the
    # letter must not be immediately followed by another letter.
    trigger = (r"(?:final\s+answer|the\s+answer\s+is|answer\s+is|answers?\s*[:=]"
               r"|correct\s+answer\s+is|correct\s+option\s+is|correct\s+choice\s+is"
               r"|correct\s+answer\s*[:=]|the\s+correct\s+answer\s+is)")
    t1 = rf"{trigger}\s*(?:is|=|:|-|→|->)?\s*{_PRE}({cls})(?![A-Za-z])"
    if (r := _last(t1, text)) is not None:
        return r

    # Tier 2 — \boxed{ D } (also \boxed{\text{D}})
    t2 = rf"\\boxed\{{\s*{_PRE}({cls})(?![A-Za-z])"
    if (r := _last(t2, text)) is not None:
        return r

    # Tier 3 — a line whose only content is the letter (± markdown / paren / trailing
    # punctuation). This is the classic "…\n\nD" or "**D**." sign-off.
    t3 = rf"^{_PRE}({cls})\s*(?:\)|\}}|\]|\*|_|`|\.|:)*\s*$"
    if (r := _last(t3, text)) is not None:
        return r

    # Tier 4 — a parenthesized letter (D). gpqa flexible-extract style; take the last.
    t4 = rf"\(\s*({cls})\s*\)"
    if (r := _last(t4, text)) is not None:
        return r

    # Tier 5 — "option D" / "choice D" / "answer D" weak commitment.
    t5 = rf"(?:option|choice|answer)\s+{_PRE}({cls})(?![A-Za-z])"
    if (r := _last(t5, text)) is not None:
        return r

    # Tier 6 — LAST RESORT: the first bare capital letter in range (old Redux default).
    # Noisy; only reached when the model never committed in any recognizable form.
    if allow_bare:
        t6 = rf"(?<![A-Za-z])({cls})(?![A-Za-z])"
        if (r := _first(t6, text)) is not None:
            return r
    return None


def gold_letter(target: str):
    """Normalize a stored gold `target` ("B", "(D)", " d ") to a bare upper letter."""
    if target is None:
        return None
    m = re.search(r"[A-Za-z]", str(target))
    return m.group(0).upper() if m else None
