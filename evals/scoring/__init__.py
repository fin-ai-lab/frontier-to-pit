"""Ported scoring utilities for the evals.

* ``firm_match`` — ported from forecasting-thoughts ``src/fcst_thoughts/ma_match.py``
  (firm-name normalization + look-ahead mention check for the M&A leakage task).

Instruction-following scoring now comes from lm-eval's ``ifeval`` task (no longer
vendored here).
"""
