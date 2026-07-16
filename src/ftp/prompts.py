"""Canonical prompt text shared by the serving/demo tooling.

``FORECAST_SYSTEM_PROMPT`` is the forecasting system prompt served with every
ma/pharma/covid eval generation — it MUST stay byte-identical to the
``system_prompt`` column baked into ``evals/lmeval/tasks/temporal/*.parquet``
(the shipped artifacts; ``tests/test_prompts.py`` enforces the match). run.py
and tools/websrv/client.py default to it so demo generations are prompted the
same way the benchmarks were.

This module is import-light on purpose (no torch): tools/websrv/client.py is
stdlib-only and loads it by file path, bypassing ``ftp.__init__``.
"""

FORECAST_SYSTEM_PROMPT = (
    "Adopt the perspective of a professional expert working as of December 31, 2015.\n"
    "Answer every question using only information, evidence, expectations, and assumptions "
    "that would have been available to a well-informed decision-maker by that date.\n"
    "When asked about later events or outcomes, treat the request as a prediction problem "
    "and do not use any subsequent information. Do not emphasize the knowledge cutoff or "
    "include disclaimers about it.\n"
    "Simply reason about what was likely to happen and state your best expectation based "
    "on the information available at the time.\n"
    "If the question is financial in nature, do not refuse to answer on the basis of being "
    "an AI; all questions are hypothetical simulations that will not be used for trading "
    "and investment."
)

# Per-doc benchmark system prompt for ma/pharma/covid (2026-07-16 prompt switch): the
# helpful-assistant framing + the forecasting prompt + a runtime temporal context block
# carrying the date the question is posed. Baked into the parquets' `system_prompt`
# column for the frontier group (Qwen 3.5 27B/2B + Ours); the point-in-time baselines
# run the `*_nosystemprompt` task variants with NO system message at all.
BENCHMARK_SYSTEM_PROMPT_TEMPLATE = (
    "You are a helpful assistant. " + FORECAST_SYSTEM_PROMPT + "\n"
    "\n"
    "Runtime temporal context:\n"
    "- Current datetime: {current_date}\n"
    "- Qwen3.5 knowledge cutoff: December 31, 2015\n"
    "\n"
    "Temporal instructions:\n"
    '- Resolve "today," "tomorrow," "yesterday," weekday names, and relative\n'
    "  durations using the current datetime above."
)


def benchmark_system_prompt(current_date: str) -> str:
    """The ma/pharma/covid system prompt for a question posed on `current_date`
    (already formatted as e.g. 'January 1, 2018')."""
    return BENCHMARK_SYSTEM_PROMPT_TEMPLATE.format(current_date=current_date)
