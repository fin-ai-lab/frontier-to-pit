"""The serving-side prompts must byte-match the shipped datasets.

ma/pharma/covid carry a PER-DOC benchmark system prompt (helpful-assistant +
forecasting prompt + runtime temporal context with the date the question is
posed) built by ftp.prompts.benchmark_system_prompt; the date column feeding it
differs per task (ma/pharma: prompt_date, covid: date).
"""
import pathlib

import pytest

from ftp.prompts import FORECAST_SYSTEM_PROMPT, benchmark_system_prompt

_TEMPORAL = pathlib.Path(__file__).resolve().parents[1] / "evals" / "lmeval" / "tasks" / "temporal"


@pytest.mark.parametrize(("task", "date_col"),
                         [("ma", "prompt_date"), ("pharma", "prompt_date"), ("covid", "date")])
def test_benchmark_system_prompt_matches_parquet(task, date_col):
    pd = pytest.importorskip("pandas")
    df = pd.read_parquet(_TEMPORAL / f"{task}.parquet", columns=["system_prompt", date_col])
    for sp, d in zip(df["system_prompt"], df[date_col], strict=True):
        assert sp == benchmark_system_prompt(pd.Timestamp(d).strftime("%B %-d, %Y"))
        assert FORECAST_SYSTEM_PROMPT in sp
