"""The serving-side forecasting system prompt must byte-match the shipped datasets."""
import pathlib

import pytest

from ftp.prompts import FORECAST_SYSTEM_PROMPT

_TEMPORAL = pathlib.Path(__file__).resolve().parents[1] / "evals" / "lmeval" / "tasks" / "temporal"


@pytest.mark.parametrize("task", ["ma", "pharma", "covid"])
def test_forecast_system_prompt_matches_parquet(task):
    pd = pytest.importorskip("pandas")
    col = pd.read_parquet(_TEMPORAL / f"{task}.parquet", columns=["system_prompt"])
    vals = set(col["system_prompt"])
    assert vals == {FORECAST_SYSTEM_PROMPT}
