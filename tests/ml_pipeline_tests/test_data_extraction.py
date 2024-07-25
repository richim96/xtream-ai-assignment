"""Tests for the data extraction workflow"""

import pandas as pd

from service.ml_pipeline.data_extraction import extract_from_csv

from _const import CSV_SOURCE


def test_extract_from_csv() -> None:
    """Test csv data extraction on the locally available dataset."""
    df: pd.DataFrame = extract_from_csv(CSV_SOURCE)

    assert df.shape[0] >= 0

    if len(df.shape) == 2:
        assert df.shape[1] >= 1
