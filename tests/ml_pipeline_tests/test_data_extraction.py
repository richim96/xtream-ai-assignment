"""Tests for the data extraction workflow"""

import pytest
import polars as pl

from diamonds.ml_pipeline.data_extraction import extract_from_csv
from _const import CSV_SOURCE


def test_extract_from_csv() -> None:
    """Test csv data extraction on the locally available dataset."""
    df: pl.DataFrame = extract_from_csv(CSV_SOURCE)

    # The dataframe has rows
    assert df.shape[0] > 0
    # The datafram has columns
    assert df.shape[1] > 0
