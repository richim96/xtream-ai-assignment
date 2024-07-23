"""Tests for the data extraction workflow"""

import pytest
import polars as pl

from diamonds.ml_pipeline.data_extraction import data_extract


SOURCE: str = "~/xtream-ai-assignment/data/raw/diamonds.csv"


def test_data_extract() -> None:
    """Tests the data ingestion workload using the locally available training
    data. The test passess upon successful data extraction.
    """
    lf: pl.LazyFrame = data_extract(SOURCE)
    df: pl.DataFrame = lf.collect()

    assert df.shape[0] > 0
    assert df.shape[1] > 0
