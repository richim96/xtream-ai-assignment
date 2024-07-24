"""Tests for the data preprocessing workflow"""

import pytest
import pandas as pd

from diamonds.ml_pipeline.data_extraction import extract_from_csv
from diamonds.ml_pipeline.data_preprocessing import (
    filter_numerical,
    dummy_encode,
    cols_drop,
)
from _const import CSV_SOURCE, NUMERICAL_COLS, CATEGORICAL_COLS


def test_filter_numerical() -> None:
    """Test the numerical columns filter on the locally available dataset."""
    df: pd.DataFrame = extract_from_csv(CSV_SOURCE)

    assert df.shape[0] != filter_numerical(df, NUMERICAL_COLS, 0).shape[0]
    assert df.shape[0] > filter_numerical(df, NUMERICAL_COLS, -1).shape[0]
    assert df.shape[0] == filter_numerical(df, NUMERICAL_COLS, -2).shape[0]
    assert df.shape[0] != filter_numerical(df, NUMERICAL_COLS, 25).shape[0]
    assert df.shape[0] > filter_numerical(df, NUMERICAL_COLS, 1800).shape[0]


def test_dummy_encode() -> None:
    """Test one-hot encoding function on the locally available dataset."""
    df: pd.DataFrame = extract_from_csv(CSV_SOURCE)

    assert df.shape[0] == dummy_encode(df, CATEGORICAL_COLS).shape[0]
    assert df.shape[1] < dummy_encode(df, CATEGORICAL_COLS).shape[1]


def test_cols_drop() -> None:
    """Test column dropping on the locally available dataset."""
    df: pd.DataFrame = extract_from_csv(CSV_SOURCE)

    assert df.shape[0] == cols_drop(df, CATEGORICAL_COLS + NUMERICAL_COLS).shape[0]
    assert df.shape[1] > cols_drop(df, NUMERICAL_COLS).shape[1]
    assert CATEGORICAL_COLS[0] not in cols_drop(df, CATEGORICAL_COLS).columns
