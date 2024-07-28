"""Tests for the data preprocessing workflow"""

import pandas as pd

from const import CSV_SOURCE, COLS_NUMERIC, COLS_CATEGORICAL, XGB_TARGETS

from xtream_service.ml_pipeline.data_processing import (
    extract_from_csv,
    dummy_encode,
    filter_numeric,
    to_categorical_dtype,
    train_test_data_get,
)


def test_extract_from_csv() -> None:
    """Test csv data extraction on the locally available dataset."""
    df: pd.DataFrame = extract_from_csv(CSV_SOURCE)

    assert df.shape[0] >= 0

    if len(df.shape) == 2:
        assert df.shape[1] >= 1


DF: pd.DataFrame = extract_from_csv(CSV_SOURCE)


def test_filter_numeric() -> None:
    """Test the numerical columns filter on the locally available dataset."""
    assert DF.shape[0] != filter_numeric(DF, COLS_NUMERIC, 0).shape[0]
    assert DF.shape[0] > filter_numeric(DF, COLS_NUMERIC, -1).shape[0]
    assert DF.shape[0] == filter_numeric(DF, COLS_NUMERIC, -2).shape[0]
    assert DF.shape[0] != filter_numeric(DF, COLS_NUMERIC, 25).shape[0]
    assert DF.shape[0] > filter_numeric(DF, COLS_NUMERIC, 1800).shape[0]


def test_dummy_encode() -> None:
    """Test one-hot encoding on the locally available dataset."""
    assert DF.shape[0] == dummy_encode(DF, COLS_CATEGORICAL).shape[0]
    assert DF.shape[1] < dummy_encode(DF, COLS_CATEGORICAL).shape[1]


def test_to_categorical_dtype() -> None:
    """Test data casting on locally available dataset."""
    df: pd.DataFrame = DF.copy()
    df = to_categorical_dtype(df, targets=XGB_TARGETS, ordered=True)

    assert DF is not df
    assert DF["cut"].dtype != df["cut"].dtype


def test_train_test_data_get() -> None:
    """Test data split on the locally available dataset."""
    assert isinstance(train_test_data_get(DF, target=COLS_NUMERIC[0]), list)
    assert len(train_test_data_get(DF, target=COLS_NUMERIC[1], test_size=0.1)) == 4

    x_train, x_test, y_train, y_test = train_test_data_get(
        DF, target=COLS_NUMERIC[2], seed=24
    )

    assert x_train.shape[0] == y_train.shape[0] and x_test.shape[0] == y_test.shape[0]
    assert x_train.shape[1] == x_test.shape[1] and y_train.shape[0] > y_test.shape[0]
