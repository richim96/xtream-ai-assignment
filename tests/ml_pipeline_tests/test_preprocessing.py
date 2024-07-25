"""Tests for the data preprocessing workflow"""

import pandas as pd

from _const import CSV_SOURCE, COLS_NUMERIC, COLS_CATEGORICAL

from service.ml_pipeline.data_extraction import extract_from_csv
from service.ml_pipeline.preprocessing import (
    dummy_encode,
    filter_numeric,
    to_categorical_dtype,
    train_test_data_get,
)

_DF: pd.DataFrame = extract_from_csv(CSV_SOURCE)


def test_filter_numeric() -> None:
    """Test the numerical columns filter on the locally available dataset."""
    assert _DF.shape[0] != filter_numeric(_DF, COLS_NUMERIC, 0).shape[0]
    assert _DF.shape[0] > filter_numeric(_DF, COLS_NUMERIC, -1).shape[0]
    assert _DF.shape[0] == filter_numeric(_DF, COLS_NUMERIC, -2).shape[0]
    assert _DF.shape[0] != filter_numeric(_DF, COLS_NUMERIC, 25).shape[0]
    assert _DF.shape[0] > filter_numeric(_DF, COLS_NUMERIC, 1800).shape[0]


def test_dummy_encode() -> None:
    """Test one-hot encoding on the locally available dataset."""
    assert _DF.shape[0] == dummy_encode(_DF, COLS_CATEGORICAL).shape[0]
    assert _DF.shape[1] < dummy_encode(_DF, COLS_CATEGORICAL).shape[1]


def test_to_categorical_dtype() -> None:
    """Test data casting on locally available dataset."""
    df: pd.DataFrame = _DF.copy()
    targets: list[tuple[str, list[str]]] = [
        ("cut", ["Fair", "Good", "Very Good", "Ideal", "Premium"]),
        ("color", ["D", "E", "F", "G", "H", "I", "J"]),
    ]
    df = to_categorical_dtype(df, targets=targets, ordered=True)

    assert _DF is not df
    assert _DF["cut"].dtype != df["cut"].dtype


def test_train_test_data_get() -> None:
    """Test data split on the locally available dataset."""
    assert isinstance(train_test_data_get(_DF, target=COLS_NUMERIC[0]), list)
    assert len(train_test_data_get(_DF, target=COLS_NUMERIC[1], test_size=0.1)) == 4

    x_train, x_test, y_train, y_test = train_test_data_get(
        _DF, target=COLS_NUMERIC[2], seed=24
    )

    assert x_train.shape[0] == y_train.shape[0] and x_test.shape[0] == y_test.shape[0]
    assert x_train.shape[1] == x_test.shape[1] and y_train.shape[0] > y_test.shape[0]
