"""Tests for the data preprocessing workflow"""

import pandas as pd

from xtream_service.ml_pipeline.const import (
    NUMERIC,
    CATEGORICAL,
    XGB_TO_CATG,
)
from xtream_service.ml_pipeline.data_processing import (
    extract_from_csv,
    dummy_encode,
    filter_numeric,
    to_categorical_dtype,
    train_test_data_get,
)

DF: pd.DataFrame = extract_from_csv("assets/data/raw/diamonds/diamonds.csv")


def test_filter_numeric() -> None:
    """Test the numerical columns filter on the locally available dataset."""
    assert DF.shape[0] != filter_numeric(DF, NUMERIC, 0).shape[0]
    assert DF.shape[0] > filter_numeric(DF, NUMERIC, -1).shape[0]
    assert DF.shape[0] == filter_numeric(DF, NUMERIC, -2).shape[0]
    assert DF.shape[0] != filter_numeric(DF, NUMERIC, 25).shape[0]
    assert DF.shape[0] > filter_numeric(DF, NUMERIC, 1800).shape[0]


def test_dummy_encode() -> None:
    """Test one-hot encoding on the locally available dataset."""
    assert DF.shape[0] == dummy_encode(DF, CATEGORICAL).shape[0]
    assert DF.shape[1] < dummy_encode(DF, CATEGORICAL).shape[1]


def test_to_categorical_dtype() -> None:
    """Test data casting on locally available dataset."""
    df: pd.DataFrame = DF.copy()
    df = to_categorical_dtype(df, targets=XGB_TO_CATG, ordered=True)

    assert DF is not df
    assert DF["cut"].dtype != df["cut"].dtype


def test_train_test_data_get() -> None:
    """Test data split on the locally available dataset."""
    assert isinstance(train_test_data_get(DF, target=NUMERIC[0]), list)
    assert len(train_test_data_get(DF, target=NUMERIC[1], test_size=0.1)) == 4

    x_train, x_test, y_train, y_test = train_test_data_get(
        DF, target=NUMERIC[2], seed=24
    )

    assert x_train.shape[0] == y_train.shape[0] and x_test.shape[0] == y_test.shape[0]
    assert x_train.shape[1] == x_test.shape[1] and y_train.shape[0] > y_test.shape[0]
