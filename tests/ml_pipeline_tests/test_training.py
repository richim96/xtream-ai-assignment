"""Tests for the model training workflow"""

import pandas as pd

from sklearn.linear_model import LinearRegression

from _const import CSV_SOURCE, COLS_CATEGORICAL, COLS_TO_DROP, COLS_NUMERIC

from service.ml_pipeline.data_extraction import extract_from_csv
from service.ml_pipeline.preprocessing import (
    dummy_encode,
    filter_numeric,
    train_test_data_get,
)
from service.ml_pipeline.training import linear_model_train

_DF: pd.DataFrame = extract_from_csv(CSV_SOURCE)
_DF = _DF.drop(columns=COLS_TO_DROP)
_DF = filter_numeric(_DF, COLS_NUMERIC, 0)
_DF = dummy_encode(_DF, COLS_CATEGORICAL)

_X_TRAIN, _, _Y_TRAIN, _ = train_test_data_get(_DF, target="price")


def test_linear_model_train() -> None:
    """Test the succesful creation of a linear model."""
    linear_model_train(_X_TRAIN, _Y_TRAIN)
    assert isinstance(
        linear_model_train(_X_TRAIN, _Y_TRAIN, log_transform=True), LinearRegression
    )
