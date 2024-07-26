"""Tests for the model training workflow"""

import pandas as pd

from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from _const import CSV_SOURCE, COLS_CATEGORICAL, COLS_TO_DROP, COLS_NUMERIC, XGB_TARGETS

from xtream.ml_pipeline.data_extraction import extract_from_csv
from xtream.ml_pipeline.preprocessing import (
    dummy_encode,
    filter_numeric,
    to_categorical_dtype,
    train_test_data_get,
)
from xtream.ml_pipeline.optimization import StdOptimizer
from xtream.ml_pipeline.training import linear_regression_train, xgb_regressor_train

_DF: pd.DataFrame = extract_from_csv(CSV_SOURCE)
_DF = filter_numeric(_DF, COLS_NUMERIC, 0)


def test_linear_regression_train() -> None:
    """Test the succesful creation of a linear model."""
    df: pd.DataFrame = _DF.drop(columns=COLS_TO_DROP)

    assert df is not _DF

    df = dummy_encode(df, COLS_CATEGORICAL)
    x_train, _, y_train, _ = train_test_data_get(df, target="price")

    linear_regression_train(x_train, y_train, log_transform=True)
    assert isinstance(linear_regression_train(x_train, y_train), LinearRegression)


def test_xgb_regressor_train() -> None:
    """Test the successful creation of a gradient boosting model."""
    df: pd.DataFrame = _DF.copy()

    assert df is not _DF

    df = to_categorical_dtype(df, targets=XGB_TARGETS)
    x_train, _, y_train, _ = train_test_data_get(df, target="price")
    optimizer: StdOptimizer = StdOptimizer(x_train, y_train)
    optimizer.opt_n_trials = 2  # trials take a long time to run for hook testing

    xgb_regressor_train(x_train, y_train, optimizer=optimizer)
    assert isinstance(xgb_regressor_train(x_train, y_train), XGBRegressor)
