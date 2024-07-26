"""Tests for the model training workflow"""

import pandas as pd

from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from _const import CSV_SOURCE, COLS_CATEGORICAL, COLS_TO_DROP, COLS_NUMERIC, XGB_TARGETS

from xtream_service.ml_pipeline import data_extraction, preprocessing, training
from xtream_service.ml_pipeline.optimization import StdOptimizer

_DF: pd.DataFrame = data_extraction.extract_from_csv(CSV_SOURCE)
_DF = preprocessing.filter_numeric(_DF, COLS_NUMERIC, 0)


def test_linear_regression_train() -> None:
    """Test the succesful creation of a linear model."""
    df: pd.DataFrame = _DF.drop(columns=COLS_TO_DROP)

    assert df is not _DF

    df = preprocessing.dummy_encode(df, COLS_CATEGORICAL)
    x_train, _, y_train, _ = preprocessing.train_test_data_get(df, target="price")

    training.linear_regression_train(x_train, y_train, log_transform=True)
    assert isinstance(
        training.linear_regression_train(x_train, y_train), LinearRegression
    )


def test_xgb_regressor_train() -> None:
    """Test the successful creation of a gradient boosting model."""
    df: pd.DataFrame = _DF.copy()

    assert df is not _DF

    df = preprocessing.to_categorical_dtype(df, targets=XGB_TARGETS)
    x_train, _, y_train, _ = preprocessing.train_test_data_get(df, target="price")
    optimizer: StdOptimizer = StdOptimizer(x_train, y_train)
    optimizer.opt_n_trials = 2  # trials take a long time to run for hook testing

    training.xgb_regressor_train(x_train, y_train, optimizer=optimizer)
    assert isinstance(training.xgb_regressor_train(x_train, y_train), XGBRegressor)
