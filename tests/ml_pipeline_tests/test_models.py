"""Tests for the model creation workflow"""

import pandas as pd

from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from const import CSV_SOURCE, COLS_CATEGORICAL, COLS_TO_DROP, COLS_NUMERIC, XGB_TARGETS

from xtream_service.utils.utils import uuid_get
from xtream_service.ml_pipeline import data_processing, models
from xtream_service.ml_pipeline.optimization import StdOptimizer

_DF: pd.DataFrame = data_processing.extract_from_csv(CSV_SOURCE)
_DF = data_processing.filter_numeric(_DF, COLS_NUMERIC, 0)


def test_linear_regression_model() -> None:
    """Test the succesful creation of a linear model."""
    df: pd.DataFrame = _DF.drop(columns=COLS_TO_DROP)
    df = data_processing.dummy_encode(df, COLS_CATEGORICAL)
    x_train, x_test, y_train, y_test = data_processing.train_test_data_get(
        df, target="price"
    )

    lr: models.LinearRegressionModel = models.LinearRegressionModel(
        model=LinearRegression(), data_uid=uuid_get()
    )
    lr.train(x_train, y_train, log_transform=True)
    lr.evaluate(x_test, y_test)

    assert df is not _DF
    assert isinstance(lr.model, LinearRegression)
    assert "mean_absolute_error" in lr.info()["metrics"]


def test_xgb_regressor_model() -> None:
    """Test the successful creation of a gradient boosting model."""
    df: pd.DataFrame = _DF.copy()
    df = data_processing.to_categorical_dtype(df, targets=XGB_TARGETS)
    x_train, x_test, y_train, y_test = data_processing.train_test_data_get(
        df, target="price"
    )

    xgb: models.XgbRegressorModel = models.XgbRegressorModel(
        model=XGBRegressor(enable_categorical=True, random_state=42),
        data_uid=uuid_get(),
    )
    xgb.optimizer = StdOptimizer(x_train, y_train)
    xgb.optimizer.opt_n_trials = 2
    xgb.train(x_train, y_train)
    xgb.evaluate(x_test, y_test)

    assert df is not _DF
    assert isinstance(xgb.model, XGBRegressor)
    assert "r2_score" in xgb.info()["metrics"]
