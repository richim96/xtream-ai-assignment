"""Module for the model training workflow"""

import optuna
import numpy as np
import pandas as pd

from optuna.study import Study
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from xtream_service.ml_pipeline import LOGGER
from xtream_service.ml_pipeline.optimization import StdOptimizer


def linear_regression_train(
    x_train: pd.Series, y_train: pd.Series, log_transform: bool = False
) -> LinearRegression:
    """Train a simple linear regression model.

    Parameters
    ----------
    x_train : pd.Series
        Training series for the predictors.
    y_train : pd.Seris
        Training series for the target variable.
    log_transform : bool, default=False
        If `True`, train the model applying a log tranformation on the target.

    Return
    ----------
    LinearRegression
        The trained linear regression model.
    """
    lr: LinearRegression = LinearRegression()
    if log_transform:
        y_train = np.log(y_train)

    lr.fit(x_train, y_train)
    LOGGER.info(
        "Linear model with log transformation trained."
        if log_transform
        else "Linear model trained."
    )
    return lr


def xgb_regressor_train(
    x_train: pd.Series,
    y_train: pd.Series,
    categorical: bool = True,
    seed: int = 42,
    optimizer: StdOptimizer | None = None,
) -> XGBRegressor:
    """Train a gradient boosting model. Hyperparameter optimization available.

    Parameters
    ----------
    x_train : pd.Series
        Training series for the predictors.
    y_train : pd.Seris
        Training series for the target variable.
    categorical : bool, default=True
        If `True`, it instructs to the regressor that the data is also categorical.
    seed : int, default=42
        Random-state seed allowing for reproducible outputs across multiple calls.
    optimizer : StdOptimizer, default=None
        An 'Optimizer()' instance to fine tune the model's hyperparameters.

    Return
    ----------
    XGBRegressor
        The trained gradient boosting model.
    """
    xgb: XGBRegressor
    study: Study

    if optimizer is not None:
        LOGGER.info("Fine-tuning XGBRegressor's hyperparameters...")
        study = optuna.create_study(
            direction=optimizer.study_direction, study_name="XGB Hyperparameter Tuning"
        )
        study.optimize(
            func=optimizer.std_objective_fn,
            n_trials=optimizer.opt_n_trials,
            show_progress_bar=True,
        )
        xgb = XGBRegressor(
            **study.best_params, enable_categorical=categorical, random_state=seed
        )
    else:
        xgb = XGBRegressor(enable_categorical=categorical, random_state=seed)

    xgb.fit(x_train, y_train)
    LOGGER.info("Gradient boosting model trained.")
    return xgb
