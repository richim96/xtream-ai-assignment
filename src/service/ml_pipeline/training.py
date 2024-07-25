"""Submodule for the model training workflow"""

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression


def linear_model_train(
    data_train: pd.Series, target_train: pd.Series, log_transform: bool = False
) -> LinearRegression:
    """Train a simple linear regression model.

    Parameters
    ----------
    data_train : pd.Series
        Training series for the predictors.
    target_train : pd.Seris
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
        y_train_log: pd.Series = np.log(target_train)
        lr.fit(data_train, y_train_log)
    else:
        lr.fit(data_train, target_train)

    return lr


def xgboost_train() -> None:
    pass
