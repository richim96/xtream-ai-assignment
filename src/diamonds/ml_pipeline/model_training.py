"""Sub-module for the model training workflow"""

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression

from diamonds.ml_pipeline.data_preprocessing import cols_drop


def linear_model_train(
    X_train: pd.Series,
    y_train: pd.Series,
    log_transform: bool = False,
) -> LinearRegression:
    """Train a simple linear regression model.

    Parameters
    ----------
    X_train : pd.Series
        Training series for the predictors.
    y_train : pd.Seris
        Training series for the target variable.
    log_transform : bool
        If true, retrain the model with a log tranformation on the target variable.

    Return
    ----------
    LinearRegression
        The trained linear regression model.
    """
