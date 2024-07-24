"""Sub-module for the model training workflow"""

import polars as pl

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


def linear_model_train(
    df: pl.DataFrame, test_size: float = 0.2, state: int = 42
) -> LinearRegression:
    """Train a simple linear regression model.

    Parameters
    ----------
    df : pl.DataFrame
        Polars datatframe.
    test_size : float
        Determines the size of the testing set. The training set is complemented
        by default. The acceptables values range between 0.0 and 1.0.
    state : int
        Random-state seed to control data shuffling before the split. It allows
        for reproducible output across multiple calls.

    Return
    ----------
    LinearRegression
        The trained linear regression model.
    """
    pass
