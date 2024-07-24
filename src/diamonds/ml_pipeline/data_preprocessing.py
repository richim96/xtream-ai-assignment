"""Sub-module for the data preprocessing workflow"""

import pandas as pd
from sklearn.model_selection import train_test_split


def filter_numerical(df: pd.DataFrame, cols: list[str], n: int = 0) -> pd.DataFrame:
    """From the given numerical columns, drop all rows with a value smaller than
    or equal to n.

    Parameters
    ----------
    df : pd.DataFrame
        Pandas dataframe.
    cols : list[str]
        Numerical columns on which to operate the filter.
    n : int
        Value against which to filter the given numerical columns.

    Return
    ----------
    pd.DataFrame
        The filtered pandas dataframe.
    """
    pd_expression: pd.Expr = df[cols[0]] > n

    for col in cols[1:]:
        pd_expression &= df[col] > n

    return df[pd_expression]


def dummy_encode(df: pd.DataFrame, cols: str | list[str]) -> pd.DataFrame:
    """Perform one-hot encoding on the given categorical columns.

    Parameters
    ----------
    df : pd.DataFrame
        Pandas dataframe.
    cols : list[str]
        Categorical columns to encode.

    Return
    ----------
    pd.DataFrame
        The encoded pandas dataframe.
    """
    return pd.get_dummies(data=df, columns=cols, drop_first=True)


def cols_drop(df: pd.DataFrame, cols: str | list[str]) -> pd.DataFrame:
    """Drop the given columns.

    Parameters
    ----------
    df : pd.DataFrame
        Pandas dataframe.
    cols : list[str]
        Columns to drop.

    Return
    ----------
    pd.DataFrame
        The updated pandas dataframe.
    """
    return df.drop(columns=cols)


def train_test(
    df: pd.DataFrame, target: str, test_size: float = 0.2, state: int = 42
) -> tuple[pd.Series]:
    """
    Parameters
    ----------
    df : pd.DataFrame
        Pandas datatframe.
    target : str
        Target column for the model's prediction.
    test_size : float
        Determines the size of the testing set. The training set is complemented
        by default. The acceptables values range between 0.0 and 1.0.
    state : int
        Random-state seed to control data shuffling before the split. It allows
        for reproducible output across multiple calls.

    Return
    ----------
    tuple[pd.Series]
        X and Y train and test series.
    """
    x: pd.DataFrame = cols_drop(df, target)
    y: pd.Series = df[target]

    return train_test_split(x, y, test_size=test_size, random_state=state)
