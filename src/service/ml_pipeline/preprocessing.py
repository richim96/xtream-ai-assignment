"""Submodule for the data preprocessing workflow"""

import pandas as pd

from sklearn.model_selection import train_test_split


def filter_numeric(df: pd.DataFrame, cols: list[str], n: int | float) -> pd.DataFrame:
    """From the given numerical columns, drop all rows with a value smaller than
    or equal to n.

    Parameters
    ----------
    df : pd.DataFrame
        Pandas dataframe.
    cols : list[str]
        Numerical columns on which to operate the filter.
    n : int | float
        Value against which to filter the given numerical columns.

    Return
    ----------
    pd.DataFrame
        The filtered pandas dataframe.
    """
    pd_expression = df[cols[0]] > n

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


def categorical_to_ordinal(
    df: pd.DataFrame, cols: list[str], categories: list[str], ordered: bool = True
) -> pd.DataFrame:
    """
    Parameters
    ----------
    df : pd.DataFrame
        Pandas datatframe.
    cols : list[str]
        Categorical columns to transform.
    categories : list[str]
        The categories to transform into ordinal values.
    ordered : bool
        If true, each column is considered an ordered categorical.

    Return
    ----------
    pd.DataFrame
        The transformed pandas dataframe.
    """
    for col in cols:
        df[col] = pd.Categorical(df[col], categories=categories, ordered=ordered)

    return df


def train_test_data_get(
    df: pd.DataFrame, target: str, test_size: float = 0.2, seed: int = 42
) -> list[pd.Series]:
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
    seed : int
        Random-state seed to control data shuffling before the split. It allows
        for reproducible output across multiple calls.

    Return
    ----------
    list[pd.Series]
        X and Y train and test sets.
    """
    x: pd.DataFrame = df.drop(columns=target)
    y: pd.Series = df[target]

    return train_test_split(x, y, test_size=test_size, random_state=seed)
