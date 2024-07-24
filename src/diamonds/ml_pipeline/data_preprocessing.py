"""Sub-module for the data preprocessing workflow"""

import polars as pl
from polars.expr.expr import Expr


def filter_numerical(df: pl.DataFrame, cols: list[str], n: int = 0) -> pl.DataFrame:
    """From the given numerical columns, drop all rows with a value smaller than
    or equal to n.

    Parameters
    ----------
    df : pl.DataFrame
        Polars dataframe.
    cols : list[str]
        Numerical columns on which to operate the filter.
    n : int
        Value against which to filter the given numerical columns.

    Return
    ----------
    pl.DataFrame
        The filtered polars dataframe.
    """
    pl_expression: Expr = pl.col(cols[0]) > n

    for col in cols[1:]:
        pl_expression &= pl.col(col) > n

    return df.filter(pl_expression)


def dummy_encode(df: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    """Perform one-hot encoding on the given columns.

    Parameters
    ----------
    df : pl.DataFrame
        Polars dataframe.
    cols : list[str]
        Categorical columns to encode.

    Return
    ----------
    pl.DataFrame
        The encoded polars dataframe.
    """
    return df.to_dummies(columns=cols, drop_first=True)


def cols_drop(df: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    """Drop the given columns.

    Parameters
    ----------
    df : pl.DataFrame
        Polars dataframe.
    cols : list[str]
        Columns to drop.

    Return
    ----------
    pl.DataFrame
        The new polars dataframe.
    """
    return df.drop(cols)
