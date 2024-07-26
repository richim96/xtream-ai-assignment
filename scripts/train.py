"""Automated script to train models with fresh data"""

import pandas as pd

from xtream_service.ml_pipeline import (
    data_extraction,
    preprocessing,
    optimization,
    training,
)

# Define the data source before starting the training cycle.
DATA_SOURCE: str = (
    "https://raw.githubusercontent.com/xtreamsrl/xtream-ai-assignment-engineer/main/datasets/diamonds/diamonds.csv"
)


def main() -> None:
    """Run the ML pipeline end-to-end, save training logs and finde the sota model."""
    # Extract and clean data
    df: pd.DataFrame = data_extraction.extract_from_csv(DATA_SOURCE)
    df = preprocessing.filter_numeric(df, cols=["carat", "price", "x", "y", "z"], n=0)

    # Prepare data for Liner Regression
    df_linear: pd.DataFrame = df.drop(columns=["depth", "table", "y", "z"])
    df_linear = preprocessing.dummy_encode(df_linear, ["cut", "color", "clarity"])

    # Prepare data for XGB Regression
    df_xgb: pd.DataFrame = df.copy()


if __name__ == "__main__":
    main()
