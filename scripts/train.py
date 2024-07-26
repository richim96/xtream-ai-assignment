"""Automated script to train models with fresh data"""

# from datetime import datetime
from uuid import uuid4, UUID

import pandas as pd

from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from xtream_service.ml_pipeline.data_extraction import extract_from_csv
from xtream_service.ml_pipeline import (
    preprocessing,
    training,
    model_selection,
    model_serialization,
)
from xtream_service.ml_pipeline.optimization import StdOptimizer

# Define the data source before starting the training cycle.
DATA_SOURCE: str = "assets/data/raw/diamonds/diamonds.csv"


def main() -> None:
    """Run the ML pipeline end-to-end, save training logs and finde the sota model."""
    lr: LinearRegression
    xgb: XGBRegressor

    model_objs: list[dict] = []

    # ----- Extract and clean data ----- #
    df: pd.DataFrame = extract_from_csv(DATA_SOURCE)
    df = preprocessing.filter_numeric(df, cols=["carat", "price", "x", "y", "z"], n=0)

    # ----- Linear Regression Models ----- #
    df_linear_uid: UUID = uuid4()
    df_linear: pd.DataFrame = df.drop(columns=["depth", "table", "y", "z"])
    df_linear = preprocessing.dummy_encode(df_linear, ["cut", "color", "clarity"])
    df_linear.to_csv(f"assets/data/processed/{df_linear_uid}.csv")

    x_train_lr, x_test_lr, y_train_lr, y_test_lr = preprocessing.train_test_data_get(
        df_linear, target="price"
    )

    lr = training.linear_regression_train(x_train_lr, y_train_lr)
    model_objs.append(model_serialization.model_object_make(lr, df_linear_uid))
    lr = training.linear_regression_train(x_train_lr, y_train_lr, log_transform=True)
    model_objs.append(model_serialization.model_object_make(lr, df_linear_uid))

    # ----- Gradient Boosting Models ----- #
    df_xgb_uid: UUID = uuid4()
    df_xgb: pd.DataFrame = df.copy()
    df_xgb = preprocessing.to_categorical_dtype(
        df_xgb,
        targets={
            "cut": ["Fair", "Good", "Very Good", "Ideal", "Premium"],
            "color": ["D", "E", "F", "G", "H", "I", "J"],
            "clarity": ["IF", "VVS1", "VVS2", "VS1", "VS2", "SI1", "SI2", "I1"],
        },
    )
    df_xgb.to_csv(f"assets/data/processed/{df_xgb_uid}.csv")

    x_train_xgb, x_test_xgb, y_train_xgb, y_test_xgb = (
        preprocessing.train_test_data_get(df_xgb, target="price")
    )

    xgb = training.xgb_regressor_train(x_train_xgb, y_train_xgb)
    model_objs.append(model_serialization.model_object_make(xgb, df_xgb_uid))
    xgb = training.xgb_regressor_train(
        x_train_xgb, y_train_xgb, optimizer=StdOptimizer(x_train_xgb, y_train_xgb)
    )
    model_objs.append(model_serialization.model_object_make(xgb, df_xgb_uid))

    # Evaluate models and store metadata logs
    print(model_objs)


if __name__ == "__main__":
    main()
