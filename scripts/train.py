"""Automated script to train models with fresh data"""

from uuid import uuid4, UUID

import pandas as pd

from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from xtream_service.ml_pipeline.data_extraction import extract_from_csv
from xtream_service.ml_pipeline import models, model_selection, preprocessing
from xtream_service.ml_pipeline.optimization import StdOptimizer

# Define the data source before starting the training cycle.
DATA_SOURCE: str = "assets/data/raw/diamonds/diamonds.csv"


# TODO: break down script into separate functions
if __name__ == "__main__":
    # Run the ML pipeline end-to-end, save training logs and finde the sota model
    lr: models.LinearRegressionModel
    xgb: models.XgbRegressorModel

    model_objs: list = []

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

    # for i in range(5):
    lr = models.LinearRegressionModel(LinearRegression(), df_linear_uid)
    lr.train(x_train_lr, y_train_lr)
    lr.evaluate(x_test_lr, y_test_lr)
    model_objs.append(lr)
    lr2 = models.LinearRegressionModel(LinearRegression(), df_linear_uid)
    lr2.train(x_train_lr, y_train_lr, log_transform=True)
    lr2.evaluate(x_test_lr, y_test_lr)
    model_objs.append(lr2)

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

    xgb = models.XgbRegressorModel(
        XGBRegressor(enable_categorical=True, random_state=42), df_xgb_uid
    )
    xgb.train(x_train_xgb, y_train_xgb)
    xgb.evaluate(x_test_xgb, y_test_xgb)
    model_objs.append(xgb)
    xgb2 = models.XgbRegressorModel(
        XGBRegressor(enable_categorical=True, random_state=42), df_xgb_uid
    )
    xgb2.optimizer = StdOptimizer(x_train_xgb, y_train_xgb)
    xgb2.train(x_train_xgb, y_train_xgb)
    xgb2.evaluate(x_test_xgb, y_test_xgb)
    model_objs.append(xgb2)

    # Get Sota
    model_selection.sota_set(model_objs)
    for model in model_objs:
        if not model.is_sota:
            model.serialize("assets/models")
        else:
            model.serialize("assets/models/sota")
