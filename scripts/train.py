"""Automated script to train models with fresh data"""

import argparse
import os
import json

from datetime import datetime, timezone
from random import randint
from uuid import uuid4, UUID

import pandas as pd

from dotenv import load_dotenv, find_dotenv
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from xtream_service.ml_pipeline.data_extraction import extract_from_csv
from xtream_service.ml_pipeline import models, model_selection, preprocessing
from xtream_service.ml_pipeline.optimization import StdOptimizer

load_dotenv(find_dotenv())


def _cli_args_get():
    """Define and retrieve CLI arguments to use in the script.

    Possible Arguments
    ----------
    data_source : str
        Source of the training data.
    data_target : str
        Path to where to store processed data.
    models_target : str
        Path to where to store the models trained.
    sota_target : str
        Path to where to store the SOTA model.
    log_target : str
        Path to where to store the training log.
    n_models : int
        Number of training attempts per model (linear, gradient boosting, etc.).
    """
    parser = argparse.ArgumentParser("ML Pipeline Argument Parser")

    parser.add_argument("-ds", "--data-source", type=str, help="Data source path.")
    parser.add_argument("-dt", "--data-target", type=str, help="Data storage path.")
    parser.add_argument("-mt", "--models-target", type=str, help="Models storage path.")
    parser.add_argument(
        "-st", "--sota-target", type=str, help="SOTA model storage path."
    )
    parser.add_argument("-lt", "--log-target", type=str, help="Log storage path.")
    parser.add_argument(
        "-n", "--n-models", type=int, help="Number of training attempts per model."
    )

    return parser.parse_args()


def _linear_models_train(
    df_ln: pd.DataFrame, df_storage_path: str, n_models: int = 5
) -> list:
    """Launch training for linear models.

    Parameters
    ----------
    df_ln : pd.DataFrame
        The dataframe to process before training starts.
    df_storage_path : str
        Where to store the processed dataframe.
    n_models : int, default=5
        Number of models to train, each with a different randomm state.

    Return
    ----------
    list
        A list with the model objects.
    """
    lr: models.LinearRegressionModel

    df_uid: UUID = uuid4()
    df_ln.drop(columns=["depth", "table", "y", "z"], inplace=True)
    df_ln = preprocessing.dummy_encode(df_ln, ["cut", "color", "clarity"])
    df_ln.to_csv(f"{df_storage_path}{df_uid}.csv")

    lr_models: list = []
    for _ in range(n_models):
        x_train, x_test, y_train, y_test = preprocessing.train_test_data_get(
            df_ln, target="price", seed=randint(1, n_models * 10)
        )

        lr = models.LinearRegressionModel(LinearRegression(), df_uid)
        lr.train(x_train, y_train, log_transform=True)
        lr.evaluate(x_test, y_test)

        lr_models.append(lr)

    return lr_models


def _gradient_boosting_train(
    df_xgb: pd.DataFrame, df_storage_path: str, n_models: int = 5
) -> list:
    """Launch training for gradient boosting models.

    Parameters
    ----------
    df_xgb : pd.DataFrame
        The dataframe to process before training starts.
    df_storage_path : str
        Where to store the processed dataframe.
    n_models : int, default=5
        Number of models to train, each with a different randomm state.

    Return
    ----------
    list
        A list with the model objects.
    """
    xgb: models.XgbRegressorModel

    df_xgb_uid: UUID = uuid4()
    df_xgb = preprocessing.to_categorical_dtype(
        df_xgb,
        targets={
            "cut": ["Fair", "Good", "Very Good", "Ideal", "Premium"],
            "color": ["D", "E", "F", "G", "H", "I", "J"],
            "clarity": ["IF", "VVS1", "VVS2", "VS1", "VS2", "SI1", "SI2", "I1"],
        },
    )
    df_xgb.to_csv(f"{df_storage_path}{df_xgb_uid}.csv")

    xgb_models: list = []
    for _ in range(n_models):
        seed: int = randint(1, n_models * 10)

        x_train, x_test, y_train, y_test = preprocessing.train_test_data_get(
            df_xgb, target="price", seed=seed
        )

        xgb = models.XgbRegressorModel(
            XGBRegressor(enable_categorical=True, random_state=seed), df_xgb_uid
        )
        xgb.optimizer = StdOptimizer(x_train, y_train, seed=seed)
        xgb.train(x_train, y_train)
        xgb.evaluate(x_test, y_test)

        xgb_models.append(xgb)

    return xgb_models


if __name__ == "__main__":
    # ----- Run the ML pipeline E2E ----- #
    args = _cli_args_get()
    # Set data sources and targets
    DATA_SOURCE_PATH = args.data_source or os.getenv("DATA_SOURCE_PATH", "")
    DATA_TARGET_PATH = args.data_target or os.getenv("DATA_TARGET_PATH", "")
    MODELS_TARGET_PATH = args.models_target or os.getenv("MODELS_TARGET_PATH", "")
    SOTA_TARGET_PATH = args.sota_target or os.getenv("SOTA_TARGET_PATH", "")
    LOG_TARGET_PATH = args.log_target or os.getenv("LOG_TARGET_PATH", "")
    N_MODELS = args.n_models or int(os.getenv("N_MODELS", ""))

    # Extract and clean data
    df: pd.DataFrame = extract_from_csv(DATA_SOURCE_PATH)
    df = preprocessing.filter_numeric(df, cols=["carat", "price", "x", "y", "z"], n=0)

    # Prepare and train models, set SOTA and log training cycle
    model_objs: list = _linear_models_train(
        df.copy(), DATA_TARGET_PATH, n_models=N_MODELS
    )
    model_objs += _gradient_boosting_train(
        df.copy(), DATA_TARGET_PATH, n_models=N_MODELS
    )

    try:
        with open(LOG_TARGET_PATH, "r", encoding="utf-8") as f:
            log: dict = json.load(f)
    except FileNotFoundError:
        log = {"log_uid": str(uuid4()), "data": [], "training_cycles": 0}

    log = model_selection.sota_update(model_objs, log)
    log["data"] += [model.info() for model in model_objs]
    log["training_cycles"] += 1
    log["updated_at"] = str(datetime.now(timezone.utc))
    with open(f"{LOG_TARGET_PATH}", "w", encoding="utf-8") as f:
        json.dump(log, f, indent=4)

    # Serialize newly trained models
    for model in model_objs:
        model.serialize(MODELS_TARGET_PATH)
        if model.is_sota:
            sota_name: str = f"cycle_{str(log["training_cycles"])}_sota_"
            model.serialize(SOTA_TARGET_PATH + sota_name)
