"""Script to train models with fresh data"""

import argparse
import json
import os

from random import randint

import pandas as pd

from dotenv import load_dotenv, find_dotenv
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from xtream_service.utils import uuid_get, utc_time_get
from xtream_service.ml_pipeline._const import (
    NUMERIC,
    CATEGORICAL,
    LINEAR_TO_DROP,
    XGB_TO_CATG,
    TARGET,
)
from xtream_service.ml_pipeline import LOGGER, data_processing
from xtream_service.ml_pipeline import models, model_selection
from xtream_service.ml_pipeline.optimization import StdOptimizer

load_dotenv(find_dotenv())


def cli_args_get():
    """Define and retrieve the CLI arguments to use in the script.

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
    parser = argparse.ArgumentParser(prog="python run_ml_pipeline.py")
    parser.add_argument("-ds", "--data-source", type=str, help="Data source path.")
    parser.add_argument("-dd", "--data-dest", type=str, help="Data storage path.")
    parser.add_argument("-m", "--model-dest", type=str, help="Models storage path.")
    parser.add_argument("-s", "--sota-dest", type=str, help="SOTA model storage path.")
    parser.add_argument("-l", "--log-dest", type=str, help="Log storage path.")
    parser.add_argument(
        "-n", "--n-models", type=int, help="Number of training attempts per model type."
    )

    return parser.parse_args()


def linear_models_train(
    df_ln: pd.DataFrame, df_storage_path: str, n_models: int = 1
) -> list:
    """Training linear models.

    Parameters
    ----------
    df_ln : pd.DataFrame
        The dataframe to process before training starts.
    df_storage_path : str
        Where to store the processed dataframe.
    n_models : int, default=5
        Number of models to train, each with a different random state.

    Return
    ----------
    list
        A list with the model objects.
    """
    # Preprocess data and store processed dataset for future reference
    df_id: str = uuid_get()
    df_ln = df_ln.drop(columns=LINEAR_TO_DROP)
    df_ln = data_processing.dummy_encode(df_ln, cols=CATEGORICAL)
    df_ln.to_csv(f"{df_storage_path}{df_id}.csv")

    # Train linear models
    lr_models: list = []
    n_models = max(n_models, 1)  # the number of models must be at least one
    for _ in range(n_models):
        x_train, x_test, y_train, y_test = data_processing.train_test_data_get(
            df_ln, target=TARGET, seed=randint(1, n_models * 10)
        )
        lr = models.LinearRegressionModel(LinearRegression(), df_id)
        lr.train(x_train, y_train, log_transform=True)
        lr.evaluate(x_test, y_test, log_transform=True)
        lr_models.append(lr)

    return lr_models


def gradient_boosting_train(
    df_xgb: pd.DataFrame, df_dest: str, n_models: int = 5
) -> list:
    """Train gradient boosting models.

    Parameters
    ----------
    df_xgb : pd.DataFrame
        The dataframe to process before training starts.
    df_storage_path : str
        Where to store the processed dataframe.
    n_models : int, default=5
        Number of models to train, each with a different random state.

    Return
    ----------
    list
        A list with the model objects.
    """
    # Preprocess data and store processed dataset for future reference
    df_id: str = uuid_get()
    df_xgb = data_processing.to_categorical_dtype(df_xgb, XGB_TO_CATG)
    df_xgb.to_csv(f"{df_dest}{df_id}.csv")

    # Train XGB models
    xgb_models: list = []
    n_models = max(n_models, 1)  # the number of models must be at least one
    for _ in range(n_models):
        seed: int = randint(1, n_models * 10)
        x_train, x_test, y_train, y_test = data_processing.train_test_data_get(
            df_xgb, target=TARGET, seed=seed
        )
        xgb = models.XgbRegressorModel(
            XGBRegressor(enable_categorical=True, random_state=seed), df_id
        )
        xgb.optimizer = StdOptimizer(x_train, y_train, seed=seed)
        xgb.train(x_train, y_train)
        xgb.evaluate(x_test, y_test)
        xgb_models.append(xgb)

    return xgb_models


if __name__ == "__main__":
    # ----- Run ML pipeline E2E ----- #
    args = cli_args_get()
    DATA_SOURCE = args.data_source or os.getenv("DATA_SOURCE", "")
    DATA_DEST = args.data_dest or os.getenv("DATA_DEST", "")
    MODEL_DEST = args.model_dest or os.getenv("MODEL_DEST", "")
    SOTA_DEST = args.sota_dest or os.getenv("SOTA_DEST", "")
    LOG_DEST = args.log_dest or os.getenv("LOG_DEST", "")
    N_MODELS = args.n_models if args.n_models is not None else 1

    # Extract and clean data
    df: pd.DataFrame = data_processing.extract_from_csv(DATA_SOURCE)
    df = data_processing.filter_numeric(df, cols=NUMERIC, n=0)
    # Train models
    model_objs: list = linear_models_train(df.copy(), DATA_DEST, n_models=N_MODELS)
    model_objs += gradient_boosting_train(df.copy(), DATA_DEST, n_models=N_MODELS)

    # Set SOTA and log training cycle
    try:
        with open(LOG_DEST, "r", encoding="utf-8") as f:
            log: dict = json.load(f)
    except FileNotFoundError:
        log = {"log_id": uuid_get(), "data": [], "training_cycles": 0}
        LOGGER.info("Log file not found at: (%s. New log created.", LOG_DEST)

    log = model_selection.sota_update(model_objs, log)
    log["data"] += [model.info() for model in model_objs]
    log["training_cycles"] += 1
    log["updated_at"] = utc_time_get()
    with open(f"{LOG_DEST}", "w", encoding="utf-8") as f:
        json.dump(log, f, indent=4)

    # Serialize newly trained models
    for model in model_objs:
        model.serialize(MODEL_DEST)
        if model.is_sota:
            model.serialize(SOTA_DEST + f"cycle_{log["training_cycles"]}_sota_")
