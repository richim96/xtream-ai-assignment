"""Module for serializing ML models."""

import pickle

from abc import ABC
from datetime import datetime, timezone
from uuid import uuid4, UUID

import numpy as np
import pandas as pd

from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor

from xtream_service.ml_pipeline import LOGGER
from xtream_service.ml_pipeline.optimization import StdOptimizer


class BaseModel(ABC):
    """Abstract base class for ML models.

    Parameters
    ----------
    model
        Instance of the chosen model object.
    data_uid : str
        Identifier of the training dataset used.

    Attributes
    ----------
    uid : str
        Model identifier. It updates at every training cycle.
    model
        Instance of the chosen model object.
    data_uid : str
        Identifier of the training dataset used.
    metrics : dict[str, float] | None, default=None
        Model evaluation metrics.
    is_sota : bool, default=False
        Whether the model correspons to the state-of-the-art.
    created_at : str
        Time of the object's creation. It updates at every training cycle.

    Methods
    ----------
    info() -> dict
        Create a collection of model info for logging.
    evaluate(x_test: pd.Series, y_test: pd.Series, log_transform: bool) -> None
        Evaluate model performance on preset metrics. The current metrics for
        evaluation are R2 Score and Mean Absolute Error.
    serialize(storage_path: str) -> None
        Serialize model into a persistently stored pickle file.
    """

    def __init__(self, model, data_uid: UUID):
        # The model object identifier and timestamp will update each time the
        # actual model will get trained, for it will not be the same model.
        self.uid: str = ""
        self.model = model
        self.data_uid: str = str(data_uid)
        self.metrics: dict[str, float] | None = None
        self.is_sota: bool = False
        self.created_at: str = ""

    def info(self) -> dict:
        """Create a collection of model info for logging."""
        return {
            "model_uid": self.uid,
            "data_uid": self.data_uid,
            "metrics": self.metrics,
            "is_sota": self.is_sota,
            "created_at": self.created_at,
        }

    def evaluate(
        self, x_test: pd.Series, y_test: pd.Series, log_transform: bool = False
    ) -> None:
        """Evaluate model performance on preset metrics. The current metrics for
        evaluation are R2 Score and Mean Absolute Error.

        Parameters
        ----------
        model
            The model object to evaluate.
        x_test : pd.Series
            The predictors test series on which to carry out the prediction.
        y_test : pd.Series
            The target test series against which to evaluate the prediction.
        log_transform : bool, default=False
            Whether the target variable has been transformed during training.

        Return
        ----------
        dict[str, float]
            The metrics collected.
        """
        y_pred = self.model.predict(x_test)
        y_pred = np.expm1(y_pred) if log_transform else y_pred
        self.metrics = {
            "r2_score": r2_score(y_test, y_pred),
            "mean_absolute_error": mean_absolute_error(y_test, y_pred),
        }

    def serialize(self, storage_path: str) -> None:
        """Serialize model into a persistently stored pickle file.

        Parameters
        ----------
        storage_route : str
            Path for storing the serialized model.
        """
        with open(f"{storage_path}{self.uid}.pkl", "wb") as f:
            LOGGER.info("Model %s saved in persistent storage.", self.uid)
            pickle.dump(self.model, f)


class LinearRegressionModel(BaseModel):
    """Linear regression model.

    Parameters
    ----------
    model
        Instance of the chosen model object.
    data_uid : str
        Identifier of the training dataset used.

    Attributes
    ----------
    uid : str
        Model identifier. It updates at every training cycle.
    model
        Instance of the chosen model object.
    data_uid : str
        Identifier of the training dataset used.
    metrics : dict[str, float] | None, default=None
        Model evaluation metrics.
    is_sota : bool, default=False
        Whether the model correspons to the state-of-the-art.
    created_at : str
        Time of the object's creation. It updates at every training cycle.

    Methods
    ----------
    info() -> dict
        Create a collection of model info for logging.
    evaluate(x_test: pd.Series, y_test: pd.Series, log_transform: bool) -> None
        Evaluate model performance on preset metrics. The current metrics for
        evaluation are R2 Score and Mean Absolute Error.
    serialize(storage_path: str) -> None
        Serialize model into a persistently stored pickle file.
    train(x_train: pd.Series, y_train: pd.Series) -> None
        Train a simple linear regression model.
    """

    def train(
        self, x_train: pd.Series, y_train: pd.Series, log_transform: bool = False
    ) -> None:
        """Train a simple linear regression model.

        Parameters
        ----------
        x_train : pd.Series
            Training series for the predictors.
        y_train : pd.Seris
            Training series for the target variable.
        log_transform : bool, default=False
            If `True`, train the model applying a log tranformation on the target.
        """
        self.uid = str(uuid4())
        self.created_at = str(datetime.now(timezone.utc))
        target: pd.Series = np.log1p(y_train) if log_transform else y_train

        self.model.fit(x_train, target)

        if log_transform:
            LOGGER.info("Linear model %s trained with log transformation.", self.uid)
        else:
            LOGGER.info("Linear model %s trained", self.uid)


class XgbRegressorModel(BaseModel):
    """XGB regressor model.

    Parameters
    ----------
    model
        Instance of the chosen model object.
    data_uid : str
        Identifier of the training dataset used.
    categorical : bool, default=True
            If `True`, it instructs to the regressor that the data is also categorical.
    seed : int, default=42
        Random-state seed allowing for reproducible outputs across multiple calls.

    Attributes
    ----------
    uid : str
        Model identifier. It updates at every training cycle.
    model
        Instance of the chosen model object.
    data_uid : str
        Identifier of the training dataset used.
    metrics : dict[str, float] | None, default=None
        Model evaluation metrics.
    is_sota : bool, default=False
        Whether the model correspons to the state-of-the-art.
    created_at : str
        Time of the object's creation. It updates at every training cycle.
    optimizer : StdOptimizer, default=None
            Optimizer instance to fine tune the model's hyperparameters.
    categorical : bool, default=True
            If `True`, it instructs to the regressor that the data is also categorical.
    seed : int, default=42
        Random-state seed allowing for reproducible outputs across multiple calls.

    Methods
    ----------
    info() -> dict
        Create a collection of model info for logging.
    evaluate(x_test: pd.Series, y_test: pd.Series, log_tranform: bool) -> None
        Evaluate model performance on preset metrics. The current metrics for
        evaluation are R2 Score and Mean Absolute Error.
    serialize(storage_path: str) -> None
        Serialize model into a persistently stored pickle file.
    train(x_train: pd.Series, y_train: pd.Series) -> None
        Train a gradient boosting model. Hyperparameter optimization available.
    """

    def __init__(self, model, data_uid: UUID, categorical: bool = True, seed: int = 42):
        self.optimizer: StdOptimizer | None = None
        self.categorical = categorical
        self.seed = seed
        super().__init__(model, data_uid)

    def train(self, x_train: pd.Series, y_train: pd.Series) -> None:
        """Train a gradient boosting model. Hyperparameter optimization available.

        Parameters
        ----------
        x_train : pd.Series
            Training series for the predictors.
        y_train : pd.Seris
            Training series for the target variable.
        """
        self.uid = str(uuid4())
        self.created_at = str(datetime.now(timezone.utc))

        if self.optimizer is not None:
            LOGGER.info("Tuning hyperparameters for model %s...", self.uid)
            self.optimizer.optuna_study.optimize(
                func=self.optimizer.std_objective_fn,
                n_trials=self.optimizer.opt_n_trials,
                show_progress_bar=True,
            )
            self.model = XGBRegressor(
                **self.optimizer.optuna_study.best_params,
                enable_categorical=self.categorical,
                random_state=self.seed,
            )

        self.model.fit(x_train, y_train)
        LOGGER.info("Gradient boosting model %s trained.", self.uid)
