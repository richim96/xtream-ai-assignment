"""Module for the hyperparameter optimization workflow."""

import pandas as pd

from optuna.trial import Trial
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


class StdOptimizer:
    """Standard Optimizer for hyperparameter tuning. The parameters passed to this
    class must correspond to the ones intended to use for the model to optimize.

    Parameters
    ----------
    x_train : pd.Series
        Training series for the predictors.
    y_train : pd.Seris
        Training series for the target variable.
    seed : int
        Random-state seed allowing for reproducible outputs across multiple calls.
    test_size : float
        Determines the size of the testing set. The training set is complemented
        by default. The acceptables values range between 0.0 and 1.0.

    Attributes
    ----------
    x_train : pd.Series
        Training series for the predictors.
    y_train : pd.Seris
        Training series for the target variable.
    test_size : float, default=0.2
        Determines the size of the testing set. The training set is complemented
        by default. The acceptables values range between 0.0 and 1.0.
    seed : int, default=42
        Random-state seed allowing for reproducible outputs across multiple calls.
    study_diretion : str, default="minimize"
        Direction of the optuna study. Default set to `minimize`.
    opt_n_trials : int, default=100
        Number of trials for each optimization process. Default set to `100`.

    Methods
    ----------
    std_objective_fn(trial: Trial) -> float:
        Standard objective function for hyperparameter tuning. It trains a
        gradient boosting model with the hyperparameters to later optimize.
    """

    def __init__(
        self,
        x_train: pd.Series,
        y_train: pd.Series,
        test_size: float = 0.2,
        seed: int = 42,
    ) -> None:
        self.x_train: pd.Series = x_train
        self.y_train: pd.Series = y_train
        self.test_size: float = test_size
        self.seed: int = seed
        self.study_direction: str = "minimize"
        self.opt_n_trials: int = 100

    def std_objective_fn(self, trial: Trial) -> float:
        """Standard objective function for hyperparameter tuning. It trains a
        gradient boosting model with the hyperparameters to later optimize.

        Parameters
        ----------
        trial : Trial
            Optuna `Trial` object.

        Retrun
        ----------
        float
            Mean absolute error of the model's prediction.
        """
        params: dict = {
            "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
            "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
            "colsample_bytree": trial.suggest_categorical(
                "colsample_bytree", [0.3, 0.4, 0.5, 0.7]
            ),
            "subsample": trial.suggest_categorical(
                "subsample", [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            ),
            "learning_rate": trial.suggest_float("learning_rate", 1e-8, 1.0, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "random_state": 42,
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "enable_categorical": True,
        }

        x_train_xgb, x_test_xgb, y_train_xgb, y_test_xgb = train_test_split(
            self.x_train, self.y_train, test_size=self.test_size, random_state=self.seed
        )
        xgb: XGBRegressor = XGBRegressor(**params)
        xgb.fit(x_train_xgb, y_train_xgb)

        predictions = xgb.predict(x_test_xgb)
        mae: float = mean_absolute_error(y_test_xgb, predictions)

        return mae
