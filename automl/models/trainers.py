from typing import Any
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

from automl.logger import get_logger
from automl.config import CONFIG

logger = get_logger("ModelTrainer")


def train_model(
    model_name: str,
    params: dict,
    X_train,
    y_train,
    problem_type: str,
) -> Any:
    """
    Trains a model given its name and parameters.
    """

    logger.info(f"Training model: {model_name}")

    if problem_type == "classification":
        if model_name == "logistic_regression":
            model = LogisticRegression(
                C=params.get("C", 1.0),
                max_iter=params.get("max_iter", 500),
                n_jobs=CONFIG.n_jobs,
            )

        elif model_name == "random_forest":
            model = RandomForestClassifier(
                n_estimators=int(params["n_estimators"]),
                max_depth=int(params["max_depth"]),
                min_samples_split=int(params["min_samples_split"]),
                n_jobs=CONFIG.n_jobs,
                random_state=CONFIG.random_state,
            )

        elif model_name == "extra_trees":
            model = ExtraTreesClassifier(
                n_estimators=int(params["n_estimators"]),
                max_depth=int(params["max_depth"]),
                min_samples_split=int(params["min_samples_split"]),
                n_jobs=CONFIG.n_jobs,
                random_state=CONFIG.random_state,
            )

        elif model_name == "xgboost":
            model = XGBClassifier(
                n_estimators=int(params["n_estimators"]),
                max_depth=int(params["max_depth"]),
                learning_rate=params["learning_rate"],
                subsample=params["subsample"],
                n_jobs=CONFIG.n_jobs,
                random_state=CONFIG.random_state,
                eval_metric="logloss",
                use_label_encoder=False,
            )

        elif model_name == "lightgbm":
            model = LGBMClassifier(
                n_estimators=int(params["n_estimators"]),
                max_depth=int(params["max_depth"]),
                learning_rate=params["learning_rate"],
                num_leaves=int(params["num_leaves"]),
                n_jobs=CONFIG.n_jobs,
                random_state=CONFIG.random_state,
            )

        elif model_name == "catboost":
            model = CatBoostClassifier(
                depth=int(params["depth"]),
                learning_rate=params["learning_rate"],
                random_seed=CONFIG.random_state,
                verbose=False,
            )

        elif model_name == "mlp":
            model = MLPClassifier(
                hidden_layer_sizes=(int(params["hidden_layer_sizes"]),),
                alpha=params["alpha"],
                max_iter=500,
                random_state=CONFIG.random_state,
            )

        else:
            raise ValueError(f"Unknown model: {model_name}")

    else:  # regression
        if model_name == "linear_regression":
            model = LinearRegression(n_jobs=CONFIG.n_jobs)

        elif model_name == "random_forest":
            model = RandomForestRegressor(
                n_estimators=int(params["n_estimators"]),
                max_depth=int(params["max_depth"]),
                min_samples_split=int(params["min_samples_split"]),
                n_jobs=CONFIG.n_jobs,
                random_state=CONFIG.random_state,
            )

        elif model_name == "extra_trees":
            model = ExtraTreesRegressor(
                n_estimators=int(params["n_estimators"]),
                max_depth=int(params["max_depth"]),
                min_samples_split=int(params["min_samples_split"]),
                n_jobs=CONFIG.n_jobs,
                random_state=CONFIG.random_state,
            )

        elif model_name == "xgboost":
            model = XGBRegressor(
                n_estimators=int(params["n_estimators"]),
                max_depth=int(params["max_depth"]),
                learning_rate=params["learning_rate"],
                subsample=params["subsample"],
                n_jobs=CONFIG.n_jobs,
                random_state=CONFIG.random_state,
            )

        elif model_name == "lightgbm":
            model = LGBMRegressor(
                n_estimators=int(params["n_estimators"]),
                max_depth=int(params["max_depth"]),
                learning_rate=params["learning_rate"],
                num_leaves=int(params["num_leaves"]),
                n_jobs=CONFIG.n_jobs,
                random_state=CONFIG.random_state,
            )

        elif model_name == "catboost":
            model = CatBoostRegressor(
                depth=int(params["depth"]),
                learning_rate=params["learning_rate"],
                random_seed=CONFIG.random_state,
                verbose=False,
            )

        elif model_name == "mlp":
            model = MLPRegressor(
                hidden_layer_sizes=(int(params["hidden_layer_sizes"]),),
                alpha=params["alpha"],
                max_iter=500,
                random_state=CONFIG.random_state,
            )

        else:
            raise ValueError(f"Unknown model: {model_name}")

    model.fit(X_train, y_train)
    return model
