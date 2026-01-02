from typing import Dict, Any


def get_model_search_space(problem_type: str) -> Dict[str, Dict[str, Any]]:
    """
    Returns model classes and their hyperparameter search spaces.
    """

    if problem_type == "classification":
        return {
            "logistic_regression": {
                "model": "logistic_regression",
                "params": {
                    "C": (0.01, 10.0),
                    "max_iter": 500,
                },
            },

            "random_forest": {
                "model": "random_forest",
                "params": {
                    "n_estimators": (100, 500),
                    "max_depth": (3, 20),
                    "min_samples_split": (2, 10),
                },
            },

            "extra_trees": {
                "model": "extra_trees",
                "params": {
                    "n_estimators": (100, 500),
                    "max_depth": (3, 20),
                    "min_samples_split": (2, 10),
                },
            },

            "xgboost": {
                "model": "xgboost",
                "params": {
                    "n_estimators": (100, 500),
                    "max_depth": (3, 10),
                    "learning_rate": (0.01, 0.3),
                    "subsample": (0.6, 1.0),
                },
            },

            "lightgbm": {
                "model": "lightgbm",
                "params": {
                    "n_estimators": (100, 500),
                    "max_depth": (3, 10),
                    "learning_rate": (0.01, 0.3),
                    "num_leaves": (20, 100),
                },
            },

            "catboost": {
                "model": "catboost",
                "params": {
                    "depth": (4, 10),
                    "learning_rate": (0.01, 0.3),
                },
            },

            "mlp": {
                "model": "mlp",
                "params": {
                    "hidden_layer_sizes": (50, 200),
                    "alpha": (1e-5, 1e-2),
                },
            },
        }

    elif problem_type == "regression":
        return {
            "linear_regression": {
                "model": "linear_regression",
                "params": {},
            },

            "random_forest": {
                "model": "random_forest",
                "params": {
                    "n_estimators": (100, 500),
                    "max_depth": (3, 20),
                    "min_samples_split": (2, 10),
                },
            },

            "extra_trees": {
                "model": "extra_trees",
                "params": {
                    "n_estimators": (100, 500),
                    "max_depth": (3, 20),
                    "min_samples_split": (2, 10),
                },
            },

            "xgboost": {
                "model": "xgboost",
                "params": {
                    "n_estimators": (100, 500),
                    "max_depth": (3, 10),
                    "learning_rate": (0.01, 0.3),
                    "subsample": (0.6, 1.0),
                },
            },

            "lightgbm": {
                "model": "lightgbm",
                "params": {
                    "n_estimators": (100, 500),
                    "max_depth": (3, 10),
                    "learning_rate": (0.01, 0.3),
                    "num_leaves": (20, 100),
                },
            },

            "catboost": {
                "model": "catboost",
                "params": {
                    "depth": (4, 10),
                    "learning_rate": (0.01, 0.3),
                },
            },

            "mlp": {
                "model": "mlp",
                "params": {
                    "hidden_layer_sizes": (50, 200),
                    "alpha": (1e-5, 1e-2),
                },
            },
        }

    else:
        raise ValueError(f"Unsupported problem type: {problem_type}")
