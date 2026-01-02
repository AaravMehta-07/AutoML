import optuna
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
from automl.logger import get_logger
from automl.config import CONFIG
from automl.models.trainers import train_model
from automl.optimization.metrics import get_metric_fn

logger = get_logger("HyperOpt")


class HyperparameterOptimizer:
    def __init__(self, problem_type: str, metric_name: str = None):
        self.problem_type = problem_type

        # UPDATED: now also receives needs_proba
        self.metric_fn, self.higher_is_better, self.needs_proba = get_metric_fn(
            problem_type, metric_name
        )

    def _suggest_params(self, trial, param_space: dict) -> dict:
        params = {}
        for k, v in param_space.items():
            if isinstance(v, tuple):
                params[k] = trial.suggest_float(k, v[0], v[1])
            else:
                params[k] = v
        return params

    def optimize_model(
        self,
        model_name: str,
        param_space: dict,
        X,
        y,
    ):
        logger.info(f"Optimizing model: {model_name}")

        def objective(trial):
            params = self._suggest_params(trial, param_space)

            scores = []

            if self.problem_type == "classification":
                cv = StratifiedKFold(
                    n_splits=CONFIG.cv_folds,
                    shuffle=True,
                    random_state=CONFIG.random_state,
                )
            else:
                cv = KFold(
                    n_splits=CONFIG.cv_folds,
                    shuffle=True,
                    random_state=CONFIG.random_state,
                )

            for train_idx, val_idx in cv.split(X, y):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                model = train_model(
                    model_name,
                    params,
                    X_train,
                    y_train,
                    self.problem_type,
                )

                # UPDATED: metric-aware prediction logic
                if self.problem_type == "classification":
                    if self.needs_proba and hasattr(model, "predict_proba"):
                        preds = model.predict_proba(X_val)[:, 1]
                    else:
                        preds = model.predict(X_val)
                else:
                    preds = model.predict(X_val)

                score = self.metric_fn(y_val, preds)
                scores.append(score)

            mean_score = np.mean(scores)
            std_penalty = np.std(scores)

            final_score = (
                mean_score - std_penalty
                if self.higher_is_better
                else mean_score + std_penalty
            )

            return final_score if self.higher_is_better else -final_score

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=CONFIG.max_trials)

        best_score = study.best_value
        best_params = study.best_params

        logger.info(
            f"Best score for {model_name}: {best_score:.5f}"
        )

        return best_params, best_score
