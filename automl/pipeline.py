import pandas as pd
import joblib
import os

from automl.logger import get_logger
from automl.config import CONFIG
from automl.profiler.data_profiler import DataProfiler
from automl.features.numeric import NumericFeatureEngineer
from automl.features.datetime import DatetimeFeatureEngineer
from automl.features.feature_selector import FeatureSelector
from automl.models.model_space import get_model_search_space
from automl.models.trainers import train_model
from automl.optimization.hyperopt import HyperparameterOptimizer

from sklearn.model_selection import train_test_split

logger = get_logger("AutoMLPipeline")


class AutoMLPipeline:
    def __init__(self, data_path: str, target: str, metric: str = None):
        self.data_path = data_path
        self.target = target
        self.metric = metric

    def run(self):
        logger.info("Starting AutoML pipeline")

        # ---------------- Load Data ----------------
        df = pd.read_csv(self.data_path)

        # ---------------- Profiling ----------------
        profiler = DataProfiler(df, self.target)
        metadata = profiler.profile()

        problem_type = metadata["problem_type"]

        X = df.drop(columns=[self.target])
        y = df[self.target]

        # ---------------- Feature Engineering ----------------
        if metadata["feature_types"]["numeric"]:
            num_engineer = NumericFeatureEngineer(
                metadata["feature_types"]["numeric"]
            )
            X = num_engineer.transform(X)

        if metadata["feature_types"]["datetime"]:
            dt_engineer = DatetimeFeatureEngineer(
                metadata["feature_types"]["datetime"]
            )
            X = dt_engineer.transform(X)

        # ---------------- Feature Selection ----------------
        selector = FeatureSelector(problem_type)
        X = selector.fit_transform(X, y)

        # ---------------- Train/Test Split ----------------
        stratify = y if problem_type == "classification" else None
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=CONFIG.test_size,
            random_state=CONFIG.random_state,
            stratify=stratify,
        )

        # ---------------- Model Search ----------------
        search_space = get_model_search_space(problem_type)
        optimizer = HyperparameterOptimizer(problem_type, self.metric)

        best_model = None
        best_score = None
        best_name = None
        best_params = None

        for model_name, model_cfg in search_space.items():
            params, score = optimizer.optimize_model(
                model_name,
                model_cfg["params"],
                X_train,
                y_train,
            )

            if best_score is None or score > best_score:
                best_score = score
                best_model = model_name
                best_params = params
                best_name = model_name

        logger.info(f"Best model selected: {best_name}")

        # ---------------- Final Training ----------------
        final_model = train_model(
            best_name,
            best_params,
            X_train,
            y_train,
            problem_type,
        )

        # ---------------- Save Artifacts ----------------
        os.makedirs(CONFIG.model_dir, exist_ok=True)

        joblib.dump(
            final_model,
            os.path.join(CONFIG.model_dir, "best_model.pkl"),
        )

        joblib.dump(
            {
                "model": best_name,
                "params": best_params,
                "features": X.columns.tolist(),
                "problem_type": problem_type,
            },
            os.path.join(CONFIG.model_dir, "metadata.pkl"),
        )

        logger.info("AutoML pipeline completed successfully")
