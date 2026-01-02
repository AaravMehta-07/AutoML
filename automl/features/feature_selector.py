import pandas as pd
import numpy as np
from typing import List
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from automl.logger import get_logger
from automl.config import CONFIG

logger = get_logger("FeatureSelector")


class FeatureSelector:
    def __init__(
        self,
        problem_type: str,
        correlation_threshold: float = CONFIG.correlation_threshold,
        max_features: int = None,
    ):
        self.problem_type = problem_type
        self.correlation_threshold = correlation_threshold
        self.max_features = max_features

    def _drop_constant_features(self, X: pd.DataFrame) -> pd.DataFrame:
        nunique = X.nunique()
        constant_cols = nunique[nunique <= 1].index.tolist()
        if constant_cols:
            logger.info(f"Dropping constant features: {len(constant_cols)}")
            X = X.drop(columns=constant_cols)
        return X

    def _correlation_pruning(self, X: pd.DataFrame) -> pd.DataFrame:
        corr = X.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

        to_drop = [
            col for col in upper.columns
            if any(upper[col] > self.correlation_threshold)
        ]

        if to_drop:
            logger.info(f"Correlation pruning dropped {len(to_drop)} features")

        return X.drop(columns=to_drop)

    def _mutual_information(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        if self.problem_type == "classification":
            mi = mutual_info_classif(X, y, random_state=CONFIG.random_state)
        else:
            mi = mutual_info_regression(X, y, random_state=CONFIG.random_state)

        return pd.Series(mi, index=X.columns).sort_values(ascending=False)

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        logger.info("Starting feature selection")

        X = self._drop_constant_features(X)
        X = self._correlation_pruning(X)

        mi_scores = self._mutual_information(X, y)

        if self.max_features is not None:
            selected = mi_scores.head(self.max_features).index.tolist()
            logger.info(f"Selecting top {len(selected)} features by MI")
            X = X[selected]
        else:
            # Drop zero-information features
            selected = mi_scores[mi_scores > 0].index.tolist()
            logger.info(f"Selecting {len(selected)} informative features")
            X = X[selected]

        logger.info("Feature selection completed")
        return X
