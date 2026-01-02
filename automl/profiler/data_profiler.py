import pandas as pd
import numpy as np
from typing import Dict, List
from sklearn.utils.multiclass import type_of_target
from automl.logger import get_logger

logger = get_logger("DataProfiler")


class DataProfiler:
    def __init__(self, df: pd.DataFrame, target_col: str):
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset")

        self.df = df.copy()
        self.target_col = target_col

    def _detect_problem_type(self) -> str:
        y = self.df[self.target_col]
        target_type = type_of_target(y)

        if target_type in ("binary", "multiclass"):
            return "classification"
        elif target_type in ("continuous",):
            return "regression"
        else:
            raise ValueError(f"Unsupported target type: {target_type}")

    def _detect_feature_types(self) -> Dict[str, List[str]]:
        feature_cols = [c for c in self.df.columns if c != self.target_col]

        numeric_cols = []
        categorical_cols = []
        datetime_cols = []

        for col in feature_cols:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                numeric_cols.append(col)
            elif pd.api.types.is_datetime64_any_dtype(self.df[col]):
                datetime_cols.append(col)
            else:
                # Try parsing datetime
                try:
                    parsed = pd.to_datetime(self.df[col], errors="raise")
                    self.df[col] = parsed
                    datetime_cols.append(col)
                except Exception:
                    categorical_cols.append(col)

        return {
            "numeric": numeric_cols,
            "categorical": categorical_cols,
            "datetime": datetime_cols,
        }

    def _missing_value_report(self) -> Dict[str, float]:
        return (
            self.df.isnull()
            .mean()
            .sort_values(ascending=False)
            .to_dict()
        )

    def _numeric_stats(self, numeric_cols: List[str]) -> Dict[str, Dict]:
        stats = {}
        for col in numeric_cols:
            series = self.df[col].dropna()
            stats[col] = {
                "mean": float(series.mean()),
                "std": float(series.std()),
                "skew": float(series.skew()),
                "kurtosis": float(series.kurtosis()),
            }
        return stats

    def _categorical_stats(self, categorical_cols: List[str]) -> Dict[str, Dict]:
        stats = {}
        for col in categorical_cols:
            stats[col] = {
                "unique_values": int(self.df[col].nunique()),
                "top_frequency": float(
                    self.df[col].value_counts(normalize=True).iloc[0]
                ) if self.df[col].notnull().any() else 0.0
            }
        return stats

    def profile(self) -> Dict:
        logger.info("Starting data profiling")

        problem_type = self._detect_problem_type()
        feature_types = self._detect_feature_types()
        missing_report = self._missing_value_report()

        numeric_stats = self._numeric_stats(feature_types["numeric"])
        categorical_stats = self._categorical_stats(feature_types["categorical"])

        metadata = {
            "problem_type": problem_type,
            "n_rows": int(self.df.shape[0]),
            "n_features": int(self.df.shape[1] - 1),
            "target": self.target_col,
            "feature_types": feature_types,
            "missing_values": missing_report,
            "numeric_stats": numeric_stats,
            "categorical_stats": categorical_stats,
        }

        logger.info("Data profiling completed")
        logger.info(
            f"Detected problem type: {problem_type}, "
            f"Features -> Numeric: {len(feature_types['numeric'])}, "
            f"Categorical: {len(feature_types['categorical'])}, "
            f"Datetime: {len(feature_types['datetime'])}"
        )

        return metadata
