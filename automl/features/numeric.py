import pandas as pd
import numpy as np
from typing import List
from automl.logger import get_logger
from automl.config import CONFIG

logger = get_logger("NumericFeatures")


class NumericFeatureEngineer:
    def __init__(self, numeric_cols: List[str]):
        self.numeric_cols = numeric_cols

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Starting numeric feature engineering")

        df = df.copy()

        for col in self.numeric_cols:
            if col not in df.columns:
                continue

            series = df[col]

            # Log transform (safe)
            if (series > 0).all():
                df[f"{col}_log"] = np.log(series + 1e-9)

            # Square root transform
            if (series >= 0).all():
                df[f"{col}_sqrt"] = np.sqrt(series)

            # Polynomial features (controlled)
            for degree in range(2, CONFIG.max_polynomial_degree + 1):
                df[f"{col}_pow_{degree}"] = series ** degree

            # Z-score normalization (as feature)
            mean = series.mean()
            std = series.std()
            if std > 0:
                df[f"{col}_zscore"] = (series - mean) / std

        logger.info("Numeric feature engineering completed")
        return df
