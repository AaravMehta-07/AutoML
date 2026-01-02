import pandas as pd
import numpy as np
from typing import List
from automl.logger import get_logger

logger = get_logger("DatetimeFeatures")


class DatetimeFeatureEngineer:
    def __init__(self, datetime_cols: List[str]):
        self.datetime_cols = datetime_cols

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Starting datetime feature engineering")

        df = df.copy()

        for col in self.datetime_cols:
            if col not in df.columns:
                continue

            # Ensure datetime
            df[col] = pd.to_datetime(df[col], errors="coerce")

            # Basic components
            df[f"{col}_hour"] = df[col].dt.hour
            df[f"{col}_day"] = df[col].dt.day
            df[f"{col}_dayofweek"] = df[col].dt.dayofweek
            df[f"{col}_month"] = df[col].dt.month

            # Cyclical encoding
            df[f"{col}_hour_sin"] = np.sin(2 * np.pi * df[col].dt.hour / 24)
            df[f"{col}_hour_cos"] = np.cos(2 * np.pi * df[col].dt.hour / 24)

            df[f"{col}_dow_sin"] = np.sin(2 * np.pi * df[col].dt.dayofweek / 7)
            df[f"{col}_dow_cos"] = np.cos(2 * np.pi * df[col].dt.dayofweek / 7)

            # Drop original datetime column
            df.drop(columns=[col], inplace=True)

        logger.info("Datetime feature engineering completed")
        return df
