import argparse
import pandas as pd
import joblib
import sys

from automl.logger import get_logger
from automl.features.numeric import NumericFeatureEngineer
from automl.features.datetime import DatetimeFeatureEngineer

logger = get_logger("Inference")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference using trained AutoML model"
    )

    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--meta", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--output", type=str, default="predictions.csv")

    return parser.parse_args()


def main():
    args = parse_args()
    logger.info("Starting inference")

    # -------- Load model & metadata --------
    try:
        model = joblib.load(args.model)
        metadata = joblib.load(args.meta)
    except Exception as e:
        logger.error("Failed to load model or metadata")
        raise e

    feature_list = metadata["features"]
    problem_type = metadata["problem_type"]

    # -------- Load raw data --------
    try:
        df = pd.read_csv(args.data)
    except Exception as e:
        logger.error("Failed to read input data")
        raise e

    X = df.copy()

    # -------- Detect feature types (NO PROFILER) --------
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    datetime_cols = []
    for col in X.columns:
        if col not in numeric_cols:
            try:
                pd.to_datetime(X[col])
                datetime_cols.append(col)
            except Exception:
                pass

    # -------- Feature Engineering (same logic as training) --------
    if numeric_cols:
        num_engineer = NumericFeatureEngineer(numeric_cols)
        X = num_engineer.transform(X)

    if datetime_cols:
        dt_engineer = DatetimeFeatureEngineer(datetime_cols)
        X = dt_engineer.transform(X)

    # -------- Feature alignment --------
    missing = set(feature_list) - set(X.columns)
    if missing:
        logger.error(f"Missing required features after engineering: {missing}")
        sys.exit(1)

    X = X[feature_list]

    # -------- Prediction --------
    if problem_type == "classification":
        if hasattr(model, "predict_proba"):
            preds = model.predict_proba(X)[:, 1]
        else:
            preds = model.predict(X)
    else:
        preds = model.predict(X)

    # -------- Save output --------
    out = df.copy()
    out["prediction"] = preds
    out.to_csv(args.output, index=False)

    logger.info(f"Inference complete. Predictions saved to {args.output}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Inference failed")
        sys.exit(1)
