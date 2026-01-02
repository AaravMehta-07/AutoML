import argparse
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
import sys

from automl.logger import get_logger
from automl.features.numeric import NumericFeatureEngineer
from automl.features.datetime import DatetimeFeatureEngineer

logger = get_logger("Explainability")


def parse_args():
    parser = argparse.ArgumentParser(
        description="SHAP explainability for AutoML models"
    )

    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--meta", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--rows", type=int, default=500)

    return parser.parse_args()


def main():
    args = parse_args()
    logger.info("Starting SHAP explainability")

    # -------- Load model & metadata --------
    model = joblib.load(args.model)
    meta = joblib.load(args.meta)

    feature_list = meta["features"]

    # -------- Load raw data --------
    df = pd.read_csv(args.data)
    X = df.copy()

    # -------- Detect feature types --------
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    datetime_cols = []
    for col in X.columns:
        if col not in numeric_cols:
            try:
                pd.to_datetime(X[col])
                datetime_cols.append(col)
            except Exception:
                pass

    # -------- Feature Engineering (same as inference) --------
    if numeric_cols:
        X = NumericFeatureEngineer(numeric_cols).transform(X)

    if datetime_cols:
        X = DatetimeFeatureEngineer(datetime_cols).transform(X)

    X = X[feature_list]

    # -------- Sample rows (SHAP is expensive) --------
    if len(X) > args.rows:
        X = X.sample(args.rows, random_state=42)

    # -------- SHAP Explainer --------
    logger.info("Building SHAP explainer")

    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)

    # -------- Global Importance --------
    logger.info("Generating global feature importance")
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig("shap_global_importance.png", dpi=300)
    plt.close()

    # -------- Local Explanation (first row) --------
    logger.info("Generating local explanation")
    shap.plots.waterfall(shap_values[0], show=False)
    plt.tight_layout()
    plt.savefig("shap_local_explanation.png", dpi=300)
    plt.close()

    logger.info("SHAP explainability completed successfully")
    logger.info("Saved: shap_global_importance.png, shap_local_explanation.png")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("SHAP explainability failed")
        sys.exit(1)
