import argparse
import sys
from automl.logger import get_logger

logger = get_logger("AutoML")


def parse_args():
    parser = argparse.ArgumentParser(
        description="AutoML System Built From Scratch"
    )

    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to CSV dataset"
    )

    parser.add_argument(
        "--target",
        type=str,
        required=True,
        help="Target column name"
    )

    parser.add_argument(
        "--metric",
        type=str,
        default=None,
        help="Optimization metric (auto-selected if not provided)"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("Starting AutoML system")

    # ---------------- Load Dataset (existing logic preserved) ----------------
    try:
        import pandas as pd
        df = pd.read_csv(args.data)
    except Exception as e:
        logger.error("Failed to read dataset")
        raise e

    # ---------------- Phase 2: Profiling (existing logic preserved) ----------------
    from automl.profiler.data_profiler import DataProfiler

    profiler = DataProfiler(df, args.target)
    metadata = profiler.profile()

    logger.info("Dataset metadata summary:")
    for k, v in metadata.items():
        if k not in ("numeric_stats", "categorical_stats"):
            logger.info(f"{k}: {v}")

    logger.info("Phase 2 complete. Profiling successful.")

    # ---------------- Phase 7: Full AutoML Pipeline (NEW, added safely) ----------------
    logger.info("Starting full AutoML pipeline")

    from automl.pipeline import AutoMLPipeline

    pipeline = AutoMLPipeline(
        data_path=args.data,
        target=args.target,
        metric=args.metric,
    )

    pipeline.run()

    logger.info("AutoML run completed successfully")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("Fatal error occurred")
        sys.exit(1)
