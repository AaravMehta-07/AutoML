from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
)
import numpy as np


def get_metric_fn(problem_type: str, metric_name: str = None):
    """
    Returns:
    - metric function
    - higher_is_better
    - needs_proba (IMPORTANT)
    """

    if problem_type == "classification":
        if metric_name == "roc_auc":
            return roc_auc_score, True, True
        elif metric_name == "accuracy":
            return accuracy_score, True, False
        else:
            # default = F1
            return f1_score, True, False

    else:
        if metric_name == "rmse":
            return lambda y, p: np.sqrt(mean_squared_error(y, p)), False, False
        else:
            return mean_absolute_error, False, False
