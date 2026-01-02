from dataclasses import dataclass
from typing import Optional


@dataclass
class AutoMLConfig:
    # General
    random_state: int = 42
    n_jobs: int = -1

    # Data
    test_size: float = 0.2
    cv_folds: int = 5

    # Feature Engineering
    max_polynomial_degree: int = 3
    max_categories_for_onehot: int = 15
    correlation_threshold: float = 0.95

    # Optimization
    max_trials: int = 25
    
    timeout: Optional[int] = None  # seconds

    # Logging
    log_level: str = "INFO"

    # Output paths
    model_dir: str = "reports"
    report_dir: str = "reports"


CONFIG = AutoMLConfig()
