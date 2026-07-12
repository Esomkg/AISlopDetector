"""Shared utilities for AISlopDetector Airflow DAGs."""

import sys
import os
from pathlib import Path

AIRFLOW_HOME = Path("/opt/airflow")
SRC_PATH = AIRFLOW_HOME / "src"
CONFIGS_PATH = AIRFLOW_HOME / "configs"
DATA_PATH = AIRFLOW_HOME / "data"

sys.path.insert(0, str(SRC_PATH.parent))

DEFAULT_ARGS = {
    "owner": "aislop",
    "retries": 1,
    "retry_delay": 5 * 60,
    "email_on_failure": False,
}

DRIFT_THRESHOLD = 0.05
LOW_CONFIDENCE_THRESHOLD = 0.6

def ensure_dirs():
    for d in ["data/raw", "data/embeddings", "checkpoints", "metrics"]:
        Path(d).mkdir(parents=True, exist_ok=True)
