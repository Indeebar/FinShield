"""
MLflow Utilities — FinShield
Centralised helpers for experiment tracking and model registry.
"""

import os
import mlflow
import mlflow.sklearn
import mlflow.pytorch
from loguru import logger
from pathlib import Path

# ------------------------------------------------------------------ #
# Config                                                               #
# ------------------------------------------------------------------ #

from pathlib import Path

MLFLOW_TRACKING_DIR = Path(__file__).parents[2] / "mlruns"
# Default to local file-based tracking (no server needed for training).
# Override with env var MLFLOW_TRACKING_URI=http://localhost:5000 when running the UI server.
MLFLOW_URI      = os.getenv("MLFLOW_TRACKING_URI", f"file:///{MLFLOW_TRACKING_DIR.as_posix()}")
EXPERIMENT_NAME = "finshield-fraud-detection"
ARTIFACT_ROOT   = MLFLOW_TRACKING_DIR


def setup_mlflow() -> None:
    """Set tracking URI and create experiment if it doesn't exist."""
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    logger.info(f"MLflow → {MLFLOW_URI} | experiment: {EXPERIMENT_NAME}")


def log_dataset_info(df_train, df_test) -> None:
    """Log dataset size and fraud rate to the active MLflow run."""
    mlflow.log_params({
        "n_train":          len(df_train),
        "n_test":           len(df_test),
        "train_fraud_rate": round(df_train["class"].mean(), 5),
        "test_fraud_rate":  round(df_test["class"].mean(), 5),
    })


def register_model(run_id: str, model_name: str, stage: str = "Staging") -> None:
    """
    Register the model from a given run into MLflow Model Registry.

    Parameters
    ----------
    run_id     : MLflow run ID returned from the training run
    model_name : Registry name e.g. 'finshield-xgboost'
    stage      : 'Staging' or 'Production'
    """
    model_uri = f"runs:/{run_id}/{model_name}"
    mv = mlflow.register_model(model_uri=model_uri, name=model_name)
    logger.info(f"Registered {model_name} v{mv.version} → {stage}")

    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=mv.version,
        stage=stage,
    )


def load_production_model(model_name: str):
    """Load the latest Production-stage model from the registry."""
    model_uri = f"models:/{model_name}/Production"
    logger.info(f"Loading production model: {model_uri}")
    return mlflow.sklearn.load_model(model_uri)
