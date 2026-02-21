"""
Model Comparison — FinShield
Fetches XGBoost and TabNet runs from MLflow and picks the best model
based on avg_precision (area under PR curve — ideal for imbalanced fraud data).
Promotes the winner to Production in MLflow Model Registry.
"""

import mlflow
import pandas as pd
from loguru import logger
from src.mlops.mlflow_utils import setup_mlflow, register_model

# The primary metric to decide the winner (higher = better)
COMPARISON_METRIC = "avg_precision"


def get_best_run(experiment_name: str = "finshield-fraud-detection") -> pd.Series:
    """
    Query MLflow for all runs in the experiment and return the best row.

    Returns
    -------
    pd.Series with run_id, metrics, params of the best run
    """
    setup_mlflow()
    client  = mlflow.tracking.MlflowClient()
    exp     = client.get_experiment_by_name(experiment_name)

    if exp is None:
        raise RuntimeError(f"Experiment '{experiment_name}' not found. Run training first.")

    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=[f"metrics.{COMPARISON_METRIC} DESC"],
    )

    if runs.empty:
        raise RuntimeError("No completed runs found. Run training scripts first.")

    best = runs.iloc[0]
    logger.info(
        f"Best run: {best['run_id'][:8]}... "
        f"| {COMPARISON_METRIC}={best[f'metrics.{COMPARISON_METRIC}']:.5f}"
    )
    return best


def compare_and_promote(register: bool = True) -> dict:
    """
    Compare all training runs, print a leaderboard, and optionally
    promote the best model to Production in MLflow registry.

    Returns
    -------
    dict: winner run_id, model_name, metrics
    """
    setup_mlflow()
    client  = mlflow.tracking.MlflowClient()
    exp     = client.get_experiment_by_name("finshield-fraud-detection")

    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=[f"metrics.{COMPARISON_METRIC} DESC"],
    )

    # Leaderboard columns
    cols = [
        "tags.mlflow.runName",
        "run_id",
        "metrics.auc_roc",
        "metrics.avg_precision",
        "metrics.f1_fraud",
        "metrics.precision_fraud",
        "metrics.recall_fraud",
    ]
    available = [c for c in cols if c in runs.columns]
    leaderboard = runs[available].copy()
    leaderboard.columns = [c.replace("metrics.", "").replace("tags.mlflow.", "")
                           for c in leaderboard.columns]
    leaderboard = leaderboard.round(5)

    print("\n" + "═" * 75)
    print("  🏆  FinShield Model Leaderboard")
    print("═" * 75)
    print(leaderboard.to_string(index=False))
    print("═" * 75)

    best      = runs.iloc[0]
    run_id    = best["run_id"]
    run_name  = best.get("tags.mlflow.runName", "unknown")

    # Infer model registry name from run name
    if "xgboost" in run_name:
        model_name = "finshield-xgboost"
    elif "tabnet" in run_name:
        model_name = "finshield-tabnet"
    else:
        model_name = "finshield-model"

    winner_metrics = {
        "auc_roc":       round(best.get("metrics.auc_roc", 0), 5),
        "avg_precision": round(best.get("metrics.avg_precision", 0), 5),
        "f1_fraud":      round(best.get("metrics.f1_fraud", 0), 5),
    }

    print(f"\n✅ Winner: {run_name}  (run {run_id[:8]}...)")
    print(f"   avg_precision = {winner_metrics['avg_precision']}")
    print(f"   f1_fraud      = {winner_metrics['f1_fraud']}")

    if register:
        logger.info(f"Promoting {model_name} → Production")
        register_model(run_id, model_name=model_name, stage="Production")
        print(f"   Registered as {model_name} → Production 🚀")

    return {"run_id": run_id, "model_name": model_name, "metrics": winner_metrics}


if __name__ == "__main__":
    compare_and_promote(register=False)   # set True after both models are trained
