"""
TabNet Trainer — FinShield
Trains a fraud detection TabNet classifier (PyTorch) with full MLflow tracking.

TabNet uses sequential attention to select which features matter most
for each decision step — it's a neural network designed for tabular data.
"""

import numpy as np
import pandas as pd
import mlflow
import torch
from pathlib import Path
from loguru import logger
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)

from src.features.loader import load_raw_csv
from src.features.engineer import FraudFeatureEngineer
from src.mlops.mlflow_utils import setup_mlflow, log_dataset_info

# ------------------------------------------------------------------ #
# Config                                                               #
# ------------------------------------------------------------------ #

MODEL_DIR = Path(__file__).parents[2] / "models"
MODEL_DIR.mkdir(exist_ok=True)

TABNET_PARAMS = {
    # Architecture
    "n_d":              32,       # width of decision step embedding
    "n_a":              32,       # width of attention embedding (usually = n_d)
    "n_steps":          4,        # number of sequential attention steps
    "gamma":            1.3,      # coefficient for feature reusage penalty
    "n_independent":    2,        # independent GLU layers per step
    "n_shared":         2,        # shared GLU layers per step
    # Training
    "momentum":         0.02,     # batch norm momentum
    "mask_type":        "entmax", # "sparsemax" or "entmax" attention
    "optimizer_fn":     torch.optim.Adam,
    "optimizer_params": {"lr": 2e-3, "weight_decay": 1e-5},
    "scheduler_params": {"step_size": 50, "gamma": 0.9},
    "scheduler_fn":     torch.optim.lr_scheduler.StepLR,
    "verbose":          10,       # print every N epochs
}

TRAINING_PARAMS = {
    "max_epochs":       100,
    "patience":         20,       # early stopping patience
    "batch_size":       1024,
    "virtual_batch_size": 128,    # ghost batch norm size
}


# ------------------------------------------------------------------ #
# Train                                                                #
# ------------------------------------------------------------------ #

def train(
    threshold: float = 0.5,
    test_size: float = 0.2,
    register: bool   = False,
) -> dict:
    """
    Full TabNet training pipeline with MLflow tracking.

    Parameters
    ----------
    threshold : Decision threshold for is_fraud classification
    test_size : Train/test split fraction
    register  : Register model to MLflow registry after training

    Returns
    -------
    dict with run_id, metrics, model_path
    """
    setup_mlflow()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # 1. Load + engineer features
    logger.info("Loading and engineering features...")
    df = load_raw_csv()
    engineer = FraudFeatureEngineer()
    df = engineer.fit_transform(df)

    feature_cols = FraudFeatureEngineer.all_features()
    X = df[feature_cols].values.astype(np.float32)
    y = df["class"].values.astype(int)

    # 2. Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    # Validation split from training set (TabNet needs it for early stopping)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
    )
    logger.info(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")

    # 3. Class weights (handle 577:1 imbalance)
    fraud_count = y_train.sum()
    legit_count = len(y_train) - fraud_count
    # TabNet uses weights as sample weights not class weights → expand to per-sample
    sample_weights = np.where(y_train == 1, legit_count / fraud_count, 1.0)

    # 4. MLflow run
    with mlflow.start_run(run_name="tabnet-fraud") as run:
        run_id = run.info.run_id
        logger.info(f"MLflow run: {run_id}")

        all_params = {**TABNET_PARAMS, **TRAINING_PARAMS,
                      "threshold": threshold, "device": device,
                      "n_features": len(feature_cols)}
        # Filter out non-serializable objects (optimizer, scheduler fn)
        loggable_params = {
            k: str(v) if callable(v) else v
            for k, v in all_params.items()
        }
        mlflow.log_params(loggable_params)
        log_dataset_info(
            pd.DataFrame({"class": y_train}),
            pd.DataFrame({"class": y_test}),
        )

        # 5. Build + train model
        # Remove training-specific params from model init
        model_init_params = {k: v for k, v in TABNET_PARAMS.items()}
        model = TabNetClassifier(**model_init_params, device_name=device)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_name=["val"],
            eval_metric=["auc"],
            weights=sample_weights,
            max_epochs=TRAINING_PARAMS["max_epochs"],
            patience=TRAINING_PARAMS["patience"],
            batch_size=TRAINING_PARAMS["batch_size"],
            virtual_batch_size=TRAINING_PARAMS["virtual_batch_size"],
            num_workers=0,
            drop_last=False,
        )

        # 6. Evaluate
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred  = (y_proba >= threshold).astype(int)

        metrics = {
            "auc_roc":         round(roc_auc_score(y_test, y_proba), 5),
            "avg_precision":   round(average_precision_score(y_test, y_proba), 5),
            "f1_fraud":        round(f1_score(y_test, y_pred), 5),
            "precision_fraud": round(precision_score(y_test, y_pred, zero_division=0), 5),
            "recall_fraud":    round(recall_score(y_test, y_pred), 5),
        }
        mlflow.log_metrics(metrics)
        logger.info(f"Metrics: {metrics}")
        print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))

        # 7. Feature importance via attention masks
        feat_importances = dict(zip(feature_cols, model.feature_importances_))
        top5 = sorted(feat_importances.items(), key=lambda x: x[1], reverse=True)[:5]
        top5_dict = {k: round(float(v), 5) for k, v in top5}
        mlflow.log_dict(top5_dict, "tabnet_top5_attention_features.json")
        logger.info(f"Top 5 attention features: {top5_dict}")

        # 8. Save model
        model_path = MODEL_DIR / "tabnet_fraud"
        model.save_model(str(model_path))
        mlflow.log_artifact(str(model_path) + ".zip")
        logger.success(f"TabNet model saved → {model_path}.zip")

        # 9. Optionally register
        if register:
            from src.mlops.mlflow_utils import register_model
            register_model(run_id, model_name="finshield-tabnet", stage="Staging")

    return {"run_id": run_id, "metrics": metrics, "model_path": str(model_path)}


# ------------------------------------------------------------------ #
# CLI entry                                                            #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train TabNet fraud detector")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--register",  action="store_true")
    args = parser.parse_args()

    result = train(
        threshold=args.threshold,
        test_size=args.test_size,
        register=args.register,
    )
    print(f"\n✅ TabNet training complete!")
    print(f"   Run ID   : {result['run_id']}")
    print(f"   AUC-ROC  : {result['metrics']['auc_roc']}")
    print(f"   F1 Fraud : {result['metrics']['f1_fraud']}")
    print(f"   Model    : {result['model_path']}")
