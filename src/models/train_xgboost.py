"""
XGBoost Trainer — FinShield
Trains a fraud detection XGBoost classifier with full MLflow tracking.
"""

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import mlflow
import mlflow.xgboost
import shap
from pathlib import Path
from loguru import logger
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.features.loader import load_raw_csv
from src.features.engineer import FraudFeatureEngineer
from src.mlops.mlflow_utils import setup_mlflow, log_dataset_info

# ------------------------------------------------------------------ #
# Config                                                               #
# ------------------------------------------------------------------ #

MODEL_DIR   = Path(__file__).parents[2] / "models"
MODEL_DIR.mkdir(exist_ok=True)

XGBOOST_PARAMS = {
    "n_estimators":     500,
    "max_depth":        6,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "random_state":     42,
    "eval_metric":      "aucpr",   # area under PR curve — better for imbalanced
    "early_stopping_rounds": 30,
    "tree_method":      "hist",    # fast CPU training
    "scale_pos_weight": 577,       # len(non-fraud) / len(fraud) ~ 577 for this dataset
}


# ------------------------------------------------------------------ #
# Train                                                                #
# ------------------------------------------------------------------ #

def train(
    threshold: float = 0.5,
    test_size: float = 0.2,
    register: bool  = False,
) -> dict:
    """
    Full XGBoost training pipeline with MLflow tracking.

    Parameters
    ----------
    threshold : Decision threshold for is_fraud flag (default 0.5)
    test_size : Train/test split ratio
    register  : Whether to register model to MLflow registry after training

    Returns
    -------
    dict with run_id, metrics, and model path
    """
    setup_mlflow()

    # 1. Load + engineer features
    logger.info("Loading data...")
    df = load_raw_csv()

    engineer = FraudFeatureEngineer()
    df = engineer.fit_transform(df)

    feature_cols = FraudFeatureEngineer.all_features()
    X = df[feature_cols].values
    y = df["class"].values

    # 2. Train/test split — stratified to preserve fraud ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=42,
        stratify=y,
    )
    logger.info(f"Train: {X_train.shape} | Test: {X_test.shape}")

    # 3. MLflow run
    with mlflow.start_run(run_name="xgboost-fraud") as run:
        run_id = run.info.run_id
        logger.info(f"MLflow run: {run_id}")

        # Log params
        mlflow.log_params({**XGBOOST_PARAMS, "threshold": threshold, "test_size": test_size})
        log_dataset_info(
            pd.DataFrame({"class": y_train}),
            pd.DataFrame({"class": y_test}),
        )
        mlflow.log_params({"n_features": len(feature_cols)})

        # 4. Train model
        model = xgb.XGBClassifier(**XGBOOST_PARAMS)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=50,
        )

        # 5. Evaluate
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred  = (y_proba >= threshold).astype(int)

        metrics = {
            "auc_roc":          round(roc_auc_score(y_test, y_proba), 5),
            "avg_precision":    round(average_precision_score(y_test, y_proba), 5),
            "f1_fraud":         round(f1_score(y_test, y_pred), 5),
            "precision_fraud":  round(precision_score(y_test, y_pred), 5),
            "recall_fraud":     round(recall_score(y_test, y_pred), 5),
        }
        mlflow.log_metrics(metrics)
        logger.info(f"Metrics: {metrics}")

        # Print classification report
        print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))

        # 6. SHAP values (top feature importance)
        logger.info("Computing SHAP values...")
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test[:500])  # sample for speed
        mean_shap   = np.abs(shap_values).mean(axis=0)
        top5_idx    = mean_shap.argsort()[::-1][:5]
        top5_features = {feature_cols[i]: round(float(mean_shap[i]), 5) for i in top5_idx}
        mlflow.log_dict(top5_features, "shap_top5_features.json")
        logger.info(f"Top 5 SHAP features: {top5_features}")

        # 7. Save artifacts
        model_path = MODEL_DIR / "xgboost_fraud.json"
        model.save_model(str(model_path))

        engineer_path = MODEL_DIR / "feature_engineer.pkl"
        joblib.dump(engineer, str(engineer_path))

        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(engineer_path))
        mlflow.xgboost.log_model(model, artifact_path="xgboost-model")

        logger.success(f"Model saved → {model_path}")

        # 8. Optionally register to Model Registry
        if register:
            from src.mlops.mlflow_utils import register_model
            register_model(run_id, model_name="finshield-xgboost", stage="Staging")

    return {"run_id": run_id, "metrics": metrics, "model_path": str(model_path)}


# ------------------------------------------------------------------ #
# CLI entry point                                                       #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train XGBoost fraud detector")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--register",  action="store_true")
    args = parser.parse_args()

    result = train(
        threshold=args.threshold,
        test_size=args.test_size,
        register=args.register,
    )
    print(f"\n✅ Training complete!")
    print(f"   Run ID   : {result['run_id']}")
    print(f"   AUC-ROC  : {result['metrics']['auc_roc']}")
    print(f"   F1 Fraud : {result['metrics']['f1_fraud']}")
    print(f"   Model    : {result['model_path']}")
