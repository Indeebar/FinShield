"""
Model Predictor — FinShield Serving Layer
Handles model loading and inference. Decoupled from FastAPI routes
so it can be unit-tested and swapped independently.
"""

import time
import uuid
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
from pathlib import Path
from loguru import logger
from functools import lru_cache

from src.features.engineer import FraudFeatureEngineer
from src.serving.schemas import TransactionRequest, PredictionResponse
from src.rag.explainer import explain as rag_explain

MODEL_DIR   = Path(__file__).parents[2] / "models"
THRESHOLD   = float(0.5)
MODEL_NAME  = "xgboost"      # "xgboost" or "tabnet"


# ------------------------------------------------------------------ #
# Model Loading (cached — loaded once at startup)                      #
# ------------------------------------------------------------------ #

@lru_cache(maxsize=1)
def load_model() -> xgb.XGBClassifier:
    """
    Load the XGBoost model from local models/ directory.
    Falls back gracefully if not found (returns None — health check reflects this).
    """
    model_path = MODEL_DIR / "xgboost_fraud.json"
    if not model_path.exists():
        logger.warning(f"Model not found at {model_path}. Run training first.")
        return None

    model = xgb.XGBClassifier()
    model.load_model(str(model_path))
    logger.success(f"XGBoost model loaded from {model_path}")
    return model


@lru_cache(maxsize=1)
def load_engineer() -> FraudFeatureEngineer:
    """
    Load the fitted FraudFeatureEngineer from disk.
    If not found, return an unfitted engineer (uses in-batch stats — less accurate).
    """
    eng_path = MODEL_DIR / "feature_engineer.pkl"
    if eng_path.exists():
        eng = joblib.load(str(eng_path))
        logger.success(f"Feature engineer loaded from {eng_path}")
        return eng

    logger.warning("feature_engineer.pkl not found — using fresh (unfitted) engineer.")
    return FraudFeatureEngineer()


# ------------------------------------------------------------------ #
# Inference                                                            #
# ------------------------------------------------------------------ #

def _request_to_dataframe(req: TransactionRequest) -> pd.DataFrame:
    """Convert a TransactionRequest into a single-row DataFrame."""
    row = {
        "time":   req.time,
        "amount": req.amount,
    }
    for i in range(1, 29):
        row[f"v{i}"] = getattr(req, f"v{i}")
    return pd.DataFrame([row])


def predict(req: TransactionRequest) -> PredictionResponse:
    """
    Full inference pipeline for a single transaction.

    Steps
    -----
    1. Convert request → DataFrame
    2. Feature engineering
    3. Model inference → fraud_score
    4. SHAP top-5 features
    5. RAG explanation
    6. Return PredictionResponse

    Returns
    -------
    PredictionResponse with full fraud analysis
    """
    t_start = time.perf_counter()
    txn_id  = f"txn_{uuid.uuid4().hex[:8]}"

    # 1. Raw DataFrame
    df_raw = _request_to_dataframe(req)

    # 2. Feature engineering
    engineer = load_engineer()
    df_eng   = engineer.transform(df_raw)
    feature_cols = FraudFeatureEngineer.all_features()
    X = df_eng[feature_cols].values.astype(np.float32)

    # 3. Model inference
    model = load_model()
    if model is None:
        # Model not trained yet — return placeholder response for dev/testing
        logger.warning("Model not loaded — returning dummy prediction")
        return PredictionResponse(
            transaction_id=txn_id,
            fraud_score=0.0,
            is_fraud=False,
            threshold_used=THRESHOLD,
            model_name="none (not trained)",
            shap_top_features={},
            explanation="⚠️ Model not trained yet. Run src/models/train_xgboost.py first.",
            similar_cases=[],
            latency_ms=round((time.perf_counter() - t_start) * 1000, 2),
        )

    fraud_score = float(model.predict_proba(X)[0, 1])
    is_fraud    = fraud_score >= THRESHOLD

    # 4. SHAP top-5 features
    try:
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)[0]   # shape: (n_features,)
        top5_idx    = np.abs(shap_values).argsort()[::-1][:5]
        shap_top    = {feature_cols[i]: round(float(shap_values[i]), 5) for i in top5_idx}
    except Exception as e:
        logger.warning(f"SHAP failed: {e}")
        shap_top = {}

    # 5. RAG explanation
    try:
        rag_result = rag_explain(
            fraud_score       = fraud_score,
            is_fraud          = is_fraud,
            shap_top_features = shap_top,
            amount            = req.amount,
            is_night          = bool(df_eng["is_night"].iloc[0]),
            n_cases           = 2,
            persist_chroma    = True,
        )
        explanation    = rag_result.explanation
        similar_cases  = rag_result.similar_cases
    except Exception as e:
        logger.warning(f"RAG failed: {e}")
        explanation   = f"Fraud score: {fraud_score:.4f}. Explanation unavailable."
        similar_cases = []

    latency_ms = round((time.perf_counter() - t_start) * 1000, 2)
    logger.info(f"[{txn_id}] score={fraud_score:.4f} is_fraud={is_fraud} latency={latency_ms}ms")

    return PredictionResponse(
        transaction_id    = txn_id,
        fraud_score       = round(fraud_score, 6),
        is_fraud          = is_fraud,
        threshold_used    = THRESHOLD,
        model_name        = MODEL_NAME,
        shap_top_features = shap_top,
        explanation       = explanation,
        similar_cases     = similar_cases,
        latency_ms        = latency_ms,
    )
