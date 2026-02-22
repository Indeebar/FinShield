"""
FastAPI Application — FinShield
Three endpoints: POST /predict, GET /explain/{txn_id}, GET /health
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from contextlib import asynccontextmanager

from src.serving.schemas import (
    TransactionRequest,
    PredictionResponse,
    ExplainResponse,
    HealthResponse,
)
from src.serving.predictor import predict as run_predict, load_model, load_engineer

# In-memory store for recent predictions (txn_id → PredictionResponse)
# In production this would be PostgreSQL — sufficient for demo
_prediction_store: dict[str, PredictionResponse] = {}


# ------------------------------------------------------------------ #
# Lifespan (startup/shutdown)                                          #
# ------------------------------------------------------------------ #

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load model and engineer on startup to avoid cold-start latency."""
    logger.info("🚀 FinShield API starting up...")
    load_model()      # warm up model cache
    load_engineer()   # warm up engineer cache

    # Pre-build ChromaDB index in background
    try:
        from src.rag.retriever import get_collection
        col = get_collection(persist=True)
        logger.info(f"ChromaDB ready — {col.count()} fraud cases indexed")
    except Exception as e:
        logger.warning(f"ChromaDB startup warning: {e}")

    yield   # --- app is running ---

    logger.info("FinShield API shutting down.")


# ------------------------------------------------------------------ #
# App                                                                  #
# ------------------------------------------------------------------ #

app = FastAPI(
    title       = "FinShield — Fraud Detection API",
    description = "Real-time fraud scoring with XGBoost/TabNet + RAG explainability",
    version     = "1.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)


# ------------------------------------------------------------------ #
# Routes                                                               #
# ------------------------------------------------------------------ #

@app.post(
    "/predict",
    response_model = PredictionResponse,
    summary        = "Score a transaction for fraud",
    tags           = ["Inference"],
)
async def predict(request: TransactionRequest) -> PredictionResponse:
    """
    Score a single financial transaction for fraud.

    - Runs feature engineering on the raw input
    - Runs XGBoost inference → fraud probability score
    - Computes SHAP top-5 feature impacts
    - Retrieves similar historical fraud cases via RAG
    - Returns structured fraud analysis
    """
    try:
        result = run_predict(request)
        _prediction_store[result.transaction_id] = result
        return result
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get(
    "/explain/{txn_id}",
    response_model = ExplainResponse,
    summary        = "Get explanation for a scored transaction",
    tags           = ["Explainability"],
)
async def explain(txn_id: str) -> ExplainResponse:
    """
    Retrieve the fraud explanation for a previously scored transaction.

    Uses the cached prediction result and returns the RAG explanation,
    similar historical cases, and the ChromaDB query that was used.
    """
    prediction = _prediction_store.get(txn_id)
    if not prediction:
        raise HTTPException(
            status_code = 404,
            detail      = f"No prediction found for transaction '{txn_id}'. "
                          f"Call POST /predict first.",
        )

    return ExplainResponse(
        transaction_id = txn_id,
        fraud_score    = prediction.fraud_score,
        is_fraud       = prediction.is_fraud,
        explanation    = prediction.explanation,
        similar_cases  = prediction.similar_cases,
        query_used     = f"fraud_score={prediction.fraud_score:.4f} top_features={list(prediction.shap_top_features.keys())[:3]}",
    )


@app.get(
    "/health",
    response_model = HealthResponse,
    summary        = "API health check",
    tags           = ["System"],
)
async def health() -> HealthResponse:
    """
    Check the health of all system components:
    - Model loading status
    - ChromaDB index availability
    """
    model       = load_model()
    model_ok    = model is not None
    model_name  = "xgboost" if model_ok else "none"

    chroma_count = 0
    try:
        from src.rag.retriever import get_collection
        col          = get_collection(persist=True)
        chroma_count = col.count()
    except Exception as e:
        logger.warning(f"ChromaDB health check failed: {e}")

    status = "healthy" if (model_ok and chroma_count > 0) else "degraded"

    return HealthResponse(
        status       = status,
        model_loaded = model_ok,
        model_name   = model_name,
        chroma_docs  = chroma_count,
    )


@app.get("/", include_in_schema=False)
async def root():
    return {
        "project": "FinShield",
        "docs":    "/docs",
        "health":  "/health",
    }
