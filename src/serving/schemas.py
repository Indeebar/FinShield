"""
Pydantic Schemas — FinShield Serving Layer
Request and response models for all API endpoints.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional


# ------------------------------------------------------------------ #
# Request                                                              #
# ------------------------------------------------------------------ #

class TransactionRequest(BaseModel):
    """
    Input transaction for fraud scoring.
    Mirrors the creditcard.csv feature structure.
    """
    # Core fields
    time:   float = Field(..., description="Seconds elapsed since first transaction in dataset")
    amount: float = Field(..., gt=0, description="Transaction amount (must be > 0)")

    # PCA features V1–V28 (output of PCA on original bank features — anonymised)
    v1:  float = 0.0; v2:  float = 0.0; v3:  float = 0.0; v4:  float = 0.0
    v5:  float = 0.0; v6:  float = 0.0; v7:  float = 0.0; v8:  float = 0.0
    v9:  float = 0.0; v10: float = 0.0; v11: float = 0.0; v12: float = 0.0
    v13: float = 0.0; v14: float = 0.0; v15: float = 0.0; v16: float = 0.0
    v17: float = 0.0; v18: float = 0.0; v19: float = 0.0; v20: float = 0.0
    v21: float = 0.0; v22: float = 0.0; v23: float = 0.0; v24: float = 0.0
    v25: float = 0.0; v26: float = 0.0; v27: float = 0.0; v28: float = 0.0

    @field_validator("amount")
    @classmethod
    def amount_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("amount must be greater than 0")
        return round(v, 2)

    model_config = {
        "json_schema_extra": {
            "example": {
                "time": 86400.0,
                "amount": 1500.0,
                "v1": -1.36,  "v2":  0.97,  "v3":  0.23,
                "v4":  0.0,   "v14": -3.2,
            }
        }
    }


# ------------------------------------------------------------------ #
# Response                                                             #
# ------------------------------------------------------------------ #

class PredictionResponse(BaseModel):
    """Full fraud scoring response including explanation."""
    transaction_id:    str
    fraud_score:       float = Field(..., ge=0.0, le=1.0, description="Model probability (0=legit, 1=fraud)")
    is_fraud:          bool  = Field(..., description="True if score > threshold")
    threshold_used:    float
    model_name:        str   = Field(..., description="'xgboost' or 'tabnet'")
    shap_top_features: dict  = Field(..., description="Top 5 SHAP feature importances")
    explanation:       str   = Field(..., description="RAG-generated natural language explanation")
    similar_cases:     list  = Field(..., description="Top matching historical fraud cases")
    latency_ms:        float = Field(..., description="Total inference latency in ms")

    model_config = {"json_schema_extra": {
        "example": {
            "transaction_id": "txn_abc123",
            "fraud_score":    0.87,
            "is_fraud":       True,
            "threshold_used": 0.5,
            "model_name":     "xgboost",
            "shap_top_features": {"v14": -0.83, "amount_vs_mean_ratio": 0.62},
            "explanation":    "🚨 FRAUD DETECTED (score: 0.8700)...",
            "similar_cases":  [],
            "latency_ms":     42.3,
        }
    }}


class ExplainResponse(BaseModel):
    """Response for the /explain/{txn_id} endpoint."""
    transaction_id: str
    fraud_score:    float
    is_fraud:       bool
    explanation:    str
    similar_cases:  list
    query_used:     str


class HealthResponse(BaseModel):
    """API health check response."""
    status:        str    = Field(..., description="'healthy' or 'degraded'")
    model_loaded:  bool
    model_name:    str
    chroma_docs:   int    = Field(..., description="Number of docs in ChromaDB index")
    version:       str    = "1.0.0"
