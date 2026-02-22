"""
Pydantic Schemas — FinShield Serving Layer
Request/response models for all API endpoints.
"""

from pydantic import BaseModel, Field, field_validator


class TransactionInput(BaseModel):
    """Raw transaction payload sent to POST /predict."""

    # Time field (seconds offset, as in Kaggle dataset)
    time: float = Field(..., description="Seconds elapsed since first transaction in dataset")

    # Amount
    amount: float = Field(..., ge=0.0, description="Transaction amount in USD (must be >= 0)")

    # PCA-reduced features V1–V28 (already anonymised by Kaggle)
    v1:  float; v2:  float; v3:  float; v4:  float
    v5:  float; v6:  float; v7:  float; v8:  float
    v9:  float; v10: float; v11: float; v12: float
    v13: float; v14: float; v15: float; v16: float
    v17: float; v18: float; v19: float; v20: float
    v21: float; v22: float; v23: float; v24: float
    v25: float; v26: float; v27: float; v28: float

    def to_raw_dict(self) -> dict:
        """Return a flat dict with column names matching the training DataFrame."""
        d = self.model_dump()
        # Rename 'vN' keys to 'vN' — already lowercase, matches engineer expectations
        return d


class PredictResponse(BaseModel):
    """Response from POST /predict."""

    txn_id:            str
    fraud_score:       float = Field(..., ge=0.0, le=1.0)
    is_fraud:          bool
    shap_top_features: dict
    explanation:       str


class ExplainResponse(BaseModel):
    """Response from GET /explain/{txn_id}."""

    txn_id:            str
    fraud_score:       float = Field(..., ge=0.0, le=1.0)
    is_fraud:          bool
    shap_top_features: dict
    explanation:       str


class HealthResponse(BaseModel):
    """Response from GET /health."""

    status:        str
    model_loaded:  bool
    chroma_ok:     bool
    db_ok:         bool
