"""
Integration tests for FastAPI serving layer.
Uses httpx TestClient — no real model needed (graceful fallback tested).
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from src.serving.app import app

client = TestClient(app)


# ------------------------------------------------------------------ #
# /health                                                              #
# ------------------------------------------------------------------ #

class TestHealthEndpoint:

    def test_health_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_schema(self):
        response = client.get("/health")
        data = response.json()
        assert "status"       in data
        assert "model_loaded" in data
        assert "model_name"   in data
        assert "chroma_docs"  in data
        assert "version"      in data

    def test_health_status_is_string(self):
        response = client.get("/health")
        data = response.json()
        assert data["status"] in ("healthy", "degraded")

    def test_root_returns_200(self):
        response = client.get("/")
        assert response.status_code == 200


# ------------------------------------------------------------------ #
# /predict                                                             #
# ------------------------------------------------------------------ #

class TestPredictEndpoint:

    def _sample_payload(self, amount=150.0):
        return {
            "time":   86400.0,
            "amount": amount,
            "v1": -1.36, "v2": 0.97, "v3": 0.23, "v4": 0.0,
            "v5": 0.0,   "v6": 0.0,  "v7": 0.0,  "v8": 0.0,
            "v9": 0.0,   "v10": 0.0, "v11": 0.0, "v12": 0.0,
            "v13": 0.0,  "v14": -3.2,"v15": 0.0, "v16": 0.0,
            "v17": 0.0,  "v18": 0.0, "v19": 0.0, "v20": 0.0,
            "v21": 0.0,  "v22": 0.0, "v23": 0.0, "v24": 0.0,
            "v25": 0.0,  "v26": 0.0, "v27": 0.0, "v28": 0.0,
        }

    def test_predict_returns_200(self):
        response = client.post("/predict", json=self._sample_payload())
        assert response.status_code == 200

    def test_predict_response_has_required_fields(self):
        response = client.post("/predict", json=self._sample_payload())
        data = response.json()
        assert "transaction_id"    in data
        assert "fraud_score"       in data
        assert "is_fraud"          in data
        assert "threshold_used"    in data
        assert "model_name"        in data
        assert "shap_top_features" in data
        assert "explanation"       in data
        assert "similar_cases"     in data
        assert "latency_ms"        in data

    def test_fraud_score_between_0_and_1(self):
        response = client.post("/predict", json=self._sample_payload())
        data = response.json()
        assert 0.0 <= data["fraud_score"] <= 1.0

    def test_is_fraud_is_boolean(self):
        response = client.post("/predict", json=self._sample_payload())
        data = response.json()
        assert isinstance(data["is_fraud"], bool)

    def test_transaction_id_returned_is_string(self):
        response = client.post("/predict", json=self._sample_payload())
        assert isinstance(response.json()["transaction_id"], str)

    def test_predict_rejects_negative_amount(self):
        payload = self._sample_payload(amount=-100.0)
        response = client.post("/predict", json=payload)
        assert response.status_code == 422    # Pydantic validation error

    def test_predict_rejects_zero_amount(self):
        payload = self._sample_payload(amount=0.0)
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_predict_missing_required_field_returns_422(self):
        response = client.post("/predict", json={"amount": 100.0})  # missing time
        assert response.status_code == 422


# ------------------------------------------------------------------ #
# /explain/{txn_id}                                                    #
# ------------------------------------------------------------------ #

class TestExplainEndpoint:

    def test_explain_unknown_txn_returns_404(self):
        response = client.get("/explain/txn_doesnotexist")
        assert response.status_code == 404

    def test_explain_after_predict_returns_200(self):
        # First create a prediction
        payload = {
            "time": 86400.0, "amount": 500.0,
            **{f"v{i}": 0.0 for i in range(1, 29)},
        }
        predict_resp = client.post("/predict", json=payload)
        txn_id = predict_resp.json()["transaction_id"]

        # Then explain it
        explain_resp = client.get(f"/explain/{txn_id}")
        assert explain_resp.status_code == 200

    def test_explain_response_has_required_fields(self):
        payload = {
            "time": 86400.0, "amount": 500.0,
            **{f"v{i}": 0.0 for i in range(1, 29)},
        }
        predict_resp = client.post("/predict", json=payload)
        txn_id = predict_resp.json()["transaction_id"]

        explain_resp = client.get(f"/explain/{txn_id}")
        data = explain_resp.json()
        assert "transaction_id" in data
        assert "fraud_score"    in data
        assert "is_fraud"       in data
        assert "explanation"    in data
        assert "similar_cases"  in data
        assert "query_used"     in data

    def test_explain_txn_id_matches(self):
        payload = {
            "time": 3600.0, "amount": 99.0,
            **{f"v{i}": 0.0 for i in range(1, 29)},
        }
        txn_id = client.post("/predict", json=payload).json()["transaction_id"]
        data   = client.get(f"/explain/{txn_id}").json()
        assert data["transaction_id"] == txn_id
