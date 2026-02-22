"""
Unit tests for FastAPI serving layer — schemas and model loader.
TDD: These tests are written BEFORE implementation code.
"""

import pytest
from pathlib import Path


# ------------------------------------------------------------------ #
# Pydantic Schema Tests                                                #
# ------------------------------------------------------------------ #

class TestTransactionInput:

    def test_valid_transaction_accepted(self):
        from src.serving.schemas import TransactionInput
        data = {f"v{i}": 0.1 * i for i in range(1, 29)}
        data["time"]   = 100000.0
        data["amount"] = 150.0
        txn = TransactionInput(**data)
        assert txn.amount == 150.0
        assert txn.time   == 100000.0

    def test_negative_amount_rejected(self):
        from src.serving.schemas import TransactionInput
        from pydantic import ValidationError
        data = {f"v{i}": 0.0 for i in range(1, 29)}
        data["time"]   = 0.0
        data["amount"] = -10.0
        with pytest.raises(ValidationError):
            TransactionInput(**data)

    def test_missing_v_field_rejected(self):
        from src.serving.schemas import TransactionInput
        from pydantic import ValidationError
        # Only v1..v27 — missing v28
        data = {f"v{i}": 0.0 for i in range(1, 28)}
        data["time"]   = 0.0
        data["amount"] = 50.0
        with pytest.raises(ValidationError):
            TransactionInput(**data)

    def test_to_raw_dict_includes_all_pca_fields(self):
        from src.serving.schemas import TransactionInput
        data = {f"v{i}": float(i) for i in range(1, 29)}
        data["time"]   = 500.0
        data["amount"] = 99.99
        txn = TransactionInput(**data)
        raw = txn.to_raw_dict()
        assert "time"   in raw
        assert "amount" in raw
        for i in range(1, 29):
            assert f"v{i}" in raw


class TestPredictResponse:

    def test_predict_response_fields(self):
        from src.serving.schemas import PredictResponse
        resp = PredictResponse(
            txn_id="abc123",
            fraud_score=0.87,
            is_fraud=True,
            shap_top_features={"v14": 0.5, "amount_log": 0.3},
            explanation="FRAUD DETECTED"
        )
        assert resp.txn_id       == "abc123"
        assert resp.fraud_score  == 0.87
        assert resp.is_fraud     is True
        assert "v14"             in resp.shap_top_features

    def test_fraud_score_bounded(self):
        from src.serving.schemas import PredictResponse
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            PredictResponse(
                txn_id="x",
                fraud_score=1.5,     # > 1.0
                is_fraud=True,
                shap_top_features={},
                explanation=""
            )


class TestHealthResponse:

    def test_health_response_fields_present(self):
        from src.serving.schemas import HealthResponse
        h = HealthResponse(status="ok", model_loaded=True, chroma_ok=True, db_ok=False)
        assert h.status        == "ok"
        assert h.model_loaded  is True
        assert h.chroma_ok     is True
        assert h.db_ok         is False


# ------------------------------------------------------------------ #
# ModelLoader Tests                                                    #
# ------------------------------------------------------------------ #

class TestModelLoader:

    def test_is_loaded_false_before_load(self):
        from src.serving.model_loader import ModelLoader
        loader = ModelLoader()
        assert loader.is_loaded() is False

    def test_load_raises_if_model_file_missing(self, tmp_path):
        from src.serving.model_loader import ModelLoader
        loader = ModelLoader(
            model_path=str(tmp_path / "nonexistent.json"),
            engineer_path=str(tmp_path / "nonexistent.pkl"),
        )
        with pytest.raises(RuntimeError, match="Model file not found"):
            loader.load()

    def test_load_raises_if_engineer_file_missing(self, tmp_path):
        from src.serving.model_loader import ModelLoader
        # Create a dummy model file, but no engineer
        dummy_model = tmp_path / "model.json"
        dummy_model.write_text("{}")
        loader = ModelLoader(
            model_path=str(dummy_model),
            engineer_path=str(tmp_path / "nonexistent.pkl"),
        )
        with pytest.raises(RuntimeError, match="Feature engineer file not found"):
            loader.load()

    def test_model_property_raises_if_not_loaded(self):
        from src.serving.model_loader import ModelLoader
        loader = ModelLoader()
        with pytest.raises(RuntimeError):
            _ = loader.model

    def test_engineer_property_raises_if_not_loaded(self):
        from src.serving.model_loader import ModelLoader
        loader = ModelLoader()
        with pytest.raises(RuntimeError):
            _ = loader.engineer
