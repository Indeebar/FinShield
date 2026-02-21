"""
Unit tests for TabNet training utilities and model comparison.
Uses synthetic data — no full dataset required.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from sklearn.metrics import roc_auc_score


# ------------------------------------------------------------------ #
# Fixtures                                                             #
# ------------------------------------------------------------------ #

@pytest.fixture
def synthetic_df():
    """700 legit + 14 fraud rows (simulates severe class imbalance)."""
    np.random.seed(0)
    n_legit, n_fraud = 700, 14
    n = n_legit + n_fraud
    data = {
        "time":   np.linspace(0, 172800, n),
        "amount": np.abs(np.random.normal(88, 130, n)),
        "class":  [0] * n_legit + [1] * n_fraud,
    }
    for i in range(1, 29):
        data[f"v{i}"] = np.random.randn(n)
    return pd.DataFrame(data)


# ------------------------------------------------------------------ #
# TabNet smoke tests                                                   #
# ------------------------------------------------------------------ #

class TestTabNetModel:

    def test_tabnet_instantiates(self):
        """TabNetClassifier can be created with our default params."""
        from pytorch_tabnet.tab_model import TabNetClassifier
        model = TabNetClassifier(n_d=8, n_a=8, n_steps=2, verbose=0)
        assert model is not None

    def test_tabnet_trains_on_synthetic_data(self, synthetic_df):
        """Smoke test: TabNet trains and outputs valid probabilities."""
        from pytorch_tabnet.tab_model import TabNetClassifier
        from src.features.engineer import FraudFeatureEngineer
        from sklearn.model_selection import train_test_split

        eng = FraudFeatureEngineer()
        df  = eng.fit_transform(synthetic_df)
        X   = df[FraudFeatureEngineer.all_features()].values.astype(np.float32)
        y   = df["class"].values.astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
        )

        model = TabNetClassifier(n_d=8, n_a=8, n_steps=2, verbose=0)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            max_epochs=5,
            patience=3,
            batch_size=64,
            virtual_batch_size=16,
            num_workers=0,
        )

        proba = model.predict_proba(X_test)[:, 1]
        assert proba.shape[0] == X_test.shape[0]
        assert 0.0 <= proba.min() and proba.max() <= 1.0

    def test_tabnet_outputs_feature_importances(self, synthetic_df):
        """feature_importances_ should be a 1D array of length n_features."""
        from pytorch_tabnet.tab_model import TabNetClassifier
        from src.features.engineer import FraudFeatureEngineer
        from sklearn.model_selection import train_test_split

        eng = FraudFeatureEngineer()
        df  = eng.fit_transform(synthetic_df)
        X   = df[FraudFeatureEngineer.all_features()].values.astype(np.float32)
        y   = df["class"].values.astype(int)

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_tr, X_v, y_tr, y_v = train_test_split(
            X_tr, y_tr, test_size=0.15, random_state=42, stratify=y_tr
        )

        model = TabNetClassifier(n_d=8, n_a=8, n_steps=2, verbose=0)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_v, y_v)],
            max_epochs=3,
            patience=2,
            batch_size=64,
            virtual_batch_size=16,
            num_workers=0,
        )
        importances = model.feature_importances_
        assert importances.shape[0] == X.shape[1]
        assert importances.sum() > 0  # at least some feature used


# ------------------------------------------------------------------ #
# Model comparison tests                                               #
# ------------------------------------------------------------------ #

class TestModelComparison:

    @patch("src.models.compare_models.mlflow.search_runs")
    @patch("src.models.compare_models.mlflow.tracking.MlflowClient")
    @patch("src.models.compare_models.setup_mlflow")
    def test_compare_returns_winner(self, mock_setup, mock_client, mock_search):
        """compare_and_promote() returns winner dict without crashing."""
        from src.models.compare_models import compare_and_promote

        # Mock MLflow experiment
        mock_exp = MagicMock()
        mock_exp.experiment_id = "1"
        mock_client.return_value.get_experiment_by_name.return_value = mock_exp

        # Mock 2 runs
        mock_search.return_value = pd.DataFrame([
            {
                "run_id": "abc123",
                "tags.mlflow.runName": "xgboost-fraud",
                "metrics.auc_roc": 0.97,
                "metrics.avg_precision": 0.80,
                "metrics.f1_fraud": 0.75,
                "metrics.precision_fraud": 0.78,
                "metrics.recall_fraud": 0.72,
            },
            {
                "run_id": "def456",
                "tags.mlflow.runName": "tabnet-fraud",
                "metrics.auc_roc": 0.96,
                "metrics.avg_precision": 0.78,
                "metrics.f1_fraud": 0.73,
                "metrics.precision_fraud": 0.76,
                "metrics.recall_fraud": 0.71,
            },
        ])

        result = compare_and_promote(register=False)
        assert result["run_id"] == "abc123"
        assert result["model_name"] == "finshield-xgboost"
        assert result["metrics"]["avg_precision"] == 0.80
