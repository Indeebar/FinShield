"""
Unit tests for XGBoost training utilities and MLflow utils.
Tests use synthetic data to avoid needing the full 144MB dataset.
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
def synthetic_fraud_df():
    """
    Tiny synthetic dataset that mimics creditcard.csv structure.
    500 legit + 10 fraud transactions.
    """
    np.random.seed(42)
    n_legit, n_fraud = 500, 10
    n_total = n_legit + n_fraud

    data = {
        "time":   np.linspace(0, 172800, n_total),   # 2 days in seconds
        "amount": np.abs(np.random.normal(100, 150, n_total)),
        "class":  [0] * n_legit + [1] * n_fraud,
    }
    for i in range(1, 29):
        data[f"v{i}"] = np.random.randn(n_total)

    return pd.DataFrame(data)


# ------------------------------------------------------------------ #
# Feature Engineering Integration                                      #
# ------------------------------------------------------------------ #

class TestFeatureEngineeringIntegration:

    def test_all_features_present_after_transform(self, synthetic_fraud_df):
        from src.features.engineer import FraudFeatureEngineer
        eng = FraudFeatureEngineer()
        out = eng.fit_transform(synthetic_fraud_df)
        for feat in FraudFeatureEngineer.all_features():
            assert feat in out.columns, f"Missing feature: {feat}"

    def test_no_nulls_after_transform(self, synthetic_fraud_df):
        from src.features.engineer import FraudFeatureEngineer
        eng = FraudFeatureEngineer()
        out = eng.fit_transform(synthetic_fraud_df)
        feat_cols = FraudFeatureEngineer.all_features()
        assert out[feat_cols].isnull().sum().sum() == 0


# ------------------------------------------------------------------ #
# XGBoost Model Smoke Test                                             #
# ------------------------------------------------------------------ #

class TestXGBoostModel:

    def test_xgboost_trains_and_predicts(self, synthetic_fraud_df):
        """Smoke test: model trains and produces valid probabilities."""
        import xgboost as xgb
        from src.features.engineer import FraudFeatureEngineer
        from sklearn.model_selection import train_test_split

        eng = FraudFeatureEngineer()
        df  = eng.fit_transform(synthetic_fraud_df)
        X   = df[FraudFeatureEngineer.all_features()].values
        y   = df["class"].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        model = xgb.XGBClassifier(
            n_estimators=10,
            max_depth=3,
            random_state=42,
            scale_pos_weight=50,
            eval_metric="aucpr",
        )
        model.fit(X_train, y_train)

        proba = model.predict_proba(X_test)[:, 1]
        assert proba.shape[0] == X_test.shape[0]
        assert proba.min() >= 0.0
        assert proba.max() <= 1.0

    def test_xgboost_auc_above_chance(self, synthetic_fraud_df):
        """AUC-ROC should be > 0.5 (better than random)."""
        import xgboost as xgb
        from src.features.engineer import FraudFeatureEngineer
        from sklearn.model_selection import train_test_split

        eng = FraudFeatureEngineer()
        df  = eng.fit_transform(synthetic_fraud_df)
        X   = df[FraudFeatureEngineer.all_features()].values
        y   = df["class"].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        model = xgb.XGBClassifier(n_estimators=20, random_state=42, scale_pos_weight=50)
        model.fit(X_train, y_train)

        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        assert auc > 0.5, f"AUC-ROC {auc:.3f} is not above chance"


# ------------------------------------------------------------------ #
# MLflow Utils                                                         #
# ------------------------------------------------------------------ #

class TestMLflowUtils:

    @patch("mlflow.set_tracking_uri")
    @patch("mlflow.set_experiment")
    def test_setup_mlflow_called_correctly(self, mock_exp, mock_uri):
        from src.mlops.mlflow_utils import setup_mlflow
        setup_mlflow()
        mock_uri.assert_called_once()
        mock_exp.assert_called_once()
