"""
Unit tests for feature engineering module.
"""

import pandas as pd
import numpy as np
import pytest
from src.features.engineer import FraudFeatureEngineer


@pytest.fixture
def sample_df():
    """Minimal transaction DataFrame for testing."""
    return pd.DataFrame({
        "time":   [0, 3600, 7200, 86400, 90000],
        "amount": [10.0, 250.0, 0.5, 1500.0, 75.0],
        "v1": [0.1] * 5, "v2": [-0.5] * 5,
        **{f"v{i}": [0.0] * 5 for i in range(3, 29)},
        "class":  [0, 0, 1, 0, 1],
    })


class TestFraudFeatureEngineer:

    def test_fit_stores_stats(self, sample_df):
        eng = FraudFeatureEngineer()
        eng.fit(sample_df)
        assert eng._mean_amount is not None
        assert eng._std_amount is not None
        assert eng._mean_amount == pytest.approx(sample_df["amount"].mean(), rel=1e-5)

    def test_temporal_features_created(self, sample_df):
        eng = FraudFeatureEngineer()
        out = eng.fit_transform(sample_df)
        assert "hour_of_day" in out.columns
        assert "is_night" in out.columns
        assert "is_weekend" in out.columns
        assert "day_of_week" in out.columns

    def test_amount_features_created(self, sample_df):
        eng = FraudFeatureEngineer()
        out = eng.fit_transform(sample_df)
        assert "amount_log" in out.columns
        assert "amount_zscore" in out.columns
        assert "amount_vs_mean_ratio" in out.columns

    def test_velocity_features_created(self, sample_df):
        eng = FraudFeatureEngineer()
        out = eng.fit_transform(sample_df)
        assert "rolling_count_1h" in out.columns
        assert "rolling_amount_1h" in out.columns
        assert "rolling_count_24h" in out.columns
        assert "rolling_amount_24h" in out.columns

    def test_amount_log_is_log1p(self, sample_df):
        eng = FraudFeatureEngineer()
        out = eng.fit_transform(sample_df)
        expected = np.log1p(out["amount"])
        pd.testing.assert_series_equal(out["amount_log"], expected, check_names=False)

    def test_hour_of_day_range(self, sample_df):
        eng = FraudFeatureEngineer()
        out = eng.fit_transform(sample_df)
        assert out["hour_of_day"].between(0, 23).all()

    def test_is_night_binary(self, sample_df):
        eng = FraudFeatureEngineer()
        out = eng.fit_transform(sample_df)
        assert set(out["is_night"].unique()).issubset({0, 1})

    def test_all_features_list_length(self):
        assert len(FraudFeatureEngineer.all_features()) == 39  # 28 PCA + 11 engineered

    def test_no_original_columns_dropped(self, sample_df):
        eng = FraudFeatureEngineer()
        out = eng.fit_transform(sample_df)
        assert "amount" in out.columns
        assert "class" in out.columns
