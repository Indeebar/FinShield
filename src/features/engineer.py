"""
Feature Engineering — FinShield
Transforms raw transaction data into ML-ready features.
"""

import numpy as np
import pandas as pd
from loguru import logger


class FraudFeatureEngineer:
    """
    Transforms raw transaction DataFrame into feature-rich ML input.

    Features Generated
    ------------------
    - Temporal   : hour_of_day, is_night, is_weekend
    - Amount     : amount_log, amount_zscore, amount_vs_mean_ratio
    - Velocity   : rolling_count_1h, rolling_amount_1h,
                   rolling_count_24h, rolling_amount_24h
    - PCA        : V1-V28 passed through as-is (already PCA reduced by Kaggle)
    """

    # Seconds per hour/day (dataset Time column is in seconds)
    SECONDS_PER_HOUR = 3600
    SECONDS_PER_DAY  = 86400

    def __init__(self):
        self._mean_amount: float | None = None
        self._std_amount:  float | None = None

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def fit(self, df: pd.DataFrame) -> "FraudFeatureEngineer":
        """Compute global statistics from training data (call before transform)."""
        self._mean_amount = df["amount"].mean()
        self._std_amount  = df["amount"].std()
        logger.info(
            f"Fit complete | mean_amount={self._mean_amount:.2f} "
            f"std_amount={self._std_amount:.2f}"
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply full feature engineering pipeline to a DataFrame."""
        df = df.copy()
        df = self._add_temporal_features(df)
        df = self._add_amount_features(df)
        df = self._add_velocity_features(df)
        logger.info(f"Feature engineering done | shape={df.shape}")
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convenience: fit + transform in one call."""
        return self.fit(df).transform(df)

    # ------------------------------------------------------------------ #
    # Feature Groups                                                       #
    # ------------------------------------------------------------------ #

    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract time-based signals from the Time column (seconds offset)."""
        df["hour_of_day"] = (df["time"] % self.SECONDS_PER_DAY // self.SECONDS_PER_HOUR).astype(int)
        df["is_night"]    = df["hour_of_day"].apply(lambda h: 1 if (h < 6 or h >= 22) else 0)
        # Kaggle dataset has no real calendar date; simulate day of week via modulo
        df["day_of_week"] = (df["time"] // self.SECONDS_PER_DAY % 7).astype(int)
        df["is_weekend"]  = df["day_of_week"].apply(lambda d: 1 if d >= 5 else 0)
        return df

    def _add_amount_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Log-transform, z-score, and ratio features for transaction amount."""
        df["amount_log"] = np.log1p(df["amount"])

        if self._mean_amount is not None and self._std_amount is not None:
            df["amount_zscore"] = (df["amount"] - self._mean_amount) / (self._std_amount + 1e-8)
        else:
            # Fallback: compute in-place if not fitted (e.g. single row inference)
            df["amount_zscore"] = (df["amount"] - df["amount"].mean()) / (df["amount"].std() + 1e-8)

        return df

    def _add_velocity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute rolling window transaction velocity features.

        Vectorised O(n log n) implementation using numpy prefix sums +
        searchsorted. For 284K rows this runs in ~1 second vs ~14 minutes
        for the naive O(n²) loop.

        For real-time single-transaction inference the window will be empty
        (only prior transactions count). This is expected behaviour.
        """
        df = df.sort_values("time").reset_index(drop=True)

        times   = df["time"].to_numpy()
        amounts = df["amount"].to_numpy()

        # Prefix sum of amounts over the sorted time axis
        cum_amounts = np.concatenate([[0.0], np.cumsum(amounts)])

        # For each row i, find the left edge of the 1h / 24h window using
        # binary search, then read off count and amount sum in O(1).
        starts_1h  = np.searchsorted(times, times - self.SECONDS_PER_HOUR, side="left")
        starts_24h = np.searchsorted(times, times - self.SECONDS_PER_DAY,  side="left")
        indices    = np.arange(len(times))

        rolling_count_1h  = indices - starts_1h
        rolling_count_24h = indices - starts_24h

        # sum(amounts[start:i]) = cum_amounts[i] - cum_amounts[start]
        rolling_amount_1h  = cum_amounts[indices] - cum_amounts[starts_1h]
        rolling_amount_24h = cum_amounts[indices] - cum_amounts[starts_24h]

        df["rolling_count_1h"]   = rolling_count_1h
        df["rolling_amount_1h"]  = rolling_amount_1h
        df["rolling_count_24h"]  = rolling_count_24h
        df["rolling_amount_24h"] = rolling_amount_24h

        # Ratio: how does this transaction compare to user's 24h average?
        avg_24h = rolling_amount_24h / (rolling_count_24h + 1)
        df["amount_vs_mean_ratio"] = amounts / (avg_24h + 1e-8)

        return df

    # ------------------------------------------------------------------ #
    # Feature column lists (used by model training)                        #
    # ------------------------------------------------------------------ #

    @staticmethod
    def pca_features() -> list[str]:
        """The 28 PCA-reduced features from Kaggle dataset."""
        return [f"v{i}" for i in range(1, 29)]

    @staticmethod
    def engineered_features() -> list[str]:
        """All engineered features (non-PCA)."""
        return [
            "amount_log",
            "amount_zscore",
            "amount_vs_mean_ratio",
            "hour_of_day",
            "is_night",
            "is_weekend",
            "day_of_week",
            "rolling_count_1h",
            "rolling_amount_1h",
            "rolling_count_24h",
            "rolling_amount_24h",
        ]

    @classmethod
    def all_features(cls) -> list[str]:
        """Combined feature set: PCA + engineered."""
        return cls.pca_features() + cls.engineered_features()
