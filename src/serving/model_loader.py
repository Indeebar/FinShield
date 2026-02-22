"""
Model Loader — FinShield Serving Layer
Loads XGBoost model + fitted FraudFeatureEngineer from local disk.
Raises RuntimeError with a clear message if files are missing.
"""

from pathlib import Path
import joblib
import xgboost as xgb
from loguru import logger

# Default paths (relative to repo root: F:\FinShield\models\)
_REPO_ROOT     = Path(__file__).parents[2]
DEFAULT_MODEL_PATH    = str(_REPO_ROOT / "models" / "xgboost_fraud.json")
DEFAULT_ENGINEER_PATH = str(_REPO_ROOT / "models" / "feature_engineer.pkl")


class ModelLoader:
    """
    Loads the trained XGBoost model and fitted FraudFeatureEngineer.

    Usage
    -----
    loader = ModelLoader()
    loader.load()           # call once at app startup
    loader.is_loaded()      # True after successful load
    loader.model            # xgb.XGBClassifier
    loader.engineer         # FraudFeatureEngineer (already fitted)
    """

    def __init__(
        self,
        model_path:    str = DEFAULT_MODEL_PATH,
        engineer_path: str = DEFAULT_ENGINEER_PATH,
    ) -> None:
        self._model_path    = model_path
        self._engineer_path = engineer_path
        self._model         = None
        self._engineer      = None

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def load(self) -> None:
        """
        Load model + engineer from disk.

        Raises
        ------
        RuntimeError if either file is missing.
        """
        # 1. Validate model file exists
        if not Path(self._model_path).exists():
            raise RuntimeError(
                f"Model file not found: {self._model_path}. "
                "Run `python -m src.models.train_xgboost` first."
            )

        # 2. Validate engineer file exists
        if not Path(self._engineer_path).exists():
            raise RuntimeError(
                f"Feature engineer file not found: {self._engineer_path}. "
                "Run `python -m src.models.train_xgboost` first."
            )

        # 3. Load XGBoost model
        model = xgb.XGBClassifier()
        model.load_model(self._model_path)
        self._model = model
        logger.info(f"XGBoost model loaded from {self._model_path}")

        # 4. Load feature engineer (fitted joblib pickle)
        self._engineer = joblib.load(self._engineer_path)
        logger.info(f"Feature engineer loaded from {self._engineer_path}")

    def is_loaded(self) -> bool:
        """Return True if both model and engineer are successfully loaded."""
        return self._model is not None and self._engineer is not None

    @property
    def model(self) -> xgb.XGBClassifier:
        """Trained XGBoost classifier."""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call loader.load() first.")
        return self._model

    @property
    def engineer(self):
        """Fitted FraudFeatureEngineer instance."""
        if self._engineer is None:
            raise RuntimeError("Feature engineer not loaded. Call loader.load() first.")
        return self._engineer
