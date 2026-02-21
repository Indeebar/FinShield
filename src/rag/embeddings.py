"""
Embeddings — FinShield RAG Engine
Wraps sentence-transformers for encoding fraud case documents and queries.
"""

import numpy as np
from functools import lru_cache
from loguru import logger
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"   # Fast, 384-dim, great for semantic similarity


@lru_cache(maxsize=1)
def get_encoder() -> SentenceTransformer:
    """
    Load and cache the sentence-transformer model.
    Cached so it's only loaded once per process (expensive first load).
    """
    logger.info(f"Loading sentence-transformer: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    logger.success(f"Encoder ready | dim={model.get_sentence_embedding_dimension()}")
    return model


def encode_texts(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """
    Encode a list of strings into embedding vectors.

    Parameters
    ----------
    texts      : List of strings to encode
    batch_size : Encoding batch size (32 is efficient for all-MiniLM-L6-v2)

    Returns
    -------
    np.ndarray of shape (n_texts, 384) — L2-normalised embeddings
    """
    encoder = get_encoder()
    embeddings = encoder.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,    # L2 norm → cosine similarity = dot product
        show_progress_bar=len(texts) > 10,
    )
    logger.debug(f"Encoded {len(texts)} texts → shape {embeddings.shape}")
    return embeddings


def encode_query(query: str) -> np.ndarray:
    """Encode a single query string. Returns shape (384,)."""
    return encode_texts([query])[0]


def features_to_query(
    shap_top_features: dict,
    fraud_score: float,
    amount: float,
    is_night: bool,
) -> str:
    """
    Convert model output into a natural language query for ChromaDB retrieval.

    Parameters
    ----------
    shap_top_features : dict of {feature_name: shap_value} (top 3-5 features)
    fraud_score       : Model output probability
    amount            : Transaction amount
    is_night          : Whether transaction happened at night

    Returns
    -------
    str : Human-readable query for semantic search
    """
    top_feats = list(shap_top_features.keys())[:3]

    query_parts = [f"Fraud score {fraud_score:.2f}."]

    if amount:
        if amount > 1000:
            query_parts.append("Very high transaction amount.")
        elif amount > 300:
            query_parts.append("High transaction amount.")
        else:
            query_parts.append("Moderate transaction amount.")

    if is_night:
        query_parts.append("Transaction occurred at night or unusual hours.")

    # Map feature names to natural language
    feature_descriptions = {
        "amount_vs_mean_ratio": "amount significantly above the cardholder's normal average",
        "rolling_count_1h":     "high number of transactions in last hour",
        "rolling_count_24h":    "high number of transactions in last 24 hours",
        "rolling_amount_1h":    "unusually high spending in last hour",
        "rolling_amount_24h":   "unusually high spending in last 24 hours",
        "amount_log":           "large transaction amount",
        "amount_zscore":        "amount far from cardholder baseline",
        "is_night":             "night time transaction",
        "v14":                  "strong V14 anomaly — known fraud signal",
        "v10":                  "V10 deviation",
        "v12":                  "V12 deviation",
        "v4":                   "V4 time-pattern anomaly",
        "v17":                  "V17 high-risk merchant signal",
    }

    for feat in top_feats:
        desc = feature_descriptions.get(feat.lower(), f"{feat} anomaly detected")
        query_parts.append(f"Key signal: {desc}.")

    return " ".join(query_parts)
