"""
Data Loader — FinShield
Handles CSV ingestion and database read/write for transaction data.
"""

import os
import pandas as pd
from pathlib import Path
from loguru import logger
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from dotenv import load_dotenv

load_dotenv()

RAW_DATA_PATH = Path(__file__).parents[2] / "data" / "raw" / "creditcard.csv"
PROCESSED_PATH = Path(__file__).parents[2] / "data" / "processed"


def get_engine() -> Engine:
    """Create and return a SQLAlchemy engine from DATABASE_URL env var."""
    db_url = os.getenv("DATABASE_URL", "postgresql://finshield:finshield@localhost:5432/finshield")
    engine = create_engine(db_url, pool_pre_ping=True)
    logger.info(f"DB engine created: {db_url.split('@')[-1]}")
    return engine


def load_raw_csv(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load the raw creditcard.csv dataset.

    Returns
    -------
    pd.DataFrame with columns: Time, V1..V28, Amount, Class
    """
    logger.info(f"Loading raw CSV: {path}")
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]  # normalize to lowercase
    logger.info(f"Loaded {len(df):,} rows | Fraud rate: {df['class'].mean()*100:.3f}%")
    return df


def ingest_to_db(df: pd.DataFrame, engine: Engine, chunksize: int = 5000) -> None:
    """
    Write raw transactions DataFrame to the transactions table.

    Parameters
    ----------
    df       : DataFrame from load_raw_csv()
    engine   : SQLAlchemy engine
    chunksize: rows per INSERT batch
    """
    logger.info(f"Ingesting {len(df):,} rows to transactions table...")
    df.to_sql(
        name="transactions",
        con=engine,
        if_exists="append",
        index=False,
        chunksize=chunksize,
        method="multi",
    )
    logger.success(f"Ingestion complete — {len(df):,} rows written.")


def load_from_db(engine: Engine, limit: int | None = None) -> pd.DataFrame:
    """
    Load transactions from PostgreSQL.

    Parameters
    ----------
    engine : SQLAlchemy engine
    limit  : Optional row limit (useful for dev/testing)
    """
    query = "SELECT * FROM transactions"
    if limit:
        query += f" LIMIT {limit}"
    logger.info(f"Querying DB: {query}")
    df = pd.read_sql(query, con=engine)
    logger.info(f"Loaded {len(df):,} rows from DB.")
    return df


def get_class_distribution(df: pd.DataFrame) -> dict:
    """Return fraud vs legitimate transaction counts and ratio."""
    counts = df["class"].value_counts().to_dict()
    return {
        "legitimate": counts.get(0, 0),
        "fraud": counts.get(1, 0),
        "fraud_rate_pct": round(df["class"].mean() * 100, 4),
        "imbalance_ratio": round(counts.get(0, 1) / max(counts.get(1, 1), 1), 1),
    }


if __name__ == "__main__":
    df = load_raw_csv()
    print(get_class_distribution(df))
    print(df.head(3))
