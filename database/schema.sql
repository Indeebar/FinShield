-- FinShield PostgreSQL Schema
-- Run this once to set up the database

-- Extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Raw transactions table (mirrors creditcard.csv structure)
CREATE TABLE IF NOT EXISTS transactions (
    id            UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    time          FLOAT        NOT NULL,  -- seconds elapsed from first transaction
    amount        FLOAT        NOT NULL,
    v1 FLOAT, v2 FLOAT, v3 FLOAT, v4 FLOAT, v5 FLOAT,
    v6 FLOAT, v7 FLOAT, v8 FLOAT, v9 FLOAT, v10 FLOAT,
    v11 FLOAT, v12 FLOAT, v13 FLOAT, v14 FLOAT, v15 FLOAT,
    v16 FLOAT, v17 FLOAT, v18 FLOAT, v19 FLOAT, v20 FLOAT,
    v21 FLOAT, v22 FLOAT, v23 FLOAT, v24 FLOAT, v25 FLOAT,
    v26 FLOAT, v27 FLOAT, v28 FLOAT,
    class         INTEGER      NOT NULL DEFAULT 0,  -- 0=legit, 1=fraud
    created_at    TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

-- Engineered features table (populated by feature pipeline)
CREATE TABLE IF NOT EXISTS transaction_features (
    id                    UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    transaction_id        UUID REFERENCES transactions(id) ON DELETE CASCADE,
    hour_of_day           INTEGER,         -- 0-23
    is_weekend            BOOLEAN,
    amount_log            FLOAT,           -- log1p(amount)
    amount_zscore         FLOAT,           -- z-score vs rolling mean
    rolling_count_1h      INTEGER,         -- txn count in last 1h window
    rolling_amount_1h     FLOAT,           -- total amount in last 1h window
    rolling_count_24h     INTEGER,
    rolling_amount_24h    FLOAT,
    amount_vs_mean_ratio  FLOAT,           -- amount / rolling_mean_24h
    created_at            TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Predictions table (stores model output for each transaction)
CREATE TABLE IF NOT EXISTS predictions (
    id                UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    transaction_id    UUID REFERENCES transactions(id) ON DELETE CASCADE,
    model_name        VARCHAR(50)  NOT NULL,  -- 'xgboost' or 'tabnet'
    fraud_score       FLOAT        NOT NULL,  -- probability 0.0-1.0
    is_fraud          BOOLEAN      NOT NULL,
    threshold_used    FLOAT        NOT NULL DEFAULT 0.5,
    shap_top_features JSONB,                  -- top 5 SHAP feature impacts
    latency_ms        FLOAT,                  -- inference latency
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes for fast query
CREATE INDEX IF NOT EXISTS idx_transactions_class     ON transactions(class);
CREATE INDEX IF NOT EXISTS idx_transactions_amount    ON transactions(amount);
CREATE INDEX IF NOT EXISTS idx_predictions_txn_id     ON predictions(transaction_id);
CREATE INDEX IF NOT EXISTS idx_predictions_is_fraud   ON predictions(is_fraud);
CREATE INDEX IF NOT EXISTS idx_predictions_model      ON predictions(model_name);
