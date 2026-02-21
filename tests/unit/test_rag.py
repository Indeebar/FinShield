"""
Unit tests for the RAG explainability engine.
Uses in-memory ChromaDB (EphemeralClient) — no disk I/O needed.
"""

import pytest
import numpy as np
from unittest.mock import patch


# ------------------------------------------------------------------ #
# Knowledge Base Tests                                                 #
# ------------------------------------------------------------------ #

class TestKnowledgeBase:

    def test_get_all_cases_returns_36(self):
        from src.rag.knowledge_base import get_all_cases
        cases = get_all_cases()
        assert len(cases) == 36

    def test_case_ids_are_unique(self):
        from src.rag.knowledge_base import get_case_ids
        ids = get_case_ids()
        assert len(ids) == len(set(ids)), "Case IDs must be unique"

    def test_case_texts_not_empty(self):
        from src.rag.knowledge_base import get_case_texts
        texts = get_case_texts()
        for t in texts:
            assert len(t) > 20, f"Case text too short: {t}"

    def test_metadata_has_required_keys(self):
        from src.rag.knowledge_base import get_case_metadata
        for meta in get_case_metadata():
            assert "case_id"      in meta
            assert "pattern"      in meta
            assert "amount_range" in meta
            assert "time_context" in meta


# ------------------------------------------------------------------ #
# Embeddings Tests                                                     #
# ------------------------------------------------------------------ #

class TestEmbeddings:

    def test_encode_texts_shape(self):
        from src.rag.embeddings import encode_texts
        texts = ["fraud transaction at night", "small legitimate grocery purchase"]
        embeddings = encode_texts(texts)
        assert embeddings.shape == (2, 384)   # all-MiniLM-L6-v2 dim

    def test_encode_query_shape(self):
        from src.rag.embeddings import encode_query
        vec = encode_query("high amount at 3AM")
        assert vec.shape == (384,)

    def test_embeddings_normalized(self):
        from src.rag.embeddings import encode_texts
        vecs = encode_texts(["test sentence"])
        norm = np.linalg.norm(vecs[0])
        assert abs(norm - 1.0) < 1e-5, f"Expected unit norm, got {norm}"

    def test_features_to_query_contains_fraud_score(self):
        from src.rag.embeddings import features_to_query
        query = features_to_query(
            shap_top_features={"amount_vs_mean_ratio": 0.5, "is_night": 0.3},
            fraud_score=0.92,
            amount=1500.0,
            is_night=True,
        )
        assert "0.92" in query
        assert len(query) > 20


# ------------------------------------------------------------------ #
# Retriever Tests (in-memory ChromaDB)                                #
# ------------------------------------------------------------------ #

class TestRetriever:

    def test_build_index_in_memory(self):
        from src.rag.retriever import build_index
        collection = build_index(persist=False)
        assert collection.count() == 36

    def test_retrieve_returns_correct_n(self):
        from src.rag.retriever import retrieve_similar_cases, build_index
        build_index(persist=False)  # ensure collection exists in memory
        results = retrieve_similar_cases(
            query="stolen card used at night for large purchase",
            n_results=2,
            persist=False,
        )
        assert len(results) <= 2
        for r in results:
            assert "case_id"  in r
            assert "document" in r
            assert "distance" in r

    def test_retrieved_cases_have_metadata(self):
        from src.rag.retriever import retrieve_similar_cases, build_index
        build_index(persist=False)
        results = retrieve_similar_cases(
            query="high velocity transactions in one hour",
            n_results=1,
            persist=False,
        )
        assert len(results) == 1
        assert "pattern" in results[0]["metadata"]


# ------------------------------------------------------------------ #
# Explainer Tests                                                      #
# ------------------------------------------------------------------ #

class TestExplainer:

    def test_explain_returns_dataclass(self):
        from src.rag.retriever import build_index
        from src.rag.explainer import explain, FraudExplanation
        build_index(persist=False)

        result = explain(
            fraud_score=0.87,
            is_fraud=True,
            shap_top_features={"amount_vs_mean_ratio": 0.6, "is_night": 0.3, "v14": 0.2},
            amount=1500.0,
            is_night=True,
            n_cases=2,
            persist_chroma=False,
        )
        assert isinstance(result, FraudExplanation)
        assert result.is_fraud is True
        assert 0.0 <= result.fraud_score <= 1.0
        assert len(result.explanation) > 50
        assert "FRAUD" in result.explanation

    def test_explain_legitimate_transaction(self):
        from src.rag.retriever import build_index
        from src.rag.explainer import explain
        build_index(persist=False)

        result = explain(
            fraud_score=0.12,
            is_fraud=False,
            shap_top_features={"amount_log": 0.05},
            amount=25.0,
            is_night=False,
            n_cases=1,
            persist_chroma=False,
        )
        assert "LEGITIMATE" in result.explanation
