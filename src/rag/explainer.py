"""
RAG Explainer — FinShield
Main interface for the explainability engine.
Takes model output + SHAP features, retrieves similar fraud cases,
and returns a structured natural-language explanation.
"""

from dataclasses import dataclass
from loguru import logger

from src.rag.embeddings import features_to_query
from src.rag.retriever import retrieve_similar_cases


@dataclass
class FraudExplanation:
    """Structured explanation returned by the RAG engine."""
    fraud_score:        float
    is_fraud:           bool
    shap_top_features:  dict          # {feature: shap_value}
    similar_cases:      list[dict]    # retrieved ChromaDB hits
    explanation:        str           # human-readable summary
    query_used:         str           # the query sent to ChromaDB (for transparency)


def explain(
    fraud_score:       float,
    is_fraud:          bool,
    shap_top_features: dict,
    amount:            float,
    is_night:          bool,
    n_cases:           int  = 2,
    persist_chroma:    bool = True,
) -> FraudExplanation:
    """
    Generate a natural language fraud explanation via RAG.

    Parameters
    ----------
    fraud_score       : Model probability score (0.0 – 1.0)
    is_fraud          : Whether fraud threshold was crossed
    shap_top_features : dict of top SHAP feature → value pairs
    amount            : Transaction amount (for query construction)
    is_night          : Whether transaction occurred at night
    n_cases           : Number of similar historical cases to retrieve
    persist_chroma    : Use persistent ChromaDB (False = in-memory for tests)

    Returns
    -------
    FraudExplanation dataclass
    """
    # 1. Convert features → natural language query
    query = features_to_query(
        shap_top_features=shap_top_features,
        fraud_score=fraud_score,
        amount=amount,
        is_night=is_night,
    )
    logger.debug(f"RAG query: {query}")

    # 2. Retrieve similar cases from ChromaDB
    similar_cases = retrieve_similar_cases(
        query=query,
        n_results=n_cases,
        persist=persist_chroma,
    )

    # 3. Build natural language explanation
    explanation = _build_explanation(
        fraud_score=fraud_score,
        is_fraud=is_fraud,
        shap_top_features=shap_top_features,
        similar_cases=similar_cases,
    )

    return FraudExplanation(
        fraud_score=fraud_score,
        is_fraud=is_fraud,
        shap_top_features=shap_top_features,
        similar_cases=similar_cases,
        explanation=explanation,
        query_used=query,
    )


def _build_explanation(
    fraud_score:       float,
    is_fraud:          bool,
    shap_top_features: dict,
    similar_cases:     list[dict],
) -> str:
    """Build a structured explanation string from retrieved cases + features."""

    verdict       = "🚨 FRAUD DETECTED" if is_fraud else "✅ LEGITIMATE"
    confidence    = (
        "high" if fraud_score > 0.8 else
        "moderate" if fraud_score > 0.5 else
        "low"
    )

    # Top feature descriptions
    feature_lines = []
    for feat, val in list(shap_top_features.items())[:3]:
        direction = "increased" if val > 0 else "decreased"
        feature_lines.append(f"  • {feat} ({direction} fraud risk by {abs(val):.4f})")
    features_text = "\n".join(feature_lines)

    # Similar case summaries
    case_lines = []
    for i, case in enumerate(similar_cases, 1):
        meta   = case.get("metadata", {})
        doc    = case.get("document", "")
        # Show first 200 chars of document
        short  = doc[:200] + "..." if len(doc) > 200 else doc
        case_lines.append(
            f"  Case {i} [{meta.get('case_id', '?')} — {meta.get('pattern', '?')}]: {short}"
        )
    cases_text = "\n".join(case_lines) if case_lines else "  No similar cases found."

    explanation = (
        f"{verdict} (score: {fraud_score:.4f}, confidence: {confidence})\n\n"
        f"Top contributing features:\n{features_text}\n\n"
        f"Most similar historical fraud patterns:\n{cases_text}"
    )

    return explanation
