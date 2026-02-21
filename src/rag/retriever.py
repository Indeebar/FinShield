"""
ChromaDB Retriever — FinShield RAG Engine
Manages the vector store for fraud case similarity search.
"""

import chromadb
from chromadb.config import Settings
from loguru import logger
from pathlib import Path

from src.rag.knowledge_base import get_all_cases, get_case_texts, get_case_ids, get_case_metadata

# Local persistent store path
CHROMA_PATH = Path(__file__).parents[2] / "data" / "chroma_db"
COLLECTION_NAME = "fraud_cases"


def get_chroma_client(persist: bool = True) -> chromadb.ClientAPI:
    """
    Return a ChromaDB client — persistent (default) or in-memory (for tests).

    Parameters
    ----------
    persist : If True, stores vectors on disk at CHROMA_PATH
    """
    if persist:
        CHROMA_PATH.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    else:
        client = chromadb.EphemeralClient()   # in-memory — for tests
    return client


def build_index(persist: bool = True) -> chromadb.Collection:
    """
    (Re)build the ChromaDB vector index from the fraud case knowledge base.
    Safe to call multiple times — deletes and rebuilds.

    Returns
    -------
    chromadb.Collection
    """
    client = get_chroma_client(persist=persist)

    # Delete existing collection to allow clean rebuild
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},   # cosine similarity
    )

    texts     = get_case_texts()
    ids       = get_case_ids()
    metadatas = get_case_metadata()

    # Encode all documents with sentence-transformers
    from src.rag.embeddings import encode_texts
    embeddings = encode_texts(texts).tolist()

    logger.info(f"Indexing {len(texts)} fraud cases into ChromaDB...")
    # Single upsert with embeddings + documents — no separate add() needed
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )

    logger.success(f"ChromaDB index built — {collection.count()} documents")
    return collection


def get_collection(persist: bool = True) -> chromadb.Collection:
    """
    Get the existing collection, or build it if it doesn't exist.

    Returns
    -------
    chromadb.Collection
    """
    client = get_chroma_client(persist=persist)
    try:
        collection = client.get_collection(COLLECTION_NAME)
        if collection.count() == 0:
            raise ValueError("Empty collection")
        logger.info(f"ChromaDB collection loaded — {collection.count()} documents")
        return collection
    except Exception:
        logger.warning("Collection not found or empty, rebuilding index...")
        return build_index(persist=persist)


def retrieve_similar_cases(
    query: str,
    n_results: int = 2,
    persist: bool = True,
) -> list[dict]:
    """
    Find the most semantically similar fraud cases for a given query.

    Parameters
    ----------
    query     : Natural language description of the current transaction
    n_results : Number of similar cases to return (default 2)
    persist   : Use persistent or in-memory ChromaDB

    Returns
    -------
    List of dicts:  [{case_id, document, metadata, distance}, ...]
    """
    from src.rag.embeddings import encode_query
    collection  = get_collection(persist=persist)
    query_embed = encode_query(query).tolist()

    results = collection.query(
        query_embeddings=[query_embed],
        n_results=min(n_results, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for i in range(len(results["ids"][0])):
        hits.append({
            "case_id":  results["ids"][0][i],
            "document": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": round(results["distances"][0][i], 4),
        })

    logger.debug(f"Retrieved {len(hits)} similar cases for query")
    return hits
