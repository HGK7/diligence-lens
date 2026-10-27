from __future__ import annotations

from functools import lru_cache
from uuid import NAMESPACE_URL, uuid5

from langchain_ollama import OllamaEmbeddings
from qdrant_client import QdrantClient, models

from qdrant_store import (
    SEMANTIC_CACHE_COLLECTION,
    get_qdrant_client,
)

# Pin one embedding space for evidence and cached questions.
EMBED_MODEL = "nomic-embed-text:v1.5"
SEMANTIC_CACHE_THRESHOLD = 0.95


@lru_cache(maxsize=128)
def embed_text(text: str) -> list[float]:
    # Reuse a query vector across cache lookup, retrieval, and storage.
    embedder = OllamaEmbeddings(model=EMBED_MODEL)
    return embedder.embed_query(text)


def _cache_miss() -> dict:
    # Keep one predictable result shape for every miss.
    return {
        "hit": False,
        "report": None,
        "matched_question": None,
        "similarity": None,
    }


def _ensure_semantic_cache_collection(client: QdrantClient, vector_size: int) -> None:
    # Create the cache lazily from the active model's vector size.
    if not client.collection_exists(collection_name=SEMANTIC_CACHE_COLLECTION):
        client.create_collection(
            collection_name=SEMANTIC_CACHE_COLLECTION,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
        )


def check_semantic_cache(
    question: str, score_threshold: float, threshold: float = SEMANTIC_CACHE_THRESHOLD
) -> dict:
    # Match meaning only among reviews with the same evidence boundary.
    normalized = question.strip()
    if not normalized:
        return _cache_miss()
    review_key = f"{score_threshold:.4f}"
    vector = embed_text(normalized)
    client = get_qdrant_client()
    _ensure_semantic_cache_collection(client, len(vector))
    response = client.query_points(
        collection_name=SEMANTIC_CACHE_COLLECTION,
        query=vector,
        query_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="review_threshold",
                    match=models.MatchValue(value=review_key),
                )
            ]
        ),
        with_payload=True,
        score_threshold=threshold,
        limit=1,
    )
    # Extract ScoredPoint objects from the Pydantic response model.
    points = response.points
    if not points:
        return _cache_miss()
    point = points[0]
    payload = dict(point.payload or {})
    report = payload.get("report")
    if not isinstance(report, dict):
        return _cache_miss()
    return {
        "hit": True,
        "report": report,
        "matched_question": payload.get("question"),
        "similarity": float(point.score),
    }


def store_semantic_cache(question: str, score_threshold: float, report: dict) -> None:
    # Reject incomplete entries and preserve the evidence boundary.
    normalized = question.strip()
    if not normalized or not isinstance(report, dict):
        raise ValueError("A question and report dictionary are required.")
    review_key = f"{score_threshold:.4f}"
    vector = embed_text(normalized)
    identity = f"{SEMANTIC_CACHE_COLLECTION}:{review_key}:{normalized.casefold()}"
    point_id = str(uuid5(NAMESPACE_URL, identity))
    client = get_qdrant_client()
    _ensure_semantic_cache_collection(client, len(vector))
    client.upsert(
        collection_name=SEMANTIC_CACHE_COLLECTION,
        points=[
            models.PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "question": normalized,
                    "review_threshold": review_key,
                    "report": report,
                },
            )
        ],
        wait=True,
    )
