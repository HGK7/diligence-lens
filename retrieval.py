from __future__ import annotations

import os
from time import perf_counter
from uuid import NAMESPACE_URL, uuid5

from langchain_ollama import OllamaEmbeddings
from qdrant_client import models

from chunking import (
    CHUNK_MIN_CHARS,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DOCS_PATH,
    LEGACY_DOCS_PATH,
    contract_sections,
    corpus_file_count,
    load_seed_records,
    load_uploaded_records,
    parse_clause_heading,
    records_from_bytes,
)
from diversity import diversify_by_source
from hybrid import reciprocal_rank_fusion
from lexical_retrieval import get_bm25_index, search_bm25
from qdrant_store import (
    COLLECTION_NAME,
    DB_PATH,
    QDRANT_URL,
    SEMANTIC_CACHE_COLLECTION,
    _wipe_path_store,
    clear_index,
    get_qdrant_client,
    qdrant_target,
    recover_qdrant_client,
    reset_qdrant_client,
)
from reranking import rerank
from semantic_cache import (
    EMBED_MODEL,
    SEMANTIC_CACHE_THRESHOLD,
    check_semantic_cache,
    embed_text,
    store_semantic_cache,
)

# Local Qdrant path-mode is a single-process server; batch upserts keep it stable.
UPSERT_BATCH_SIZE = int(os.getenv("UPSERT_BATCH_SIZE", "128"))
# nomic-embed is tiny (~335 MB) and already 100% GPU in `ollama ps`.
# Batch 8 under-fed an RTX 3050ti; 32 is a better default. Raise via EMBED_BATCH_SIZE
# (try 48/64). If WinError 10054 returns, drop back toward 16.
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "32"))


def _emit(
    progress_callback,
    *,
    stage: str,
    current: int,
    total: int,
    message: str,
) -> None:
    line = f"[{stage}] {current}/{total} — {message}"
    print(line, flush=True)
    if progress_callback is not None:
        progress_callback(stage=stage, current=current, total=total, message=message)


def build_index(
    records: list[dict], progress_callback=None, *, wipe: bool = True
) -> dict:
    """
    Embed records and write a fresh Qdrant collection.

    wipe=True (default): destroy prior diligence + cache data before upsert so
    changed chunk IDs cannot stack on orphans. Path mode removes DB_PATH;
    URL mode deletes the named collections.
    """
    if not records:
        raise ValueError("No readable document text was found.")

    total = len(records)
    # Open Qdrant BEFORE the long embed run so lock conflicts fail fast
    # and we reuse the process singleton (never re-open path mid-build).
    _emit(
        progress_callback,
        stage="qdrant",
        current=0,
        total=total,
        message=f"Opening Qdrant at `{qdrant_target()}` (fail-fast before embed)",
    )
    try:
        client = get_qdrant_client()
    except Exception as error:
        hint = (
            f"Check QDRANT_URL=`{QDRANT_URL}` is reachable."
            if QDRANT_URL
            else (
                "Stop every other Streamlit/Python using this folder, then retry. "
                "Do not delete `.lock` while another process still holds it."
            )
        )
        raise RuntimeError(
            f"Cannot open Qdrant at `{qdrant_target()}`: {error}. {hint}"
        ) from error

    _emit(
        progress_callback,
        stage="prepare",
        current=0,
        total=total,
        message=f"Embedding {total} chunks with Ollama `{EMBED_MODEL}` "
        f"(batch={EMBED_BATCH_SIZE})",
    )

    embedder = OllamaEmbeddings(model=EMBED_MODEL)
    texts = [record.get("embedding_text", record["text"]) for record in records]
    vectors: list[list[float]] = []

    for start in range(0, total, EMBED_BATCH_SIZE):
        end = min(start + EMBED_BATCH_SIZE, total)
        batch_texts = texts[start:end]
        try:
            batch_vectors = embedder.embed_documents(batch_texts)
        except Exception as error:
            raise RuntimeError(
                f"Ollama embedding failed on chunks {start + 1}-{end}/{total}: {error}. "
                "Check `ollama serve` is running. Watch GPU with nvidia-smi while "
                "batches print in the console."
            ) from error
        if len(batch_vectors) != len(batch_texts):
            raise RuntimeError(
                f"Ollama returned {len(batch_vectors)} vectors for "
                f"{len(batch_texts)} texts (chunks {start + 1}-{end})."
            )
        vectors.extend(batch_vectors)
        _emit(
            progress_callback,
            stage="embed",
            current=end,
            total=total,
            message=f"Embedded batch {start + 1}-{end}",
        )

    dimensions = len(vectors[0])
    _emit(
        progress_callback,
        stage="qdrant",
        current=0,
        total=total,
        message=f"Writing collection `{COLLECTION_NAME}` ({dimensions} dims)",
    )

    try:
        if wipe:
            if QDRANT_URL:
                for name in (COLLECTION_NAME, SEMANTIC_CACHE_COLLECTION):
                    if client.collection_exists(collection_name=name):
                        client.delete_collection(collection_name=name)
                        _emit(
                            progress_callback,
                            stage="qdrant",
                            current=0,
                            total=total,
                            message=f"Deleted collection `{name}`",
                        )
                if client.collection_exists(collection_name=COLLECTION_NAME):
                    raise RuntimeError(
                        f"Collection `{COLLECTION_NAME}` still exists after delete. "
                        "Stop other writers and retry with clear_index()."
                    )
            else:
                # Path mode: rmtree the store. delete_collection leaves orphans
                # when point UUID identity strings change between rebuilds.
                _emit(
                    progress_callback,
                    stage="qdrant",
                    current=0,
                    total=total,
                    message=f"Wiping path store `{DB_PATH}` before create",
                )
                _wipe_path_store()
                client = get_qdrant_client()

        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=dimensions, distance=models.Distance.COSINE
            ),
        )
        _emit(
            progress_callback,
            stage="qdrant",
            current=0,
            total=total,
            message=f"Created collection `{COLLECTION_NAME}`",
        )

        # Content-only UUIDs silently collapsed duplicate chunks (same source/
        # section/index/text) on upsert — ~59 fewer points than embeds.
        # Ordinal keeps one Qdrant point per embedded record.
        points = []
        for ordinal, (record, vector) in enumerate(
            zip(records, vectors, strict=True), start=1
        ):
            identity = (
                f"{COLLECTION_NAME}:{ordinal}:{record['source']}:"
                f"{record['section']}:{record['chunk_index']}"
            )
            payload = {k: v for k, v in record.items() if k != "embedding_text"}
            payload["ordinal"] = ordinal
            points.append(
                models.PointStruct(
                    id=str(uuid5(NAMESPACE_URL, identity)),
                    vector=vector,
                    payload=payload,
                )
            )

        for start in range(0, len(points), UPSERT_BATCH_SIZE):
            end = min(start + UPSERT_BATCH_SIZE, len(points))
            batch = points[start:end]
            client.upsert(collection_name=COLLECTION_NAME, points=batch, wait=True)
            _emit(
                progress_callback,
                stage="upsert",
                current=end,
                total=total,
                message=f"Upserted points {start + 1}-{end}",
            )

    except Exception as e:
        raise RuntimeError(f"Failed to rebuild index collections: {e}") from e

    _emit(
        progress_callback,
        stage="done",
        current=total,
        total=total,
        message=f"Indexed {total} chunks · {dimensions} dims",
    )
    return {"records": total, "dimensions": dimensions}


def index_exists() -> bool:
    return bool(get_index_stats().get("ready"))


def get_index_stats() -> dict:
    """
    Truth for "is the index done?": collection must exist AND hold points.

    Embeddings alone do not count — vectors live in RAM until upsert finishes.
    """
    stats = {
        "ready": False,
        "exists": False,
        "points": 0,
        "collection": COLLECTION_NAME,
        "path": qdrant_target(),
        "mode": "url" if QDRANT_URL else "path",
        "error": None,
    }
    if not QDRANT_URL and not DB_PATH.exists():
        stats["error"] = "Qdrant path missing"
        return stats
    try:
        client = get_qdrant_client()
        exists = client.collection_exists(collection_name=COLLECTION_NAME)
        stats["exists"] = exists
        if not exists:
            return stats
        info = client.get_collection(collection_name=COLLECTION_NAME)
        points = int(info.points_count or 0)
        stats["points"] = points
        stats["ready"] = points > 0
        return stats
    except Exception as error:
        stats["error"] = str(error)
        return stats


def search_evidence(
    question: str, limit: int = 10, score_threshold: float | None = None
) -> list[dict]:
    client = get_qdrant_client()
    if not client.collection_exists(collection_name=COLLECTION_NAME):
        raise RuntimeError("Build the evidence index before searching.")

    embedder = OllamaEmbeddings(model=EMBED_MODEL)
    query_vector = embedder.embed_query(question)

    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        with_payload=True,
        score_threshold=score_threshold,
        limit=limit,
    )
    hits = response.points

    results = []
    for hit in hits:  # each hit is ScoredPoint
        payload = dict(hit.payload or {})
        payload["score"] = float(hit.score)
        results.append(payload)
    return results


def hybrid_search(
    question: str,
    *,
    final_limit: int = 8,
    rerank_threshold: float | None = None,
    timings: dict | None = None,
    on_stage=None,
) -> list[dict]:
    """
    Dense + BM25 → RRF → rerank → optional rerank cut → source diversity.

    Keeps `search_evidence` as the dense-only baseline. Pass an empty `timings`
    dict to collect per-stage seconds for review telemetry.

    `rerank_threshold`: keep passages with `rerank_score >=` this value.
    `None` means no cut (eval / raw probes). Empty after the cut is a valid
    pack — callers should abstain.

    `on_stage(stage, message)` is optional UI/CLI progress (same idea as index build).
    """
    stage_timings = timings if timings is not None else {}

    def _emit(stage: str, message: str) -> None:
        if on_stage is not None:
            on_stage(stage, message)

    _emit("retrieve_dense", "Dense vector search…")
    started = perf_counter()
    dense = search_evidence(question, limit=20, score_threshold=None)
    stage_timings["dense_seconds"] = perf_counter() - started
    _emit(
        "retrieve_dense",
        f"Dense done · {len(dense)} hits · {stage_timings['dense_seconds']:.2f}s",
    )

    _emit("retrieve_bm25", "BM25 lexical search…")
    started = perf_counter()
    lexical = search_bm25(get_bm25_index(), question, limit=20)
    stage_timings["bm25_seconds"] = perf_counter() - started
    _emit(
        "retrieve_bm25",
        f"BM25 done · {len(lexical)} hits · {stage_timings['bm25_seconds']:.2f}s",
    )

    _emit("retrieve_fusion", "Reciprocal rank fusion…")
    started = perf_counter()
    fused = reciprocal_rank_fusion([dense, lexical], limit=25)
    stage_timings["fusion_seconds"] = perf_counter() - started
    _emit(
        "retrieve_fusion",
        f"Fused to {len(fused)} · {stage_timings['fusion_seconds']:.2f}s",
    )

    _emit("retrieve_rerank", "Cross-encoder rerank…")
    started = perf_counter()
    reranked = rerank(question, fused, limit=15)
    stage_timings["rerank_seconds"] = perf_counter() - started

    if rerank_threshold is None:
        after_cut = reranked
        cut_note = "no score cut"
    else:
        after_cut = [
            item
            for item in reranked
            if isinstance(item.get("rerank_score"), (int, float))
            and float(item["rerank_score"]) >= rerank_threshold
        ]
        cut_note = f"kept {len(after_cut)}/{len(reranked)} at ≥{rerank_threshold:g}"
    stage_timings["n_after_rerank_cut"] = len(after_cut)
    stage_timings["rerank_threshold"] = rerank_threshold
    _emit(
        "retrieve_rerank",
        f"Rerank done · {cut_note} · {stage_timings['rerank_seconds']:.2f}s",
    )

    _emit("retrieve_diversity", "Source diversity cap…")
    started = perf_counter()
    selected = diversify_by_source(
        after_cut,
        limit=final_limit,
        max_per_source=2,
    )
    stage_timings["diversity_seconds"] = perf_counter() - started
    stage_timings["n_dense"] = len(dense)
    stage_timings["n_bm25"] = len(lexical)
    stage_timings["n_fused"] = len(fused)
    stage_timings["n_reranked"] = len(reranked)
    stage_timings["n_final"] = len(selected)
    _emit(
        "retrieve_done",
        f"Evidence pack ready · {len(selected)} passages · "
        f"{stage_timings['diversity_seconds']:.2f}s diversify",
    )
    return selected


# Alias used in the learning guide.
search_evidence_hybrid = hybrid_search

# Re-exports so existing `from retrieval import ...` call sites keep working.
__all__ = [
    "CHUNK_MIN_CHARS",
    "CHUNK_OVERLAP",
    "CHUNK_SIZE",
    "COLLECTION_NAME",
    "DB_PATH",
    "DOCS_PATH",
    "EMBED_BATCH_SIZE",
    "EMBED_MODEL",
    "LEGACY_DOCS_PATH",
    "QDRANT_URL",
    "SEMANTIC_CACHE_COLLECTION",
    "SEMANTIC_CACHE_THRESHOLD",
    "UPSERT_BATCH_SIZE",
    "build_index",
    "check_semantic_cache",
    "clear_index",
    "contract_sections",
    "corpus_file_count",
    "embed_text",
    "get_bm25_index",
    "get_index_stats",
    "get_qdrant_client",
    "hybrid_search",
    "index_exists",
    "load_seed_records",
    "load_uploaded_records",
    "parse_clause_heading",
    "qdrant_target",
    "records_from_bytes",
    "recover_qdrant_client",
    "reset_qdrant_client",
    "search_evidence",
    "search_evidence_hybrid",
    "store_semantic_cache",
]
