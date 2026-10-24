from __future__ import annotations

import atexit
import os
import threading
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from langchain_ollama import OllamaEmbeddings
from pypdf import PdfReader
from qdrant_client import QdrantClient, models

# Local default: embedded path-mode Qdrant under ./data/qdrant.
# Compose/server: set QDRANT_URL=http://qdrant:6333 (see docs/DOCKER.md).
QDRANT_URL = os.getenv("QDRANT_URL", "").strip()
DB_PATH = Path(os.getenv("QDRANT_PATH", "data/qdrant"))
# Active diligence corpus for Diligence Lens (CUAD subset exports land here).
DOCS_PATH = Path("dataroom")
# Legacy Nextwork seed folder; still accepted if present during migration.
LEGACY_DOCS_PATH = Path("vendor_docs")
COLLECTION_NAME = "diligence_dataroom"
# Pin one embedding space for evidence and cached questions.

EMBED_MODEL = "nomic-embed-text:v1.5"
SEMANTIC_CACHE_COLLECTION = "semantic_cache"
SEMANTIC_CACHE_THRESHOLD = 0.95

# Steady chunking defaults: section-first, then paragraph windows.
# Tuned for ~50-100 contract docs without blowing 4 GB embedding VRAM.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
CHUNK_MIN_CHARS = 80
# Local Qdrant path-mode is a single-process server; batch upserts keep it stable.
UPSERT_BATCH_SIZE = int(os.getenv("UPSERT_BATCH_SIZE", "128"))
# nomic-embed is tiny (~335 MB) and already 100% GPU in `ollama ps`.
# Batch 8 under-fed an RTX 3050; 32 is a better default. Raise via EMBED_BATCH_SIZE
# (try 48/64). If WinError 10054 returns, drop back toward 16.
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "32"))

_client: QdrantClient | None = None
_client_lock = threading.Lock()


def qdrant_target() -> str:
    """Human-readable store location for logs / sidebar."""
    return QDRANT_URL or str(DB_PATH)


def get_qdrant_client() -> QdrantClient:
    """
    One Qdrant client per Python interpreter.

    - QDRANT_URL set → HTTP server (Compose / Cloud). Preferred for >20k points.
    - else path mode under DB_PATH. Do not open/close per Streamlit rerun or the
      embedded server dies (dial tcp … connection refused / WinError 32 on .lock).
    """
    global _client
    with _client_lock:
        if _client is not None:
            return _client
        if QDRANT_URL:
            _client = QdrantClient(url=QDRANT_URL)
            return _client
        DB_PATH.mkdir(parents=True, exist_ok=True)
        try:
            _client = QdrantClient(path=str(DB_PATH))
        except Exception:
            lock_path = DB_PATH / ".lock"
            if lock_path.exists():
                lock_path.unlink(missing_ok=True)
            _client = QdrantClient(path=str(DB_PATH))
        return _client


def reset_qdrant_client() -> None:
    """Close the shared client (e.g. before deleting the on-disk store)."""
    global _client
    with _client_lock:
        if _client is not None:
            try:
                _client.close()
            except Exception:
                pass
            _client = None


def clear_index() -> dict:
    """
    Delete diligence + semantic-cache collections (or wipe path store).

    Call this instead of stacking rebuilds on a bloated collection.
    """
    deleted: list[str] = []
    if QDRANT_URL:
        client = get_qdrant_client()
        for name in (COLLECTION_NAME, SEMANTIC_CACHE_COLLECTION):
            if client.collection_exists(collection_name=name):
                client.delete_collection(collection_name=name)
                deleted.append(name)
        return {
            "mode": "url",
            "deleted": deleted,
            "path": QDRANT_URL,
            "points": 0,
        }

    reset_qdrant_client()
    if DB_PATH.exists():
        import shutil

        shutil.rmtree(DB_PATH, ignore_errors=False)
    DB_PATH.mkdir(parents=True, exist_ok=True)
    return {
        "mode": "path",
        "deleted": [COLLECTION_NAME, SEMANTIC_CACHE_COLLECTION],
        "path": str(DB_PATH),
        "points": 0,
    }


atexit.register(reset_qdrant_client)


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


def _normalize_text(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.splitlines()).strip()


def _split_paragraphs(text: str) -> list[str]:
    blocks = []
    current: list[str] = []
    for line in text.splitlines():
        if not line.strip():
            if current:
                blocks.append("\n".join(current).strip())
                current = []
            continue
        current.append(line)
    if current:
        blocks.append("\n".join(current).strip())
    return [block for block in blocks if block]


def _window_chunk(
    text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> list[str]:
    """Character windows that prefer breaking near sentence ends."""
    normalized = _normalize_text(text)
    if not normalized:
        return []
    if len(normalized) <= size:
        return [normalized]

    output: list[str] = []
    start = 0
    length = len(normalized)
    while start < length:
        end = min(length, start + size)
        if end < length:
            window = normalized[start:end]
            # Prefer a sentence boundary in the last third of the window.
            search_from = max(0, len(window) - size // 3)
            best = -1
            for marker in (". ", "? ", "! ", ";\n", ".\n"):
                idx = window.rfind(marker, search_from)
                if idx > best:
                    best = idx + len(marker.strip())
            if best >= size // 2:
                end = start + best
        piece = normalized[start:end].strip()
        if len(piece) >= CHUNK_MIN_CHARS:
            output.append(piece)
        if end >= length:
            break
        start = max(0, end - overlap)
    return output


def _chunks(
    text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> list[str]:
    """
    Steady strategy:
    1. Pack whole paragraphs while under size.
    2. Split oversized paragraphs with sentence-aware windows.
    """
    normalized = _normalize_text(text)
    if not normalized:
        return []

    packed: list[str] = []
    buffer = ""
    for paragraph in _split_paragraphs(normalized):
        if len(paragraph) > size:
            if buffer:
                packed.append(buffer)
                buffer = ""
            packed.extend(_window_chunk(paragraph, size=size, overlap=overlap))
            continue
        candidate = paragraph if not buffer else f"{buffer}\n\n{paragraph}"
        if len(candidate) <= size:
            buffer = candidate
        else:
            if buffer:
                packed.append(buffer)
            buffer = paragraph
    if buffer:
        packed.append(buffer)
    return [item for item in packed if item]


def _markdown_sections(text: str) -> list[tuple[str, str]]:
    sections = []
    title = "Document"
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        # Markdown headings and ALL-CAPS contract section labels.
        is_heading = stripped.startswith("#") or (
            len(stripped) <= 80
            and stripped.isupper()
            and any(ch.isalpha() for ch in stripped)
            and not stripped.startswith("HTTP")
        )
        if is_heading and stripped.startswith("#"):
            if lines:
                sections.append((title, "\n".join(lines).strip()))
            title = stripped.lstrip("#").strip() or "Document"
            lines = []
        elif is_heading and stripped.isupper() and lines:
            sections.append((title, "\n".join(lines).strip()))
            title = stripped.title()
            lines = []
        else:
            lines.append(line)
    if lines:
        sections.append((title, "\n".join(lines).strip()))
    return [(name, body) for name, body in sections if body]


def _record_chunks(
    text: str, source: str, section: str, page: int | None = None
) -> list[dict]:
    records = []
    for chunk_index, chunk in enumerate(_chunks(text), start=1):
        records.append(
            {
                "source": source,
                "section": section,
                "page": page,
                "chunk_index": chunk_index,
                "text": chunk,
            }
        )
    return records


def records_from_bytes(name: str, data: bytes) -> list[dict]:
    source = Path(name).name
    suffix = Path(source).suffix.lower()
    records = []
    if suffix == ".pdf":
        reader = PdfReader(BytesIO(data))
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            records.extend(
                _record_chunks(text, source, f"Page {page_number}", page_number)
            )
        return records

    text = data.decode("utf-8", errors="replace")
    if suffix == ".md":
        for section, body in _markdown_sections(text):
            records.extend(_record_chunks(body, source, section))
    else:
        # Plain contracts: try section labels, else whole-document packing.
        sections = _markdown_sections(text)
        if len(sections) > 1:
            for section, body in sections:
                records.extend(_record_chunks(body, source, section))
        else:
            records.extend(_record_chunks(text, source, "Document"))
    return records


def _iter_corpus_files(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    files = []
    for path in sorted(folder.rglob("*")):
        if not path.is_file():
            continue
        # Skip nested dataset archives / raw dumps; only index the active room.
        if "contract-nli" in path.parts or "raw" in path.parts:
            continue
        if path.suffix.lower() == ".json":
            continue
        if path.name.upper() in {"LICENSE", "LICENSE.TXT", "NOTICE", "README.MD"}:
            continue
        if path.suffix.lower() in {".md", ".txt", ".pdf"}:
            files.append(path)
    return files


def load_seed_records() -> list[dict]:
    """Load the active dataroom first; fall back to legacy vendor_docs if empty."""
    records = []
    primary = _iter_corpus_files(DOCS_PATH)
    if primary:
        root = DOCS_PATH
        sources = primary
    else:
        root = LEGACY_DOCS_PATH
        sources = _iter_corpus_files(root)
    for path in sources:
        try:
            label = str(path.relative_to(root)).replace("\\", "/")
        except ValueError:
            label = path.name
        records.extend(records_from_bytes(label, path.read_bytes()))
    return records


def load_uploaded_records(uploaded_files) -> list[dict]:
    records = []
    for uploaded_file in uploaded_files or []:
        records.extend(records_from_bytes(uploaded_file.name, uploaded_file.getvalue()))
    return records


def corpus_file_count() -> int:
    primary = _iter_corpus_files(DOCS_PATH)
    if primary:
        return len(primary)
    return len(_iter_corpus_files(LEGACY_DOCS_PATH))


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
        progress_callback(
            stage=stage, current=current, total=total, message=message
        )


def build_index(records: list[dict], progress_callback=None) -> dict:
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
    texts = [record["text"] for record in records]
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
        # Always wipe both collections before create — stacking upserts on an
        # old ID scheme caused 49859+49918=99777 points.
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
                "Stop other Streamlit processes and use Clear index."
            )

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
            payload = dict(record)
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
            client.upsert(
                collection_name=COLLECTION_NAME, points=batch, wait=True
            )
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
