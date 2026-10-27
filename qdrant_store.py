from __future__ import annotations

import atexit
import os
import sys
import threading
import time
import types
from pathlib import Path

from qdrant_client import QdrantClient

# Local default: embedded path-mode Qdrant under ./data/qdrant.
# Compose/server: set QDRANT_URL=http://qdrant:6333 (see docs/DOCKER.md).
QDRANT_URL = os.getenv("QDRANT_URL", "").strip()
DB_PATH = Path(os.getenv("QDRANT_PATH", "data/qdrant"))
COLLECTION_NAME = "diligence_dataroom"
SEMANTIC_CACHE_COLLECTION = "semantic_cache"

# Streamlit re-imports project modules on hot reload and clears module globals.
# Keep the live client on a synthetic sys.modules entry that survives reload so
# we never open a second path-mode client against the same folder.
_HOLDER_NAME = "_diligence_lens_qdrant_holder"


def _holder() -> types.ModuleType:
    mod = sys.modules.get(_HOLDER_NAME)
    if mod is None:
        mod = types.ModuleType(_HOLDER_NAME)
        mod.client = None  # type: ignore[attr-defined]
        mod.lock = threading.Lock()  # type: ignore[attr-defined]
        sys.modules[_HOLDER_NAME] = mod
    return mod


def qdrant_target() -> str:
    """Human-readable store location for logs / sidebar."""
    return QDRANT_URL or str(DB_PATH)


def _is_lock_error(error: BaseException) -> bool:
    text = str(error).lower()
    return (
        "already accessed" in text
        or "lock" in text
        or "winerror 32" in text
        or "being used by another process" in text
    )


def _open_path_client() -> QdrantClient:
    """
    Open embedded Qdrant once for this process.

    Path mode uses an exclusive folder lock. A stale `.lock` after a crash can
    block reopen; a live second process must not be fought by deleting the lock.
    """
    DB_PATH.mkdir(parents=True, exist_ok=True)
    try:
        return QdrantClient(path=str(DB_PATH))
    except Exception as first:
        if not _is_lock_error(first):
            raise
        # Brief wait: Streamlit reload may still be closing the prior handle.
        time.sleep(0.2)
        try:
            return QdrantClient(path=str(DB_PATH))
        except Exception as second:
            if not _is_lock_error(second):
                raise
            lock_path = DB_PATH / ".lock"
            if not lock_path.exists():
                raise RuntimeError(
                    f"Cannot open path Qdrant at `{DB_PATH}`: {second}. "
                    "Stop every other `streamlit run` / rebuild using this folder."
                ) from second
            # Stale lock after an unclean exit — only safe once retries failed
            # and we do not hold a client in this process.
            try:
                lock_path.unlink()
            except OSError as unlink_error:
                raise RuntimeError(
                    f"Cannot open path Qdrant at `{DB_PATH}` (lock held): {second}. "
                    "Stop other Streamlit/Python processes using this folder, "
                    "then retry. Do not delete `.lock` while another process "
                    f"still holds it ({unlink_error})."
                ) from second
            time.sleep(0.1)
            try:
                return QdrantClient(path=str(DB_PATH))
            except Exception as third:
                raise RuntimeError(
                    f"Cannot open path Qdrant at `{DB_PATH}` after clearing "
                    f"stale lock: {third}. Stop other processes and retry."
                ) from third


def get_qdrant_client() -> QdrantClient:
    """
    One Qdrant client per Python process (reload-safe).

    - QDRANT_URL set → HTTP server (Compose / Cloud). Preferred for >20k points.
    - else path mode under DB_PATH. Never open a second path client for the same
      folder (Streamlit hot reload used to reset module globals and deadlock).
    """
    h = _holder()
    with h.lock:
        if h.client is not None:
            return h.client
        if QDRANT_URL:
            h.client = QdrantClient(url=QDRANT_URL)
            return h.client
        h.client = _open_path_client()
        return h.client


def reset_qdrant_client() -> None:
    """Close the shared client (e.g. before deleting the on-disk store)."""
    h = _holder()
    with h.lock:
        if h.client is not None:
            try:
                h.client.close()
            except Exception:
                pass
            h.client = None


def recover_qdrant_client() -> QdrantClient:
    """
    Close and reopen the shared client.

    Use from the portal when the sidebar shows a lock / connection error after
    a hot reload or a failed clear.
    """
    reset_qdrant_client()
    time.sleep(0.15)
    return get_qdrant_client()


def _wipe_path_store() -> None:
    """
    Full wipe for local path-mode Qdrant.

    delete_collection alone is not enough when point IDs change (e.g. new
    section labels): upserts add alongside orphans and points balloon.
    Closing the client then removing DB_PATH is the reliable reset on Windows.
    """
    import shutil

    reset_qdrant_client()
    if not DB_PATH.exists():
        DB_PATH.mkdir(parents=True, exist_ok=True)
        return

    last_error: Exception | None = None
    for attempt in range(1, 6):
        try:
            shutil.rmtree(DB_PATH)
            last_error = None
            break
        except OSError as error:
            last_error = error
            time.sleep(0.4 * attempt)
    if last_error is not None:
        raise RuntimeError(
            f"Could not wipe `{DB_PATH}`: {last_error}. "
            "Stop Streamlit and any other process using this folder, then retry."
        ) from last_error
    DB_PATH.mkdir(parents=True, exist_ok=True)


def clear_index() -> dict:
    """
    Delete diligence + semantic-cache data.

    URL mode: delete named collections.
    Path mode: remove the on-disk store (avoids orphan-point stacking).
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
            "deleted": deleted or [COLLECTION_NAME, SEMANTIC_CACHE_COLLECTION],
            "path": QDRANT_URL,
            "points": 0,
        }

    _wipe_path_store()
    return {
        "mode": "path",
        "deleted": [COLLECTION_NAME, SEMANTIC_CACHE_COLLECTION],
        "path": str(DB_PATH),
        "points": 0,
    }


atexit.register(reset_qdrant_client)
