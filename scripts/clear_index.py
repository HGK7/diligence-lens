"""Wipe diligence + semantic-cache store.

Path mode: stop Streamlit first — only one process may hold data/qdrant.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from retrieval import clear_index, get_index_stats  # noqa: E402


def main() -> None:
    result = clear_index()
    stats = get_index_stats()
    print(result, flush=True)
    print("stats after clear:", stats, flush=True)
    points = int(stats.get("points") or 0)
    if points != 0 and not stats.get("error"):
        raise SystemExit(
            f"Clear did not reach 0 points (got {points}). "
            "Stop Streamlit / other Qdrant users and retry."
        )


if __name__ == "__main__":
    main()
