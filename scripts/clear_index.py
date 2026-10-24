"""Wipe diligence + semantic-cache store. Stop Streamlit first (path-mode lock)."""
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
    if stats.get("points", 0) != 0 and not stats.get("error"):
        raise SystemExit("Clear did not reach 0 points")


if __name__ == "__main__":
    main()
