"""
CLI rebuild for Diligence Lens (no Streamlit).

  python scripts/rebuild_index.py
  python scripts/rebuild_index.py --clear   # wipe store, verify empty, then rebuild
  python scripts/clear_index.py             # wipe only
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from retrieval import (  # noqa: E402
    build_index,
    clear_index,
    get_index_stats,
    load_seed_records,
)


def _require_empty_store() -> None:
    stats = get_index_stats()
    points = int(stats.get("points") or 0)
    if points != 0:
        raise SystemExit(
            f"Store not empty after clear: points={points} "
            f"(path={stats.get('path')}). Stop Streamlit and retry."
        )
    print(f"[clear] verified empty · {stats.get('path')}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild the diligence evidence index from dataroom/."
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Wipe Qdrant (path rmtree or URL collection delete), verify 0 points, "
        "then rebuild. Use when points >> embedded after a chunking change.",
    )
    parser.add_argument(
        "--no-wipe",
        action="store_true",
        help="Skip build_index's default pre-upsert wipe (dangerous; for debug only).",
    )
    args = parser.parse_args()

    if args.clear:
        result = clear_index()
        print(f"[clear] {result}", flush=True)
        _require_empty_store()

    records = load_seed_records()
    print(f"[prepare] loaded {len(records)} chunks", flush=True)
    if not records:
        raise SystemExit("No dataroom records. Run scripts/export_cuad.py first.")

    def on_progress(*, stage: str, current: int, total: int, message: str) -> None:
        pass  # console already gets _emit prints

    summary = build_index(
        records,
        progress_callback=on_progress,
        wipe=not args.no_wipe,
    )
    stats = get_index_stats()
    print(
        f"[verify] embedded={summary['records']} points={stats['points']} "
        f"ready={stats['ready']}",
        flush=True,
    )
    if stats["points"] != summary["records"]:
        raise SystemExit(
            f"Point mismatch: {stats['points']} != {summary['records']}. "
            "Re-run with: python scripts/rebuild_index.py --clear"
        )


if __name__ == "__main__":
    main()
