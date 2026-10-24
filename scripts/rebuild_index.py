"""
CLI rebuild for Diligence Lens (no Streamlit).

  python scripts/rebuild_index.py
  python scripts/clear_index.py   # optional wipe first
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from retrieval import build_index, get_index_stats, load_seed_records  # noqa: E402


def main() -> None:
    records = load_seed_records()
    print(f"[prepare] loaded {len(records)} chunks", flush=True)
    if not records:
        raise SystemExit("No dataroom records. Run scripts/export_cuad.py first.")

    def on_progress(*, stage: str, current: int, total: int, message: str) -> None:
        pass  # console already gets _emit prints

    summary = build_index(records, progress_callback=on_progress)
    stats = get_index_stats()
    print(
        f"[verify] embedded={summary['records']} points={stats['points']} "
        f"ready={stats['ready']}",
        flush=True,
    )
    if stats["points"] != summary["records"]:
        raise SystemExit(
            f"Point mismatch: {stats['points']} != {summary['records']}"
        )


if __name__ == "__main__":
    main()
