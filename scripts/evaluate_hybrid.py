"""Evaluate dense + BM25 fused with RRF (Part 5).

  python scripts/evaluate_hybrid.py
  python scripts/evaluate_hybrid.py --limit 10
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chunking import load_seed_records  # noqa: E402
from evaluation.case_utils import expects_retrieval_hit  # noqa: E402
from hybrid import reciprocal_rank_fusion  # noqa: E402
from lexical_retrieval import build_bm25, search_bm25  # noqa: E402
from retrieval import search_evidence  # noqa: E402


def reciprocal_rank(results: list[dict], expected_sources: set[str]) -> float:
    for rank, result in enumerate(results, start=1):
        if result["source"] in expected_sources:
            return 1.0 / rank
    return 0.0


def source_recalled(results: list[dict], expected_sources: set[str]) -> bool:
    return any(result["source"] in expected_sources for result in results)


def evaluate_case(case: dict, bm25_index, limit: int, candidate_limit: int) -> dict:
    started = perf_counter()
    dense = search_evidence(
        case["question"],
        limit=candidate_limit,
        score_threshold=None,
    )
    lexical = search_bm25(bm25_index, case["question"], limit=candidate_limit)
    fused = reciprocal_rank_fusion([dense, lexical], limit=limit)
    elapsed = perf_counter() - started
    expected = set(case["expected_sources"])
    return {
        "id": case["id"],
        "question": case["question"],
        "recall": int(source_recalled(fused, expected)),
        "reciprocal_rank": reciprocal_rank(fused, expected),
        "latency_seconds": elapsed,
        "returned_sources": [item["source"] for item in fused],
        "n_dense": len(dense),
        "n_bm25": len(lexical),
        "n_fused": len(fused),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path("evaluation/retrieval_cases.json"),
    )
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument(
        "--candidates",
        type=int,
        default=20,
        help="How many hits each of dense/BM25 contribute before fusion",
    )
    args = parser.parse_args()

    records = load_seed_records()
    print(f"[hybrid] BM25 over {len(records)} chunks", flush=True)
    bm25_index = build_bm25(records)

    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    scored = [case for case in cases if expects_retrieval_hit(case)]
    rows = [
        evaluate_case(case, bm25_index, args.limit, args.candidates)
        for case in scored
    ]

    n = len(rows)
    recall = sum(row["recall"] for row in rows) / n if n else 0.0
    mrr = sum(row["reciprocal_rank"] for row in rows) / n if n else 0.0
    latency = sum(row["latency_seconds"] for row in rows) / n if n else 0.0

    print(
        f"Scored {n}/{len(cases)} cases "
        f"(skipped {len(cases) - n} expect_hit=false)"
    )
    print(f"Hybrid Recall@{args.limit}: {recall:.3f}")
    print(f"Hybrid MRR@{args.limit}: {mrr:.3f}")
    print(f"Hybrid Mean Latency: {latency:.3f}")
    for row in rows:
        print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
