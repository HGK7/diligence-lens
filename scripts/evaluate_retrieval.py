from __future__ import annotations

import argparse
import json

from pathlib import Path
from time import perf_counter
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from evaluation.case_utils import expects_retrieval_hit  # noqa: E402
from retrieval import search_evidence


def reciprocal_rank(results: list[dict], expected_sources: set[str]) -> float:
    
    for rank, result in enumerate(results, start=1):
        
        if result["source"] in expected_sources:
            
            return 1.0 / rank

    return 0.0

def source_recalled(results: list[dict], expected_sources: set[str]) -> bool:
    
    # Return true, if expected source present in any of the results
    return any(result["source"] in expected_sources for result in results)

def evaluate_case(case: dict, limit: int) -> dict:
    
    started = perf_counter()
    results = search_evidence(
        case["question"],
        limit=limit,
        score_threshold=None,
    )
    elapsed = perf_counter() - started
    expected = set(case["expected_sources"])
    
    return{
        
        "id": case["id"],
        "question": case["question"],
        "recall": int(source_recalled(results, expected)),
        "reciprocal_rank": reciprocal_rank(results, expected),
        "latency_seconds" : elapsed,
        "returned_sources": [item["source"] for item in results]
    }    
    
    
def main() -> None:
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        
        "--cases",
        type=Path,
        default=Path("evaluation/retrieval_cases.json")
    )
    
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()
    
    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    scored_cases = [case for case in cases if expects_retrieval_hit(case)]
    boundary_cases = [case for case in cases if not expects_retrieval_hit(case)]
    rows = [evaluate_case(case, args.limit) for case in scored_cases]

    n = len(rows)
    recall = sum(row["recall"] for row in rows) / n if n else 0.0
    mrr = sum(row["reciprocal_rank"] for row in rows) / n if n else 0.0
    latency = sum(row["latency_seconds"] for row in rows) / n if n else 0.0

    print(f"Scored retrieval cases: {n} (skipped {len(boundary_cases)} expect_hit=false)")
    print(f"Recall@{args.limit}: {recall:.3f}")
    print(f"MRR@{args.limit}: {mrr:.3f}")
    print(f"Mean Latency: {latency:.3f}")

    for row in rows:
        print(json.dumps(row, ensure_ascii=False))
        print(" ")
    if boundary_cases:
        print("Boundary / negative cases (not in Recall/MRR):")
        for case in boundary_cases:
            print(
                json.dumps(
                    {
                        "id": case["id"],
                        "category": case.get("category"),
                        "question": case["question"],
                        "expected_decision": case.get("expected_decision"),
                    },
                    ensure_ascii=False,
                )
            )
        

if __name__ == "__main__":
    main()
    
