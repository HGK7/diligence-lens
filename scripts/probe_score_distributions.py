"""Part 9 helper: print dense / fusion / rerank scores for gold cases.

Marks each hit as relevant (YES) if its source is in expected_sources.
Optional --threshold / --final-limit simulate a review evidence pack.

  python scripts/probe_score_distributions.py
  python scripts/probe_score_distributions.py --all
  python scripts/probe_score_distributions.py --all --threshold 2 --final-limit 8

Stop Streamlit first if Qdrant path mode holds the lock.
First run loads the MiniLM cross-encoder (slow once).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chunking import load_seed_records  # noqa: E402
from evaluation.case_utils import expects_retrieval_hit  # noqa: E402
from hybrid import reciprocal_rank_fusion  # noqa: E402
from lexical_retrieval import build_bm25, search_bm25  # noqa: E402
from reranking import get_reranker, rerank  # noqa: E402
from retrieval import search_evidence  # noqa: E402


def _fmt(value: object, digits: int) -> str:
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return "n/a"


def probe_case(
    case: dict,
    bm25_index,
    *,
    candidate_limit: int,
    limit: int,
    threshold: float | None,
    final_limit: int,
) -> dict:
    expected = set(case.get("expected_sources") or [])
    scored = expects_retrieval_hit(case)
    print("=" * 88)
    print(f"id: {case['id']}  category={case.get('category', '?')}  expect_hit={scored}")
    print(f"Q:  {case['question']}")
    if scored:
        print("expected sources:")
        for source in case["expected_sources"]:
            print(f"  - {source}")
    else:
        print(
            "expected sources: (none — boundary/negative case; "
            f"decision hint={case.get('expected_decision', 'n/a')})"
        )
    print()

    dense = search_evidence(
        case["question"],
        limit=candidate_limit,
        score_threshold=None,
    )
    lexical = search_bm25(bm25_index, case["question"], limit=candidate_limit)
    fused = reciprocal_rank_fusion([dense, lexical], limit=candidate_limit)
    ranked = rerank(case["question"], fused, limit=limit)

    header = (
        f"{'rank':<4} {'rel':<4} {'keep':<4} {'dense':>7} {'fusion':>8} "
        f"{'rerank':>8}  source (tail)"
    )
    print(header)
    print("-" * len(header))

    yes_ranks: list[int] = []
    kept: list[dict] = []
    for rank, hit in enumerate(ranked, start=1):
        relevant = hit["source"] in expected
        if relevant:
            yes_ranks.append(rank)
        rerank_score = hit.get("rerank_score")
        passes = threshold is None or (
            isinstance(rerank_score, float) and rerank_score >= threshold
        )
        if passes and len(kept) < final_limit:
            kept.append(hit)
            keep_label = "KEEP"
        else:
            keep_label = "drop"

        label = "YES" if relevant else "no"
        source_tail = hit["source"][-55:]
        print(
            f"{rank:<4} {label:<4} {keep_label:<4} "
            f"{_fmt(hit.get('score'), 3):>7} "
            f"{_fmt(hit.get('fusion_score'), 4):>8} "
            f"{_fmt(hit.get('rerank_score'), 3):>8}  "
            f"...{source_tail}"
        )

    kept_yes = sum(1 for hit in kept if hit["source"] in expected)
    kept_no = len(kept) - kept_yes
    gold_sources_in_pack = {hit["source"] for hit in kept if hit["source"] in expected}
    abstain = len(kept) == 0

    print()
    if yes_ranks:
        print(f"relevant hits in top-{limit}: ranks {yes_ranks}")
    else:
        print(f"relevant hits in top-{limit}: none")

    thr_label = "none" if threshold is None else f">={threshold:g}"
    print(
        f"pack (threshold {thr_label}, final_limit={final_limit}): "
        f"{len(kept)} kept "
        f"(YES chunks={kept_yes}, no chunks={kept_no}, "
        f"gold files in pack={len(gold_sources_in_pack)}/{len(expected)})"
    )
    if abstain:
        print("pack empty -> would ABSTAIN (insufficient_evidence)")
    print()

    return {
        "id": case["id"],
        "n_kept": len(kept),
        "kept_yes": kept_yes,
        "kept_no": kept_no,
        "gold_files_in_pack": len(gold_sources_in_pack),
        "gold_files_expected": len(expected),
        "abstain": abstain,
        "any_gold_in_top": bool(yes_ranks),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dump hybrid+rerank scores for Part 9 calibration."
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path("evaluation/retrieval_cases.json"),
    )
    parser.add_argument(
        "--case-id",
        default="governing-law-aaa-ny-01",
        help="Single case id (ignored when --all is set).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Probe every case in the JSON file.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="How many reranked rows to print / consider before packing.",
    )
    parser.add_argument("--candidates", type=int, default=20)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Keep rows with rerank_score >= this value. Omit = no cut.",
    )
    parser.add_argument(
        "--final-limit",
        type=int,
        default=8,
        help="Max passages kept for the simulated review pack (v1 default 8).",
    )
    args = parser.parse_args()

    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    if args.all:
        selected = cases
    else:
        selected = [case for case in cases if case["id"] == args.case_id]
        if not selected:
            known = ", ".join(case["id"] for case in cases)
            raise SystemExit(
                f"Unknown case id {args.case_id!r}. Known ids: {known}"
            )

    print("[probe] loading cross-encoder (once)...", flush=True)
    get_reranker()

    records = load_seed_records()
    print(f"[probe] BM25 over {len(records)} chunks", flush=True)
    bm25_index = build_bm25(records)
    thr_label = "none" if args.threshold is None else f">={args.threshold:g}"
    print(
        f"[probe] {len(selected)} case(s), "
        f"candidates={args.candidates}, print_limit={args.limit}, "
        f"threshold={thr_label}, final_limit={args.final_limit}",
        flush=True,
    )
    print()

    summaries = [
        probe_case(
            case,
            bm25_index,
            candidate_limit=args.candidates,
            limit=args.limit,
            threshold=args.threshold,
            final_limit=args.final_limit,
        )
        for case in selected
    ]

    if len(summaries) > 1:
        print("=" * 88)
        print("SUMMARY (simulated review packs)")
        positive = [
            row
            for row, case in zip(summaries, selected, strict=True)
            if expects_retrieval_hit(case)
        ]
        negative = [
            row
            for row, case in zip(summaries, selected, strict=True)
            if not expects_retrieval_hit(case)
        ]
        abstains = sum(1 for row in summaries if row["abstain"])
        with_gold = sum(1 for row in positive if row["gold_files_in_pack"] > 0)
        print(f"cases: {len(summaries)} (retrieval-scored={len(positive)}, boundary={len(negative)})")
        if positive:
            print(f"scored packs with ≥1 gold file: {with_gold}/{len(positive)}")
        print(f"empty packs (abstain): {abstains}/{len(summaries)}")
        print()
        print(f"{'id':<36} {'kept':>4} {'YES':>4} {'no':>4} {'gold':>7} abstain")
        for row in summaries:
            gold = f"{row['gold_files_in_pack']}/{row['gold_files_expected']}"
            print(
                f"{row['id']:<36} {row['n_kept']:>4} {row['kept_yes']:>4} "
                f"{row['kept_no']:>4} {gold:>7} "
                f"{'yes' if row['abstain'] else 'no'}"
            )


if __name__ == "__main__":
    main()
