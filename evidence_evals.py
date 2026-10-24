"""Quick CLI: run fixed probe queries, print top hits, dump same run to a text file.

Flow:
1. For each string in CASES, call search_evidence (dense Qdrant search).
2. Print a short label per hit to the terminal.
3. Mirror the full run (query + source/score/section + chunk text) into
   evidence_eval_results.txt so you can skim retrieval quality offline.
"""

from pathlib import Path

from retrieval import search_evidence

CASES = [
    "Where is liability capped?",
    "Which agreements permit termination after a change of control?",
    "Find clauses about consequential damages.",
    "Which party owns developed intellectual property?",
    "Does Acme Corp appear in any agreement?",
    "Which contracts require written consent before assignment?",
]

OUT_PATH = Path(__file__).with_name("evidence_eval_results.txt")


def main() -> None:
    lines: list[str] = []

    for case in CASES:
        hits = search_evidence(case, limit=5, score_threshold=0.7)

        print(case)
        lines.append(case)
        lines.append("=" * len(case))

        if not hits:
            msg = "  (no hits above threshold)"
            print(msg)
            lines.append(msg)
            lines.append("")
            continue

        for i, hit in enumerate(hits, start=1):
            label = (
                f"  {i}. {hit['source']} · score {hit['score']:.3f}"
                f" · {hit.get('section') or '(no section)'}"
            )
            print(label)
            lines.append(label)
            # Full chunk text only in the file — terminal stays skim-friendly.
            lines.append(hit.get("text") or "")
            lines.append("")

        lines.append("")

    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
