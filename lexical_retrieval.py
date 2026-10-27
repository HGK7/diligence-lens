from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from rank_bm25 import BM25Okapi

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-'.]*")


def lexical_tokens(text: str) -> list[str]:
    return [token.casefold() for token in TOKEN_PATTERN.findall(text)]


@dataclass
class BM25Index:
    records: list[dict]
    model: BM25Okapi


def build_bm25(records: list[dict]) -> BM25Index:
    tokenized_corpus = [
        lexical_tokens(
            " ".join(
                [
                    record.get("section", ""),
                    record["text"],
                ]
            )
        )
        for record in records
    ]
    return BM25Index(
        records=records,
        model=BM25Okapi(tokenized_corpus),
    )


@lru_cache(maxsize=1)
def get_bm25_index() -> BM25Index:
    """Build BM25 once per process from the active dataroom chunks."""
    from chunking import load_seed_records

    records = load_seed_records()
    if not records:
        raise RuntimeError("No dataroom records for BM25. Export/build the corpus first.")
    return build_bm25(records)


def search_bm25(
    index: BM25Index,
    question: str,
    limit: int = 20,
) -> list[dict]:
    scores = index.model.get_scores(lexical_tokens(question))
    best_indices = np.argsort(scores)[::-1][:limit]

    results = []
    for rank, corpus_index in enumerate(best_indices, start=1):
        score = float(scores[corpus_index])
        if score <= 0:
            continue
        result = dict(index.records[int(corpus_index)])
        result["bm25_score"] = score
        result["bm25_rank"] = rank
        results.append(result)
    return results
