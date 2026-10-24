# Diligence Lens — Retrieval Quality Learning Guide

This is a hands-on workbook, not a copy-paste migration plan.

You will improve retrieval in small experiments:

1. Measure the current dense-search baseline.
2. Inspect and repair contract chunking.
3. Add lexical BM25 retrieval.
4. Combine lexical and dense results with Reciprocal Rank Fusion.
5. Rerank the fused candidates.
6. Add source diversity.
7. Compare every stage using the same questions.

Do not implement all parts at once. If you change chunking, hybrid search, and
reranking together, you will not know which change helped.

## Current system, in plain English

The current retrieval flow is:

```text
question
  -> Ollama converts question to a dense vector
  -> Qdrant finds nearby chunk vectors
  -> score threshold removes weak matches
  -> top 6 chunks go to Gemini
  -> Gemini creates a structured finding
  -> exact citation quotes are checked against retrieved text
```

Relevant code:

- `retrieval.py::_chunks` builds chunks.
- `retrieval.py::_markdown_sections` guesses section boundaries.
- `retrieval.py::build_index` embeds and stores chunks.
- `retrieval.py::search_evidence` performs dense vector retrieval.
- `risk_graph.py::retrieve_evidence_node` selects evidence for a review.
- `risk_graph.py::_validate_citations` verifies generated quotes locally.

The current chunks are not completely arbitrary. They preserve paragraphs where
possible and prefer sentence endings for oversized paragraphs. The weakness is
that the parser does not reliably understand legal clauses. An ALL-CAPS line may
be treated as a heading even when it is an identifier, exhibit artifact, or OCR
noise. Meaningful clauses may also be split only because they exceed a character
budget.

---

# Part 0 — Protect the baseline

## Goal

Capture the current behavior before changing it.

Your clean learning corpus currently has approximately:

- 80 CUAD contracts
- 7,840 indexed chunks
- dense retrieval through `nomic-embed-text:v1.5`
- a default review threshold of `0.60`

Do not clear the index just to start this guide. You need the current index as the
baseline. A chunking change will require a rebuild later.

## Task

Run 10–15 representative searches in the **Evidence search** tab. Record:

- the question;
- top result filenames;
- top scores;
- which result actually answers the question;
- examples of broken or context-free chunks;
- retrieval latency.

Include several query shapes:

```text
Where is liability capped?
Which agreements permit termination after a change of control?
Find clauses about consequential damages.
Which party owns developed intellectual property?
Does Acme Corp appear in any agreement?
Which contracts require written consent before assignment?
```

The company-name example matters because semantic embeddings and exact terms have
different strengths.

## Glossary

### Baseline

A baseline is the current measured system. It gives you a result to beat.

“The new search feels better” is not a useful comparison. “Recall@5 increased
from 0.60 to 0.82 on the same 20 questions” is useful.

### Dense retrieval

Dense retrieval represents text as a fixed-length vector of floating-point
numbers. Similar meanings should produce nearby vectors even when wording differs.

Example:

```text
"ending the agreement after ownership changes"
"termination following a change of control"
```

These sentences use different words but may be close in embedding space.

Dense retrieval is good at:

- synonyms;
- paraphrases;
- broad concepts;
- natural-language questions.

It may be weak at:

- exact names;
- clause numbers;
- rare abbreviations;
- exact legal phrases;
- numbers and dates.

### Embedding

An embedding is the numeric vector produced from text. In this project Ollama
runs `nomic-embed-text:v1.5`, producing 768-dimensional vectors.

“768 dimensions” means each text becomes a list of 768 numbers. You do not assign
meaning to individual dimensions. The model learned the geometry.

### Similarity score

Qdrant compares vectors using cosine similarity. Higher generally means more
semantically similar.

A score is not a probability. A score of `0.70` does not mean “70% correct.”
Thresholds must be calibrated against your own corpus and questions.

## Checkpoint

Before Part 1, you should be able to explain:

1. Why dense retrieval can find a paraphrase.
2. Why it can miss an exact company name.
3. Why a similarity threshold is not a confidence percentage.

---

# Part 1 — Build a retrieval evaluation harness

## Why this comes first

Every later feature creates more knobs:

- chunk size;
- overlap;
- BM25 parameters;
- number of dense candidates;
- number of lexical candidates;
- fusion constant;
- reranker candidate count;
- final evidence count.

An evaluation harness prevents tuning these knobs based on one memorable query.

## The eval loop (do these in order)

```text
1. Probe   — dump top hits for a few questions (human-readable)
2. Label   — open real contracts; write questions FROM the text
3. Gold    — put exact indexed `source` filenames into JSON
4. Score   — run the harness; freeze Recall@K / MRR@K / latency
5. Read    — inspect every 0-row; do not "fix" labels to chase a score
6. Freeze  — save numbers before any chunking / hybrid / rerank change
```

Skipping step 2–3 is the most common failure mode. Placeholder filenames from a
tutorial produce **Recall = 0** even when retrieval is fine.

---

## Task 1.0 — Quick probe dump (optional but useful)

Before writing gold labels, print what dense search actually returns.

A small CLI (for example `evidence_evals.py`) should:

1. Call `search_evidence(question, limit=5, score_threshold=...)`.
2. Print the question and short hit labels to the terminal.
3. Write the **full chunk text** to a sibling `.txt` file for offline reading.

Shape reminder: `search_evidence` returns a **list of hit dicts**. Loop
`for hit in hits`, not `for result in results` where each `result` is already a
list. Nested lists cause `TypeError: list indices must be integers or slices,
not str` when you do `result["source"]`.

Qdrant path-mode note: stop Streamlit before the CLI if both open `data/qdrant`.

---

## Task 1.1 — Create labeled cases from real files

Create `evaluation/retrieval_cases.json`.

### How to label (grounded process)

1. Pick **5–10 contracts** from `dataroom/` (diverse types: supply, license, IP,
   sponsorship, manufacturing, alliance).
2. **Read** distinctive clauses (governing law, CoC termination, fees, IP ownership,
   exclusivity). Grep is fine for finding candidates; eyes confirm relevance.
3. Write a question a diligence analyst might type.
4. Set `expected_sources` to the **exact** `source` string stored in Qdrant
   (usually `Path(name).name` of the `.txt` file).
5. Broad questions may list **multiple** valid sources. Narrow questions list one.
6. Put verification notes in `notes` so future-you trusts the label.

### Anti-patterns

| Bad | Why |
|-----|-----|
| `cuad_example_*.txt` from a guide | Not in your index → permanent 0 recall |
| Copy top-1 from a probe into gold without reading | Grades the system against itself |
| Label from filename alone (`…License…`) | Filename ≠ clause content |
| One vague question for the whole set | Metrics look noisy; failures are undiagnosable |

### Case shape

```json
[
  {
    "id": "governing-law-prc-01",
    "question": "Which content license agreement is governed by the laws of the People's Republic of China?",
    "expected_sources": [
      "cuad_d2e6503b_ChinaRealEstateInformationCorp_20090929_F-1_EX-10.32_4771615_EX-10.32_Content_License_Agreement.txt"
    ],
    "expected_terms": [
      "People's Republic of China",
      "Governing Law"
    ],
    "notes": "Verified in §10.11 of that file."
  }
]
```

Start with **8–15** solid cases. Grow toward 20–25 after the first baseline.
This repo's first frozen set lives in `evaluation/retrieval_cases.json`
(10 cases, manually verified against the 80-contract slice).

Do not invent labels from filenames alone. Open the file (or the probe dump) and
confirm the clause answers the question.

---

## Task 1.2 — Write an evaluator

Create `scripts/evaluate_retrieval.py`.

### Import path gotcha

Running `python scripts/evaluate_retrieval.py` puts **`scripts/`** on `sys.path`,
not the repo root. The IDE may resolve `from retrieval import …` because the
workspace root is on the analysis path — runtime still fails with
`ModuleNotFoundError: No module named 'retrieval'`.

Fix like the other scripts:

```python
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from retrieval import search_evidence  # noqa: E402
```

### Skeleton

```python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from retrieval import search_evidence  # noqa: E402


def reciprocal_rank(results: list[dict], expected_sources: set[str]) -> float:
    for rank, result in enumerate(results, start=1):
        if result["source"] in expected_sources:
            return 1.0 / rank
    return 0.0


def source_recalled(results: list[dict], expected_sources: set[str]) -> bool:
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

    return {
        "id": case["id"],
        "question": case["question"],
        "recall": int(source_recalled(results, expected)),
        "reciprocal_rank": reciprocal_rank(results, expected),
        "latency_seconds": elapsed,
        "returned_sources": [item["source"] for item in results],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path("evaluation/retrieval_cases.json"),
    )
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    rows = [evaluate_case(case, args.limit) for case in cases]

    recall = sum(row["recall"] for row in rows) / len(rows)
    mrr = sum(row["reciprocal_rank"] for row in rows) / len(rows)
    latency = sum(row["latency_seconds"] for row in rows) / len(rows)

    print(f"Recall@{args.limit}: {recall:.3f}")
    print(f"MRR@{args.limit}: {mrr:.3f}")
    print(f"Mean latency: {latency:.3f}s")
    for row in rows:
        print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
```

Run from the **repo root** (so the default `--cases` path resolves):

```powershell
python scripts\evaluate_retrieval.py
python scripts\evaluate_retrieval.py --limit 5
```

---

## Line-by-line reasoning

```python
for rank, result in enumerate(results, start=1):
```

Ranks start at 1 because search results are conventionally ranked first, second,
third—not zeroth.

```python
return 1.0 / rank
```

This rewards early relevant results. Rank 1 scores `1.0`; rank 2 scores `0.5`;
rank 10 scores `0.1`.

```python
score_threshold=None
```

Evaluation should initially ask “where did the relevant result rank?” A threshold
can hide a relevant result and make diagnosis harder. Evaluate raw ranking first;
calibrate the threshold later.

```python
expected = set(case["expected_sources"])
```

A set gives fast membership checks and allows multiple valid contracts.
`source_recalled` is true if **any** expected source appears in the top-K.

```python
"recall": int(source_recalled(...))
```

Per case this is **binary** (0 or 1). The printed `Recall@K` average (for example
`0.800`) means “80% of questions hit at least one gold file in the top K.”
Seeing only 0s and 1s on individual rows is expected, not a bug.

If **MRR** is also only 0 or 1, successes put gold at rank 1 and failures miss
entirely—no mid-rank partial credit yet. That is still a useful baseline signal.

---

## How to read the first baseline

| Signal | Healthy read | Do not overread |
|--------|--------------|-----------------|
| Recall@10 ≈ 0.8 on 10 cases | Dense search finds the right *file* most of the time on the 80-doc slice | Not proof the best *chunk* inside that file is ideal |
| All case scores 0 or 1 | Matches binary hit/miss design | Does not mean ranking is continuous or graded |
| Misses on distinctive queries | Vocabulary / chunk-boundary / wrong-doc-type issues | Do not quietly edit gold to erase the miss |
| Easy fee / governing-law hits | Confirms end-to-end wiring | Inflates average vs hard legal nuance |

When a case scores 0, compare `returned_sources` to `expected_sources`. Ask:

1. Is gold wrong? (re-read the contract)
2. Is the clause buried in a bad chunk?
3. Would exact terms (BM25) have caught a name or statute dense search missed?

Only then change retrieval. Never change labels to chase a prettier average.

### Diligence Lens — dense baseline (80 CUAD / 7840 chunks)

Recorded after grounded `evaluation/retrieval_cases.json` (10 cases):

```text
Recall@10 ≈ 0.80
MRR@10    ≈ 0.80   (hits were typically rank-1 when found)
Corpus:   ~80 contracts, dense-only nomic-embed-text:v1.5
```

Freeze these numbers before Part 2 (chunk diagnosis) or any hybrid work.

---

## Glossary

### Ground truth

Ground truth is your manually verified expected answer. Here it is usually an
expected source or clause—not an LLM-generated answer.

### Relevance judgment

A relevance judgment labels whether a result answers a query. It may be binary
(relevant/not relevant) or graded (highly/partially/not relevant).

Start binary. Graded relevance can come later.

### Recall@K

Recall@K asks whether the expected result appeared in the first `K` results.

For 20 single-source test cases, if 16 expected sources appear in the top 5:

```text
Recall@5 = 16 / 20 = 0.80
```

With the harness above, each case contributes 0 or 1; the average is macro
Recall@K over questions. Recall measures whether retrieval found the answer, not
whether rank 1 was best (MRR covers early ranking).

### Precision@K

Precision@K is the fraction of the first `K` results that are relevant.

If 2 of the top 5 chunks are relevant:

```text
Precision@5 = 2 / 5 = 0.40
```

Precision requires judging every returned chunk, so it takes more labeling effort.

### Reciprocal Rank

Reciprocal rank is `1 / rank_of_first_relevant_result`. It strongly rewards putting
the first relevant result near the top.

### Mean Reciprocal Rank (MRR)

MRR is the average reciprocal rank across all test questions.

### Latency

Latency is elapsed time. Measure stages separately:

- embedding the query;
- Qdrant search;
- BM25 search;
- fusion;
- reranking;
- Gemini generation.

---

## Definition of done

- At least **8–10** manually checked cases with real `dataroom/` filenames (grow to 15–25).
- Probe dump or UI inspection used while labeling (not filenames alone).
- Evaluator runs from `scripts/` with correct `sys.path`.
- A saved baseline for Recall@5, Recall@10, MRR@10, and mean latency.
- Written interpretation of 0-rows (do not silently relabel).
- At least five documented chunk-quality failures (carry into Part 2).

---

# Part 2 — Diagnose chunks before rewriting them

## Why

Poor retrieval can come from:

1. The answer was split across chunks.
2. The chunk lacks its clause heading.
3. The heading detector created nonsense sections.
4. The embedding model ranked a valid chunk poorly.
5. The score threshold removed a valid result.

Changing the embedding model cannot repair missing context.

## Task 2.1 — Add a chunk inspection script

Create `scripts/inspect_chunks.py`:

```python
from collections import Counter

from retrieval import load_seed_records


records = load_seed_records()
lengths = [len(item["text"]) for item in records]
sections = Counter(item["section"] for item in records)

print("chunks:", len(records))
print("min chars:", min(lengths))
print("mean chars:", sum(lengths) / len(lengths))
print("max chars:", max(lengths))
print("most repeated section labels:")

for section, count in sections.most_common(20):
    print(count, repr(section))

for item in records[:20]:
    print(
        "\n---",
        item["source"],
        "|",
        item["section"],
        "|",
        item["chunk_index"],
    )
    print(item["text"])
```

Then sample:

- shortest chunks;
- longest chunks;
- sections repeated many times;
- retrieved chunks that looked meaningless;
- adjacent chunks from the same section.

## Glossary

### Chunk

A chunk is the retrieval unit. It is the text embedded, indexed, returned, and
usually passed to the LLM.

Small chunks improve specificity but lose context. Large chunks preserve context
but may mix several topics and weaken the embedding.

### Chunk boundary

A boundary is where one chunk ends and the next begins. Good boundaries follow
semantic structure: clause, paragraph, subsection, sentence group.

### Overlap

Overlap repeats text across adjacent chunks. It reduces the chance that an answer
is cut exactly at a boundary.

Overlap is not free:

- more vectors;
- more duplicate results;
- more storage;
- more repeated evidence sent to the LLM.

### Metadata

Metadata describes a chunk without being its main text:

```python
{
    "source": "...",
    "section": "12.3 Limitation of Liability",
    "page": None,
    "chunk_index": 4,
}
```

Metadata supports filtering, citations, grouping, and contextual display.

## Definition of done

Write a short diagnosis containing:

- three recurring structural failures;
- two examples where retrieval failed despite useful source text;
- one hypothesis you can test with clause-aware chunking.

---

# Part 3 — Implement clause-aware chunking

## Goal

Use contract structure before falling back to character windows.

Desired hierarchy:

```text
contract
  -> article
    -> numbered clause
      -> paragraph
        -> sentence-aware child chunk
```

## Task 3.1 — Detect plausible clause headings

Add a heading parser near `_markdown_sections` in `retrieval.py`.

Starting point:

```python
import re


CLAUSE_HEADING = re.compile(
    r"""
    ^\s*
    (?P<number>
        (?:SECTION\s+)?
        \d+(?:\.\d+)*
        |
        ARTICLE\s+[IVXLC]+
    )
    [\s.:\-–—]*
    (?P<title>.*?)
    \s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)


def parse_clause_heading(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or len(stripped) > 160:
        return None

    match = CLAUSE_HEADING.match(stripped)
    if not match:
        return None

    number = match.group("number").strip()
    title = match.group("title").strip()

    # Bare numbers are often dates, page numbers, or list items.
    if not title and not number.upper().startswith(("SECTION", "ARTICLE")):
        return None

    return number, title
```

## Line-by-line reasoning

```python
r""" ... """
```

This is a raw string, so regex backslashes are not consumed by Python string
escaping.

```python
(?P<number>...)
```

This is a named capture group. `match.group("number")` is clearer than remembering
that the clause number happened to be group 1.

```python
\d+(?:\.\d+)*
```

This matches `1`, `1.2`, or `12.3.4`.

- `\d+`: one or more digits.
- `(?:...)`: a non-capturing group.
- `\.`: a literal period.
- `*`: repeat zero or more times.

```python
re.VERBOSE
```

Verbose mode allows whitespace and comments inside the regular expression.

The checks after the regex are important. A regex match is only a candidate.
Contract text contains dates, numbered lists, page numbers, and exhibit labels.

## Task 3.2 — Return structured sections

Create a function that emits section number, title, and body:

```python
def contract_sections(text: str) -> list[dict]:
    sections: list[dict] = []
    current_number = ""
    current_title = "Document"
    body: list[str] = []

    def flush() -> None:
        nonlocal body
        content = "\n".join(body).strip()
        if content:
            sections.append(
                {
                    "section_number": current_number,
                    "section_title": current_title,
                    "body": content,
                }
            )
        body = []

    for line in text.splitlines():
        heading = parse_clause_heading(line)
        if heading:
            flush()
            current_number, title = heading
            current_title = title or current_number
        else:
            body.append(line)

    flush()
    return sections
```

Notice that `flush()` closes the previous clause before starting the next one.

## Task 3.3 — Separate retrieval text from citation text

Store two forms:

```python
raw_text = chunk
contextual_text = (
    f"Contract: {source}\n"
    f"Clause: {section_number} {section_title}\n"
    f"Content: {raw_text}"
)

record = {
    "source": source,
    "section": f"{section_number} {section_title}".strip(),
    "section_number": section_number,
    "section_title": section_title,
    "chunk_index": chunk_index,
    "text": raw_text,
    "embedding_text": contextual_text,
}
```

In `build_index`, embed `embedding_text`:

```python
texts = [record.get("embedding_text", record["text"]) for record in records]
```

Before creating the Qdrant payload, omit `embedding_text` if you do not need to
display it:

```python
payload = {
    key: value
    for key, value in record.items()
    if key != "embedding_text"
}
```

Citation validation must continue using raw `text`. Otherwise the model might cite
the synthetic `Contract:` or `Clause:` prefix as if it appeared in the agreement.

## Glossary

### Clause-aware chunking

Chunking that follows legal-document structure instead of only a character count.

### Heuristic

A heuristic is a practical rule that is useful but not guaranteed correct.
The heading regex is a heuristic, not a legal-document parser.

### Regex

A regular expression describes a text pattern. It is useful for predictable
numbered headings but should be tested against false positives.

### Contextual embedding

Contextual embedding prepends useful metadata before creating the vector:

```text
Clause: 12.3 Limitation of Liability
Content: Neither party shall...
```

The vector sees the context; the citation still uses raw source text.

### Parent-child chunking

Parent-child retrieval embeds small child chunks for accurate matching but returns
a larger parent clause for context.

Do not implement full parent-child retrieval yet. First test whether clause metadata
plus contextual embeddings repair your failures.

## Tests to write

Create focused tests for:

- `12.3 Limitation of Liability`
- `SECTION 8. Assignment`
- `ARTICLE IV TERMINATION`
- a date such as `10.24.2024` that must not become a clause;
- a bare list item such as `1.` that must not become a clause;
- a long clause split into multiple child chunks with identical section metadata.

## Definition of done

- Chunk inspection shows meaningful section labels.
- Citation text contains only source text.
- Evaluation beats or matches baseline without a major latency regression.
- You can explain why the contextual prefix is embedded but not cited.

One clean rebuild is required after this part.

---

# Part 4 — Add BM25 lexical retrieval

## Why

Dense retrieval answers “what text means something similar?”

Lexical retrieval answers “what text contains discriminative query terms?”

Contracts contain exact terminology, party names, defined terms, clause numbers,
and rare phrases. Combining both retrieval styles is usually stronger than either
alone.

## Task 4.1 — Install one learning-friendly dependency

```powershell
pip install rank-bm25
```

Then pin the installed version in `requirements.txt`.

`rank-bm25` is suitable for learning because its API exposes the corpus tokens and
scores directly. Later, Qdrant sparse vectors can replace it for server-scale use.

## Task 4.2 — Tokenize consistently

Starting tokenizer:

```python
import re


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-'.]*")


def lexical_tokens(text: str) -> list[str]:
    return [
        token.casefold()
        for token in TOKEN_PATTERN.findall(text)
    ]
```

Why `casefold()` instead of `lower()`? `casefold()` is a stronger Unicode-aware
normalization. Contract text is mostly English, but consistent normalization is
still useful.

Do not remove legal stopwords prematurely. Words such as `shall`, `may`, `not`,
and `except` can affect legal meaning.

## Task 4.3 — Build a BM25 index

Create `lexical_retrieval.py`:

```python
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi


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
```

## Line-by-line reasoning

```python
@dataclass
class BM25Index:
```

A dataclass groups the source records with the trained BM25 object. The position
of every record must match the position of its tokenized document.

```python
scores = index.model.get_scores(...)
```

BM25 scores every indexed chunk against the query.

```python
np.argsort(scores)[::-1][:limit]
```

- `argsort` returns indices ordered from lowest score to highest.
- `[::-1]` reverses that order.
- `[:limit]` keeps only the best candidates.

## Glossary

### Lexical retrieval

Retrieval based on words or tokens that occur in both query and document.

### Token

A token is a searchable unit. In this simple implementation, tokens are normalized
word-like strings.

### Term frequency (TF)

Term frequency asks how often a term appears in a document. Repetition can indicate
importance, but unlimited repetition should not increase relevance forever.

### Document frequency (DF)

Document frequency is how many documents contain a term.

### Inverse document frequency (IDF)

Rare terms carry more information than common terms. IDF gives rare terms more
weight.

For example, `agreement` appears almost everywhere and is weak evidence.
`indemnification` or a party name is more discriminative.

### BM25

BM25 stands for Best Matching 25. It is a classic lexical ranking function.

Conceptually:

```text
score = sum over query terms of:
        rarity(term)
        × saturated_term_frequency(term, document)
        × document_length_normalization
```

Important behavior:

- exact rare terms receive high weight;
- repeating a term helps, but gains saturate;
- long documents are normalized so length alone does not win.

### BM25Okapi

Okapi BM25 is the standard BM25 variant exposed by `rank-bm25`.

### `k1` and `b`

BM25 commonly exposes:

- `k1`: controls how quickly term-frequency gains saturate;
- `b`: controls document-length normalization.

Use defaults initially. Tuning them before you have an evaluation set is noise.

## Definition of done

- BM25 returns good results for exact party names and legal phrases.
- Dense retrieval still wins some paraphrase cases.
- Both retrievers are evaluated independently using the same cases.

---

# Part 5 — Combine dense and BM25 with Reciprocal Rank Fusion

## Why not add raw scores?

Dense cosine scores and BM25 scores are on different scales.

Example:

```text
dense score: 0.73
BM25 score: 18.42
```

Adding them would let BM25 dominate only because its numeric scale is larger.

RRF combines ranks instead of raw scores.

## Task 5.1 — Create a stable chunk identity

```python
def chunk_key(item: dict) -> tuple:
    return (
        item["source"],
        item["section"],
        item["chunk_index"],
    )
```

If identical keys can occur, include the stored `ordinal`. The key must identify
one indexed chunk consistently across retrievers.

## Task 5.2 — Implement RRF

```python
from collections import defaultdict


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict]],
    *,
    k: int = 60,
    limit: int = 20,
) -> list[dict]:
    fused_scores: dict[tuple, float] = defaultdict(float)
    candidates: dict[tuple, dict] = {}

    for results in ranked_lists:
        for rank, result in enumerate(results, start=1):
            key = chunk_key(result)
            fused_scores[key] += 1.0 / (k + rank)
            candidates.setdefault(key, result)

    ordered_keys = sorted(
        fused_scores,
        key=fused_scores.get,
        reverse=True,
    )

    fused = []
    for key in ordered_keys[:limit]:
        result = dict(candidates[key])
        result["fusion_score"] = fused_scores[key]
        fused.append(result)
    return fused
```

Call it with:

```python
dense = search_evidence(question, limit=20, score_threshold=None)
lexical = search_bm25(bm25_index, question, limit=20)
fused = reciprocal_rank_fusion([dense, lexical], limit=20)
```

## Line-by-line reasoning

```python
defaultdict(float)
```

Missing keys start at `0.0`, so each retriever can add its contribution.

```python
1.0 / (k + rank)
```

An item at rank 1 contributes more than an item at rank 20. An item found by both
retrievers receives both contributions.

```python
candidates.setdefault(key, result)
```

Store one payload for each candidate without overwriting it every time the same
chunk appears in another list.

## Glossary

### Hybrid retrieval

Hybrid retrieval combines different retrieval signals. Here:

- dense semantic similarity;
- BM25 lexical relevance.

### Candidate

A candidate is a possible result retained for later stages. Retrieval can fetch
20–50 candidates even when the final LLM receives only 6.

### Fusion

Fusion combines multiple ranked result lists.

### Reciprocal Rank Fusion (RRF)

RRF rewards high rank and agreement across retrievers without requiring score
normalization.

### Fusion constant `k`

The constant reduces how sharply rank differences affect the fused score.
`60` is a common starting value, not a universal optimum.

Do not confuse this `k` with “top K results.” They are unrelated parameters.

## Definition of done

- Hybrid Recall@10 is compared to both dense-only and BM25-only.
- Exact-term cases improve without damaging paraphrase cases.
- Every fused result exposes its component ranks for debugging.

---

# Part 6 — Add a reranker

## Why retrieval and reranking are separate

The first-stage retrievers must search thousands of chunks quickly. They use
representations that can be indexed efficiently.

A reranker sees only a small candidate set and can apply a more expensive model.

```text
7,840 chunks
  -> dense top 20 + BM25 top 20
  -> RRF produces about 25 unique candidates
  -> reranker scores those candidates carefully
  -> final top 6 goes to Gemini
```

## Task 6.1 — Install a cross-encoder

Learning-first choice:

```powershell
pip install sentence-transformers
```

Suggested small model:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

This adds PyTorch and model downloads, so record installation size and latency.
Start on CPU to avoid fighting Ollama for the 4 GB GPU.

## Task 6.2 — Implement reranking

Create `reranking.py`:

```python
from functools import lru_cache

from sentence_transformers import CrossEncoder


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    return CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device="cpu",
    )


def rerank(
    question: str,
    candidates: list[dict],
    *,
    limit: int = 6,
) -> list[dict]:
    if not candidates:
        return []

    pairs = [
        (
            question,
            " ".join(
                [
                    candidate.get("section", ""),
                    candidate["text"],
                ]
            ),
        )
        for candidate in candidates
    ]

    scores = get_reranker().predict(pairs)

    ranked = []
    for candidate, score in zip(candidates, scores, strict=True):
        item = dict(candidate)
        item["rerank_score"] = float(score)
        ranked.append(item)

    ranked.sort(key=lambda item: item["rerank_score"], reverse=True)
    return ranked[:limit]
```

## Line-by-line reasoning

```python
@lru_cache(maxsize=1)
```

Loading a model is expensive. Cache one process-wide model instead of loading it
for every Streamlit rerun or query.

```python
pairs = [(question, candidate_text), ...]
```

A cross-encoder evaluates the query and candidate together. This allows detailed
token interactions.

```python
zip(candidates, scores, strict=True)
```

`strict=True` fails if the model returns a different number of scores than input
pairs. Silent truncation would hide a correctness bug.

## Glossary

### Reranking

Reranking reorders a small candidate set using a more accurate but slower method.

### Bi-encoder

Dense retrieval is commonly a bi-encoder design:

- encode question independently;
- encode documents independently;
- compare vectors.

Documents can be precomputed and indexed, making search fast.

### Cross-encoder

A cross-encoder reads the question and document together. It can model detailed
interactions but cannot cheaply score every chunk in the corpus.

### First-stage retrieval

Fast, broad candidate generation. Dense and BM25 are first-stage retrievers here.

### Second-stage retrieval

Slower refinement over a small candidate pool. Reranking is the second stage.

## Experiments

Compare:

1. Rerank dense-only top 20.
2. Rerank hybrid top 20.
3. Hybrid without reranking.

Measure MRR and latency. A reranker that improves MRR slightly but adds 20 seconds
may be a poor fit for an interactive PoC.

## Definition of done

- The model is loaded once.
- Reranker latency is separately measured.
- MRR improves on the frozen evaluation set.
- Final evidence count remains configurable.

---

# Part 7 — Add diversity and deduplication

## Why

Topical relevance alone can return six overlapping chunks from one contract.
That is especially harmful when the review question asks for cross-document
conflicts.

## Task 7.1 — Cap chunks per source

```python
from collections import Counter


def diversify_by_source(
    ranked: list[dict],
    *,
    limit: int = 6,
    max_per_source: int = 2,
) -> list[dict]:
    selected: list[dict] = []
    source_counts: Counter[str] = Counter()

    for item in ranked:
        source = item["source"]
        if source_counts[source] >= max_per_source:
            continue

        selected.append(item)
        source_counts[source] += 1

        if len(selected) == limit:
            break

    return selected
```

This is transparent and easy to inspect. It is a good first diversity strategy.

## Task 7.2 — Remove near-duplicate overlap

Simple token overlap:

```python
def jaccard_similarity(left: str, right: str) -> float:
    left_tokens = set(lexical_tokens(left))
    right_tokens = set(lexical_tokens(right))
    union = left_tokens | right_tokens
    if not union:
        return 0.0
    return len(left_tokens & right_tokens) / len(union)
```

Skip a candidate if it has very high overlap with an already selected chunk from
the same source.

## Glossary

### Diversity

Diversity means selected evidence covers different sources or aspects instead of
repeating the same passage.

### Deduplication

Deduplication removes identical or near-identical results.

### Jaccard similarity

For two token sets:

```text
Jaccard = size(intersection) / size(union)
```

It ranges from 0 to 1.

### MMR

Maximal Marginal Relevance balances:

- relevance to the question;
- novelty compared with already selected results.

Conceptually:

```text
MMR(candidate) =
    lambda × relevance(candidate, query)
    - (1 - lambda) × similarity(candidate, selected_results)
```

Implement the simple source cap first. Then implement MMR as a separate experiment
if duplication remains a problem.

## Definition of done

- Cross-document questions receive evidence from multiple sources.
- Single-document questions still receive enough context.
- Diversity logic does not claim that different sources automatically conflict.

---

# Part 8 — Integrate the pipeline

## Target flow

Do not replace `search_evidence` immediately. Add a new function so the baseline
remains available:

```python
def search_evidence_hybrid(
    question: str,
    *,
    final_limit: int = 6,
) -> list[dict]:
    dense = search_evidence(
        question,
        limit=20,
        score_threshold=None,
    )
    lexical = search_bm25(
        get_bm25_index(),
        question,
        limit=20,
    )
    fused = reciprocal_rank_fusion(
        [dense, lexical],
        limit=25,
    )
    reranked = rerank(
        question,
        fused,
        limit=15,
    )
    return diversify_by_source(
        reranked,
        limit=final_limit,
        max_per_source=2,
    )
```

## Why these counts differ

- Retrieve 20 from each first-stage method to preserve recall.
- Fuse to about 25 unique candidates.
- Rerank more than the final count so diversity has room to skip duplicates.
- Send only 6 chunks to Gemini to control latency and prompt size.

These are starting values. Evaluation decides whether they remain.

## Add stage timing

Use `perf_counter()` around every stage:

```python
timings = {}

started = perf_counter()
dense = search_evidence(...)
timings["dense_seconds"] = perf_counter() - started

started = perf_counter()
lexical = search_bm25(...)
timings["bm25_seconds"] = perf_counter() - started
```

Expose these in the existing telemetry rather than logging only total time.

## Glossary

### Pipeline

A pipeline is an ordered sequence of transformations. Each stage has an input,
output, cost, and possible failure.

### Candidate pool

The union of possible results considered before final selection.

### Ablation

An ablation disables one component to measure its contribution.

Examples:

- hybrid with reranker;
- hybrid without reranker;
- dense with reranker;
- dense only.

Ablation prevents crediting the whole pipeline for an improvement caused by one
stage.

## Definition of done

- Dense-only search still exists for comparison.
- Hybrid stages expose ranks and timings.
- `risk_graph.py` uses the new function only after evaluation passes.

---

# Part 9 — Calibrate thresholds and final evidence

## Correct order

1. Improve ranking.
2. Measure ranking without a hard threshold.
3. Inspect score distributions for relevant and irrelevant chunks.
4. Choose a threshold.

Do not use the same numeric threshold for:

- cosine similarity;
- BM25 score;
- RRF score;
- cross-encoder score.

They are different scoring systems.

Apply a threshold only at the stage whose score you understand. Often the final
reranker score is the most useful candidate, but it still needs calibration.

## Glossary

### Score distribution

The range and frequency of scores produced over many queries/results.

### Calibration

Calibration maps a score to observed behavior. It answers questions such as:
“Among results scoring above this value, how often were they relevant?”

### False positive

An irrelevant chunk accepted as evidence.

### False negative

A relevant chunk rejected or ranked too low.

### Precision–recall tradeoff

A stricter threshold usually raises precision but lowers recall. Legal diligence
often needs high recall first, followed by reranking and explicit abstention.

## Definition of done

- Threshold was selected from labeled cases, not intuition.
- You can show false-positive and false-negative examples.
- The app abstains when final evidence is weak.

---

# Part 10 — Full-corpus and container milestone

Only after the 80-document experiment is stable:

```powershell
# Stop Streamlit first.
python scripts/clear_index.py
python scripts/export_cuad.py --replace

# For path-mode local testing:
python scripts/rebuild_index.py

# Or cut over to server Qdrant:
docker compose up --build
```

For ~50,000 points, prefer Qdrant server mode through `QDRANT_URL`. The path-mode
client is useful for learning and small local runs but warns above 20,000 points.

Re-run the exact same evaluation cases after scaling. Corpus size can lower
retrieval quality because more distractors compete for the top ranks.

## Glossary

### Distractor

A non-relevant chunk that looks similar enough to outrank relevant evidence.

### Index

The data structure used to search stored vectors or terms efficiently.

### Qdrant path mode

An embedded local Qdrant store accessed directly by one Python process. It uses a
filesystem lock and is not suitable for concurrent clients.

### Qdrant server mode

A separate Qdrant process accessed over HTTP or gRPC. Multiple clients can connect,
and it is the better shape for containerization and larger collections.

## Definition of done

- Full index point count equals embedded chunk count.
- Evaluation metrics are recorded for 80-document and full-corpus runs.
- Compose reaches host Ollama and Qdrant.
- No `.lock` workaround is needed in server mode.

---

# Suggested implementation sessions

## Session 1 — Measurement

- Label 15–25 retrieval cases.
- Implement baseline evaluator.
- Record metrics.

## Session 2 — Chunk diagnosis

- Implement chunk inspector.
- Categorize failures.
- Design clause-heading rules from actual CUAD examples.

## Session 3 — Clause-aware chunking

- Implement parser and tests.
- Add contextual embedding text.
- Clean rebuild.
- Compare baseline metrics.

## Session 4 — BM25

- Implement tokenizer and BM25 index.
- Evaluate lexical-only.
- Write down dense-vs-lexical examples.

## Session 5 — Hybrid fusion

- Implement stable chunk keys.
- Implement RRF.
- Evaluate hybrid.

## Session 6 — Reranking

- Add cross-encoder.
- Measure quality and latency.
- Run ablations.

## Session 7 — Diversity and integration

- Add source cap and deduplication.
- Integrate with `risk_graph.py`.
- Validate citations and Gemini latency.

## Session 8 — Scale

- Move to Qdrant server mode.
- Restore all CUAD contracts.
- Rebuild and run the frozen evaluation suite.

---

# Learning journal

## Experiment: dense-only baseline harness (2026-10-24)

### Hypothesis

A small, manually labeled source-level set can score dense retrieval honestly.
Placeholder guide filenames would score 0; grounded CUAD names should not.

### Change

- Built `evaluation/retrieval_cases.json` (10 cases) by reading real `dataroom/` files.
- Wired `scripts/evaluate_retrieval.py` (+ `sys.path` root insert).
- Probe dump via `evidence_evals.py` for offline chunk reading.
- Expanded this Part 1 with the full probe → label → score → freeze loop.

### Controlled variables

Same 80-contract index, same `search_evidence` dense path, no BM25/hybrid/rerank,
`score_threshold=None` during scoring.

### Results

- Recall@10 ≈ **0.80** (binary per-case hit/miss → average 8/10)
- MRR@10 ≈ **0.80** (hits typically at rank 1; misses at 0)
- Per-row scores only 0 or 1 — expected for this harness design

### Failure examples

Inspect `returned_sources` on the 0-rows before Part 2. Do not edit gold to erase them.

### Interpretation

Healthy dense baseline on the 80-doc slice. Not “done”: misses remain; metric is
file-level, not best-chunk; set is small. Freeze before chunking or hybrid work.

### Next decision

Part 2 — diagnose chunk failures on the miss cases; keep this eval set frozen.

---

# Learning journal template

Append one entry after each session:

```markdown
## Experiment: [name]

### Hypothesis
[What I expect and why.]

### Change
[Exact files/functions changed.]

### Controlled variables
[What remained unchanged.]

### Results
- Recall@5:
- Recall@10:
- MRR@10:
- Mean retrieval latency:
- Mean reranker latency:

### Failure examples
[Queries and bad results.]

### Interpretation
[What the result suggests; distinguish evidence from guesses.]

### Next decision
[One focused follow-up experiment.]
```

---

# Final mental model

```text
Chunking determines what can be retrieved.

Dense retrieval contributes semantic recall.
BM25 contributes exact-term recall.
RRF combines rankings without mixing incompatible raw scores.
Reranking improves candidate order with deeper query-document interaction.
Diversity prevents repetitive evidence.
Evaluation tells you whether any of that actually helped.
Gemini reasons only over the evidence these stages preserve.
Citation validation checks whether Gemini quoted that evidence exactly.
```

The most important lesson is that RAG quality is not primarily a prompt problem.
It is an information-retrieval system whose output later becomes an LLM prompt.
