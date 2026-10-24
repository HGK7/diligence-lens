# 7Guided lab: Replace Northstar chunks with ContractNLI

Follow this top to bottom. Check boxes as you go.  
You can paste commands into PowerShell from the project root:

`d:\Git Projects\self-correcting-rag`

**What you will do**

1. Measure the current seed-pack baseline.
2. Park the eight Northstar markdown files so they stop seeding.
3. Export a small ContractNLI slice into `vendor_docs/`.
4. Rebuild the Qdrant index (this **replaces** all evidence chunks).
5. Measure search + risk-review performance.
6. Optionally scale up.
7. Optionally restore Northstar.

**Corpus facts (on your disk)**

| Split | File | Documents | Hypotheses |
|-------|------|-----------|------------|
| train | `vendor_docs/contract-nli/contract-nli/train.json` | 423 | 17 |
| dev | `vendor_docs/contract-nli/contract-nli/dev.json` | 61 | 17 |
| test | `vendor_docs/contract-nli/contract-nli/test.json` | 123 | 17 |

This lab starts with **5 docs from** `test.json` (smoke), then offers **30** (perf).

---

## Phase 0 — Baseline on the current seed pack

Do this **before** you move any files.

### 0.1 Start the portal

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run portal.py
```

In the sidebar: **Build or rebuild index** (if index not ready).

### 0.2 Fill this table

| Metric | How to get it | Your value |
|--------|----------------|------------|
| Records indexed | Sidebar success text after rebuild | |
| Dimensions | Same message | |
| Index wall time | Phone timer around the rebuild click | |
| Search: accepted passages | Evidence Search, question below, threshold `0.55` | |
| Search: top score | Same | |
| Search feel (seconds) | Rough wall time | |
| Review: evidence passages | Risk Review metrics | |
| Review: total portal latency | Risk Review metrics | |
| Review: Gemini called | Yes/No | |
| Review decision | Heading under metrics | |
| Peak GPU memory | Task Manager → GPU, or `nvidia-smi` during rebuild | |

**Fixed baseline question (Northstar-era):**

```text
Do the incident notification and Priority 1 response commitments agree across documents?
```

### 0.3 Optional: copy baseline into the journal

Open `journal/atlas-lens-field-notes.md` and add a row under **Log of what I actually did**.

- [x] Phase 0 done

---

## Phase 1 — Stop Northstar from coming back

`portal.py` calls `seed_documents()` on every start. That **rewrites** `01_…08_*.md` into `vendor_docs/`.  
If you only delete those files, they return on the next Streamlit start.

### 1.1 Create archive folders

```powershell
New-Item -ItemType Directory -Force -Path "vendor_docs\_archived_northstar" | Out-Null
New-Item -ItemType Directory -Force -Path "datasets" | Out-Null
```

### 1.2 Move the eight seed markdown files

```powershell
Move-Item -Path "vendor_docs\0*.md" -Destination "vendor_docs\_archived_northstar\" -Force
```

Check:

```powershell
Get-ChildItem vendor_docs -File
Get-ChildItem vendor_docs\_archived_northstar
```

You should see **no** `01_…08_` files at the top level of `vendor_docs/`.  
The nested `contract-nli\` folder can stay where it is for now.

### 1.3 Disable auto-seed in `portal.py`

Open `portal.py`. Find:

```python
from seed_vendor_pack import seed_documents
```

and

```python
seed_documents()
```

Comment them out (guided edit):

```python
# from seed_vendor_pack import seed_documents
```

```python
# seed_documents()  # disabled for ContractNLI corpus experiment
```

Save the file.

- [x] Northstar archived  
- [x] `seed_documents()` disabled  

---

## Phase 2 — Export ContractNLI texts (smoke: 5 docs)

`load_seed_records()` only reads **top-level** `vendor_docs/*.{md,txt,pdf}`.  
It will **not** read nested JSON or `raw/` PDFs. So we export plain files.

### 2.1 Create the export script

Create a new file: `scripts/export_contractnli.py`

If `scripts\` does not exist:

```powershell
New-Item -ItemType Directory -Force -Path "scripts" | Out-Null
```

Paste this into `scripts/export_contractnli.py`:

```python
"""Export ContractNLI document texts into vendor_docs for Atlas Lens indexing."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def safe_name(name: str) -> str:
    cleaned = re.sub(r"[^\w.\-]+", "_", name, flags=re.UNICODE)
    return cleaned.strip("._") or "document"


def export_docs(
    json_path: Path,
    out_dir: Path,
    limit: int,
    prefix: str = "cnli",
) -> list[Path]:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    documents = payload["documents"]
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for document in documents[:limit]:
        doc_id = document["id"]
        file_name = safe_name(document.get("file_name") or f"doc-{doc_id}")
        target = out_dir / f"{prefix}_{doc_id}_{file_name}.md"
        header = (
            f"# ContractNLI document {doc_id}\n\n"
            f"- source_file: {document.get('file_name')}\n"
            f"- document_type: {document.get('document_type')}\n"
            f"- url: {document.get('url')}\n\n"
            f"---\n\n"
        )
        body = document["text"].strip() + "\n"
        target.write_text(header + body, encoding="utf-8")
        written.append(target)
    return written


def print_hypotheses(json_path: Path) -> None:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    print("\nContractNLI hypotheses (use these as portal questions):\n")
    for key, meta in payload["labels"].items():
        print(f"{key}: {meta['short_description']}")
        print(f"  {meta['hypothesis']}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json",
        type=Path,
        default=Path("vendor_docs/contract-nli/contract-nli/test.json"),
    )
    parser.add_argument("--out", type=Path, default=Path("vendor_docs"))
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--prefix", type=str, default="cnli")
    parser.add_argument(
        "--show-labels",
        action="store_true",
        help="Print the 17 hypotheses and exit",
    )
    args = parser.parse_args()

    if args.show_labels:
        print_hypotheses(args.json)
        return

    paths = export_docs(args.json, args.out, args.limit, args.prefix)
    print(f"Wrote {len(paths)} files to {args.out.resolve()}")
    for path in paths:
        print(f"  - {path.name}")


if __name__ == "__main__":
    main()
```

### 2.2 Run the smoke export (5 documents)

```powershell
.\.venv\Scripts\python.exe scripts\export_contractnli.py --limit 5
```

Expected: five `cnli_*.md` files under `vendor_docs\`.

```powershell
Get-ChildItem vendor_docs\cnli_*.md
```

### 2.3 Print the 17 hypotheses (for later questions)

```powershell
.\.venv\Scripts\python.exe scripts\export_contractnli.py --show-labels
```

Copy **three** hypotheses into a notepad for Phase 4 (one you think is common, one rare, one odd).

### 2.4 Sanity-open one file

Open any `vendor_docs\cnli_*.md` in the editor. You should see:

- a short YAML-ish header block  
- then real NDA prose

If you see raw JSON braces for the whole corpus, stop — export failed.

- [x] 5 `cnli_*.md` files exist  
- [x] Hypotheses printed  

---

## Phase 3 — Rebuild index (this replaces old chunks)

### 3.1 Confirm what will be indexed

```powershell
Get-ChildItem vendor_docs -File | Select-Object Name, Length
```

Only top-level `.md` / `.txt` / `.pdf` are indexed. Nested `contract-nli\` is ignored by `load_seed_records()`.

### 3.2 Rebuild in the UI

1. Restart Streamlit if it was already running (`Ctrl+C`, then `streamlit run portal.py`).
2. Sidebar → **Build or rebuild index**.
3. Wait for success: `Indexed N records with D dimensions.`

### 3.3 Record rebuild metrics

| Metric     | Your value |
| ---------- | ---------- |
| Records    | 67         |
| Dimensions | 768        |
| Wall time  |            |
| Peak VRAM  |            |
| Errors?    |            |

**What the code does on rebuild** (`retrieval.build_index`):

- Deletes collection `atlas_vendor_risk`  
- Recreates it  
- Deletes `semantic_cache` (old reviews become invalid)  
- Embeds every chunk with Ollama `nomic-embed-text:v1.5`  
- Upserts new points

So this step **is** the chunk replacement.

- [x] Rebuild succeeded on 5 docs  

---

## Phase 4 — Performance + behavior checks (smoke)

Use the **same threshold** as baseline when comparing: start at `0.55`.

### 4.1 Evidence Search

Tab: **Evidence Search**

For each of your three hypotheses:

| Hypothesis key | Accepted passages | Top score | Notes (on-topic?) |
|----------------|-------------------|-----------|-------------------|
| | | | |
| | | | |
| | | | |

Also try one **nonsense** question:

```text
What is the diameter of Mars?
```

Expect few or zero passages above threshold → good quality gate.

### 4.2 Risk Review

Tab: **Risk Review**

Run **one** hypothesis that got strong search hits.

Fill:

| Metric | Value |
|--------|-------|
| Cache source | |
| Evidence passages | |
| Total portal latency | |
| Gemini called | |
| Decision | |
| Citations look real? | |
| Retrieved evidence expanders look right? | |

### 4.3 Informal gold check (optional but useful)

For the same document id you care about, gold labels live in JSON:

```text
document["annotation_sets"][0]["annotations"]["nda-X"]["choice"]
```

Rough mapping (not perfect):

| ContractNLI `choice` | Atlas-ish expectation |
|----------------------|------------------------|
| Entailment | supported / `pass` / mild `concern` |
| Contradiction | `conflict` or `concern` |
| NotMentioned | `insufficient_evidence` if nothing else supports it |

Write one sentence in the journal: did the portal roughly agree?

- [ ] Search checks done  
- [ ] One full review done  

---

## Phase 5 — Scale to a perf pack (~30 docs)

Only if smoke felt healthy (no OOM, rebuild finished, search returned something).

### 5.1 Clear previous exports (optional clean slate)

```powershell
Remove-Item vendor_docs\cnli_*.md -Force
```

### 5.2 Export 30 from test.json

```powershell
.\.venv\Scripts\python.exe scripts\export_contractnli.py --limit 30
```

### 5.3 Rebuild again

Sidebar → **Build or rebuild index**.

### 5.4 Scoreboard

|                         | Seed (Phase 0) | CNLI-5 | CNLI-30 |
| ----------------------- | -------------- | ------ | ------- |
| Records                 |                |        | 400     |
| Index time              |                |        |         |
| Search latency (feel)   |                |        |         |
| Review latency          |                |        |         |
| Peak VRAM               |                |        |         |
| Retrieval quality (1–5) |                |        |         |

- [x] CNLI-30 indexed and measured  

---

## Phase 6 — Going larger (optional)

| Target | Command |
|--------|---------|
| All test (123) | `python scripts/export_contractnli.py --limit 123` |
| Dev (61) | `python scripts/export_contractnli.py --json vendor_docs/contract-nli/contract-nli/dev.json --limit 61` |
| Train (423) | only if GPU/time allow; expect long embed |

Watch VRAM. If Ollama dies, drop `--limit` and retry.

---

## Phase 7 — Rollback to Northstar (when you want demos back)

### 7.1 Remove ContractNLI exports from the index folder

```powershell
Remove-Item vendor_docs\cnli_*.md -Force
```

### 7.2 Restore seed files

```powershell
Move-Item -Path "vendor_docs\_archived_northstar\*" -Destination "vendor_docs\" -Force
```

### 7.3 Re-enable seeding in `portal.py`

Uncomment:

```python
from seed_vendor_pack import seed_documents
```

```python
seed_documents()
```

### 7.4 Rebuild index

Restart Streamlit → **Build or rebuild index**.

Confirm the default incident / P1 conflict question works again.

- [ ] Northstar restored (if you chose to roll back)

---

## Quick reference — useful one-liners

```powershell
# Hypotheses
.\.venv\Scripts\python.exe scripts\export_contractnli.py --show-labels

# Smoke export
.\.venv\Scripts\python.exe scripts\export_contractnli.py --limit 5

# Perf export
.\.venv\Scripts\python.exe scripts\export_contractnli.py --limit 30

# See what the indexer will pick up
Get-ChildItem vendor_docs -File
```

Peek at one document’s gold labels (replace `DOC_ID` and run from project root):

```powershell
.\.venv\Scripts\python.exe -c "
import json
from pathlib import Path
data = json.loads(Path('vendor_docs/contract-nli/contract-nli/test.json').read_text(encoding='utf-8'))
doc = next(d for d in data['documents'] if d['id'] == DOC_ID)
anns = doc['annotation_sets'][0]['annotations']
for key, ann in anns.items():
    hyp = data['labels'][key]['hypothesis']
    print(key, ann['choice'], '->', hyp[:80])
"
```

Change `DOC_ID` to a real id from an exported filename (`cnli_<id>_...`).

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Index still has Northstar facts | Seed files returned | Confirm `seed_documents()` commented; no `0*.md` at top level |
| `Indexed 0` / no readable text | No top-level md/txt/pdf | Re-run export; check `Get-ChildItem vendor_docs -File` |
| Rebuild very slow / GPU spike | Too many docs | Lower `--limit` |
| Search always empty | Threshold too high for new domain | Try `0.35`, compare |
| Review says no evidence | Same | Lower review slider; confirm rebuild finished |
| Old review answers reappear | Looking at session state | Run a **new** question or clear session; semantic cache was wiped on rebuild |
| Portal recreates `01_…08_` | Forgot to comment seed | Phase 1.3 again |

---

## After-action (update your blog draft)

In `journal/atlas-lens-field-notes.md`:

1. Fill the **Log** table (dates + what you ran).
2. Paste the scoreboard from Phase 5.
3. One paragraph: what surprised you about ContractNLI vs Northstar.

---

## You are here

Start at **Phase 0**.  
When stuck, note the phase number and the exact error text.

Next guided lesson after this lab (optional): map one gold `choice` to Quality Lab cases automatically.
