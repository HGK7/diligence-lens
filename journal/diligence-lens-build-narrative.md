# Diligence Lens — build narrative (draft knowledge dump)

*A readable dump of what happened while turning a Nextwork practice RAG into a portfolio-shaped PoC. Written so future-me can re-learn the path without replaying the whole chat. Draft voice; facts over polish.*

---

## 0. Where this started

I began with **Atlas Lens**: a local-first vendor-risk portal from a guided Nextwork project. The stack was already real — Streamlit UI, Ollama embeddings, Qdrant, LangGraph, Gemini structured findings, citation checks, cache telemetry, optional human approval.

The learning gap was ownership. I could run the buttons. I could not always explain the code in an interview. Parallel work included fixing bugs (`QueryResponse` vs `.points`, Risk Review metrics without rendering findings) and deciding what corpus would make the demo credible.

The product brief was vendor diligence. The deeper technical story was: **quality-gated retrieval + abstention + verified quotes + visible failure modes**. That story can wear more than one domain costume.

---

## 1. Corpus hunt (and a wrong turn)

I considered ContractNLI first because I already had it. It is a strong **ML NLI** dataset (entail / contradict / not mentioned on NDAs). It is a weak **Atlas/Diligence product** fit:

- Theme: NDAs only, not SLA / IR / DPA packs.
- Task shape: one contract + fixed hypotheses, not multi-doc synthesis.
- “Conflict” means hypothesis vs one NDA, not master vs side letter across a room.
- Labels do not map cleanly onto `pass` / `conflict` / `insufficient_evidence`.

ContractNLI is still fine as a **load probe**. It is not the recruiter headline.

I then ranked multi-doc options. **LegalBench-Deep** looked perfect (50-contract data rooms, planted conflicts) but the Hugging Face link returned **401 / unavailable**. Dead end — do not block a PoC on a gated card.

**Decision:** rebrand toward multi-contract diligence and use **CUAD** (Atticus Project, CC BY 4.0) from GitHub: https://github.com/The-Atticus-Project/cuad

Why CUAD for signal:

- Real commercial contracts, expert-annotated clause culture.
- ~510 docs → honest “data room” scale.
- Clear license + paper citation (NeurIPS 2021).
- Fits a Diligence Lens story without pretending to be a law firm product.

---

## 2. Rebrand: Atlas Lens → Diligence Lens

Cosmetic and prompt changes so the UI matches the corpus:

- Portal title, captions, default questions, quality-lab cases.
- Approval gate copy.
- Gemini system rules: multi-contract data room, cross-source conflict.
- Report heading: Diligence Lens data-room review.
- Collection name: `diligence_dataroom`.
- Active corpus folder: `dataroom/` (legacy `vendor_docs/` still a fallback).

Streamlit polish (badges, bordered telemetry, tabbed finding sections, Material icons, light theme in `.streamlit/config.toml`) is there to look like a PoC, not a homework script.

---

## 3. Legal / git hygiene (boring, necessary)

Portfolio signal includes not looking careless with other people’s data.

Added:

- `docs/ATTRIBUTION.md` — CUAD credit + BibTeX
- `docs/TERMS.md` / `docs/PRIVACY.md` — stubs (not counsel-approved; “not legal advice”)
- `NOTICE` + `LICENSE` (MIT for **code**; CUAD texts stay CC BY)
- UI expander with the same attribution
- Export `MANIFEST.json` with source path, counts, citation
- `.gitignore` for `.env`, `datasets/*`, `dataroom/*`, Qdrant `data/`, `audit/`, ContractNLI dumps, zips, secrets
- `docs/GIT_HYGIENE.md`

Pattern to remember: **full dump stays local and ignored; active room is regenerable; never commit keys or corpora.**

---

## 4. CUAD download surprise

I expected `full_contract_txt/` after extracting GitHub `data.zip`.

What I actually got:

```text
datasets/cuad/data/CUADv1.json   (~40 MB, 510 SQuAD-style docs)
datasets/cuad/data/test.json
datasets/cuad/data/train_separate_questions.json
```

`Get-ChildItem … -Filter *.txt` under `datasets/cuad` returned **nothing**. That was correct, not a failed extract.

Each `CUADv1.json` entry has `title` + `paragraphs[].context` = full contract text. So the export script learned two modes:

1. Prefer `CUADv1.json` (GitHub path).
2. Fall back to a folder of `.txt` if someone later pulls the larger HF/Zenodo dump.

First plan was “export N, scale later.” I then chose **export all 510 into `dataroom/`** and treat indexing/chunking load as a later stage. Ran:

```powershell
python scripts/export_cuad.py --replace
# → Exported 510 / all
```

Each file gets a short attribution header. `MANIFEST.json` records `limit: "all"`.

---

## 5. Chunking strategy (steady, not clever)

Naive fixed character windows smash mid-sentence and ignore contract structure.

Current approach in `retrieval.py`:

1. Normalize line endings.
2. Prefer markdown / ALL-CAPS section breaks when present.
3. Pack paragraphs under `CHUNK_SIZE` (800).
4. Oversized paragraphs → sentence-aware windows with `CHUNK_OVERLAP` (120).
5. Skip tiny crumbs (`CHUNK_MIN_CHARS`).

Interview line: “Section-aware packing first; windows only when a block is too big.”

---

## 6. Indexing pain on Windows (the real education)

### 6.1 Symptom A — Qdrant “connection refused”

```text
Post "http://127.0.0.1:50358/collections": dial tcp … actively refused
```

**What it is:** `QdrantClient(path="data/qdrant")` boots a **local** Qdrant process on a **random** port. It is not Ollama (Ollama is `:11434`).

**Why it broke:**

- Every `index_exists` / `search` / `build_index` opened a **new** client and `close()`’d it.
- Streamlit reruns that pattern constantly.
- Closing kills the embedded server; the next call may hit a dead port.
- A `.lock` file + multiple Python/Streamlit processes made it worse.

**Fix:**

- Process-wide **singleton** client (`get_qdrant_client()`), closed only at exit / explicit reset.
- Do not open-close per sidebar check.
- Upsert in batches (`UPSERT_BATCH_SIZE = 64`).
- If open fails, try deleting a stale `data/qdrant/.lock`.
- Operator step: one Streamlit process only; restart after the fix.

### 6.2 Symptom B — WinError 10054 + nvidia-smi idle

```text
[WinError 10054] An existing connection was forcibly closed by the remote host
```

**Likely layer:** Ollama HTTP during embedding, not Qdrant collections.

**Why GPU looked dead:**

- Original code called `embed_documents([...every chunk...])` in **one** shot for a full CUAD room (many thousands of chunks after splitting 510 contracts).
- On Windows that often dies mid-request. The connection resets before you see sustained GPU kernels — or Ollama never gets a healthy batch loaded.
- `ollama ps` was empty when we spot-checked; nothing was resident.

**Fix in flight:**

- Embed in small batches (`EMBED_BATCH_SIZE = 8`).
- Progress callback → console `[embed] 64/12000 — …` and Streamlit progress bar + rolling log.
- Clearer errors: which chunk range failed; reminder to keep `ollama serve` up.

**Still to verify when I re-run:**

- During embed batches, does `nvidia-smi` show Ollama/python GPU memory?
- If still 0%: Ollama may be on CPU (driver/CUDA/Ollama build). Separate debug from “connection reset.”

### 6.3 Visibility lesson

Silent spinners hide the failure mode. For long local jobs I now want:

- Stage name (`prepare` / `embed` / `qdrant` / `upsert` / `done`)
- `current/total`
- Console `print(..., flush=True)` (Streamlit UI can lag; terminal does not)
- A short note in the UI: watch `nvidia-smi` during **embed**, not during upsert

---

## 7. Mental model of the pipeline (after the dust)

```text
datasets/cuad/data/CUADv1.json     # full research dump (gitignored)
        │
        ▼  scripts/export_cuad.py --replace
dataroom/cuad_*.txt + MANIFEST.json
        │
        ▼  load_seed_records → steady chunking
list[record]  (source, section, chunk_index, text)
        │
        ▼  batched Ollama embeddings
vectors
        │
        ▼  shared local Qdrant client, batched upsert
collection diligence_dataroom
        │
        ├─ Evidence search (score_threshold)
        └─ LangGraph review → Gemini JSON → citation verify → audit.jsonl
```

Caches:

- Streamlit `@st.cache_data` on identical `(question, threshold)` in-session.
- Semantic cache collection in Qdrant for near-duplicate questions (wiped on rebuild).

---

## 8. What “done” means at this checkpoint

Shipped / decided:

- [x] Diligence Lens rebrand + CUAD as primary corpus  
- [x] Full export of 510 contracts into `dataroom/`  
- [x] Attribution / terms / privacy / gitignore hygiene  
- [x] Steady chunking  
- [x] Shared Qdrant client + batched upsert  
- [x] Batched embed + Streamlit/console progress (for the 10054 class of failures)  
- [x] Docker / README stubs for PoC packaging  

Still open when I sit down next:

- [ ] Successful full (or staged) index rebuild with progress logs watched end-to-end  
- [ ] Confirm whether Ollama is actually on GPU (`nvidia-smi` during embed)  
- [ ] Threshold / behavior tuning journal pass  
- [ ] Make Compose + host Ollama work for real  
- [ ] Portfolio page write-up once the demo is stable  

---

## 9. Interview fragments (steal these)

**Why not ContractNLI as the hero corpus?**  
“Great for NLI research; wrong task shape for multi-document diligence. I used CUAD under CC BY and attributed it.”

**What broke on Windows?**  
“Local Qdrant path-mode is a single embedded server. Opening a new client per Streamlit rerun killed the port. I moved to a process singleton and batched upserts.”

**Why batched embeddings?**  
“One `embed_documents` over the whole CUAD room reset the Ollama connection (WinError 10054). Small batches plus progress logs made the failure diagnosable; GPU usage only shows once batches actually run.”

**How do you avoid looking careless with data?**  
“Full dump gitignored under `datasets/`; regenerable `dataroom/` with MANIFEST; MIT for code; CC BY citation for CUAD; terms/privacy stubs that say this is not legal advice.”

---

## 10. File map for re-entry

| Path | Role |
|------|------|
| `portal.py` | Diligence Lens UI + index progress |
| `retrieval.py` | Chunking, singleton Qdrant, batched embed/upsert |
| `risk_graph.py` | LangGraph + Gemini finding |
| `scripts/export_cuad.py` | JSON/TXT → full `dataroom/` |
| `docs/ATTRIBUTION.md` | CUAD credit |
| `docs/GIT_HYGIENE.md` | What never to commit |
| `journal/diligence-lens-cuad-setup-guide.md` | Step checklist |
| `journal/diligence-lens-field-notes.md` | Portfolio-style brief |
| This file | Narrative dump of decisions and failures |

---

## 11. One-paragraph status for later

I rebranded the Nextwork RAG into Diligence Lens, abandoned an unavailable LegalBench-Deep link, standardized on CUAD from Atticus GitHub (JSON, not txt-in-zip), exported all 510 contracts into a regenerable dataroom with attribution, and learned the hard way that Streamlit + local Qdrant needs a singleton client while full-corpus Ollama embeds need small batches and visible progress — otherwise Windows closes the socket and `nvidia-smi` never lights up.

---

*End of draft dump. Append new incidents below rather than rewriting history.*

---

## Incident log

### 2026-10-22 — `progress_callback` TypeError after hot-reload

**Symptom.** UI error: `build_index() got an unexpected keyword argument 'progress_callback'`.

**What was actually true.** Disk `retrieval.py` already had `def build_index(records, progress_callback=None)`. `portal.py` correctly passed the kwarg. Streamlit had been started *before* that signature landed; it reloaded `portal` (new call site) but kept the old in-memory `retrieval.build_index` (no kwarg). Classic partial hot-reload skew.

**Fix (first attempt).** Rebuild path used `importlib.reload(retrieval)` — that later caused the lock incident below.

**Better fix.** Drop reload. After big signature changes, fully stop/start one Streamlit. Source already has `progress_callback`.

### 2026-10-22 — Embed finished, index NOT done (WinError 32 on `.lock`)

**Symptom.** Console showed `[embed] 49918/49918`, then:
`[WinError 32] ... being used by another process: 'data\\qdrant\\.lock'`.

**Verdict.** **Index was not done.** Embeddings sat in RAM only. Upsert never ran. No `[upsert]` / `[done]` lines = nothing durable in Qdrant for this rebuild.

**Why.** `importlib.reload(retrieval)` reset the module singleton (`_client = None`) while the *previous* `QdrantClient` still held the Windows lock. After ~50k embeds, `get_qdrant_client()` tried to open the path again → lock conflict. Also: opening Qdrant *after* embed wasted the whole embed run on a lock fail.

**How to verify (truth):**
- Console must show `[upsert] …` then `[done]`.
- Sidebar **Indexed points** > 0 and badge **Index ready**.
- Or: `get_index_stats()` → `ready=True`, `points` ≈ chunk count (~49918 for full CUAD room).

**Fix.** Removed reload. Open Qdrant **before** embed (fail fast). Sidebar shows point count. Ready = collection exists AND points > 0.

### 2026-10-22 — GPU was already on; util looked “low”

**Verify.** `ollama ps` showed `nomic-embed-text:v1.5 … 100% GPU` and `nvidia-smi` listed `llama-server.exe`. That is the truth for “is it on GPU?” — not a flat 80–100% util bar.

**Why util ~¼.** Model is ~335 MB; default embed batch was 8 chunks. RTX 3050 finishes each tiny batch fast, so average util looks low even though every token ran on GPU.

**Push.** Default `EMBED_BATCH_SIZE` 8 → **32** (env override). Upsert default 64 → **128**. Try 48/64 if stable; drop toward 16 if WinError 10054 returns.

### 2026-10-22 — 49859 points vs 49918 chunks (not a flaky upsert)

**Symptom.** Success toast + warning: point count 49859 ≠ 49918 embeds.

**Cause.** Content-addressed `uuid5(source:section:chunk_index:text)`. CUAD has repeated section/chunk twins (35 dup groups, 59 extra copies). Upsert overwrote same IDs → silent shrink. Rebuild alone would cost full embed time again and leave the same gap.

**Fix.** Point IDs include a per-build **ordinal** so every embedded record gets its own point. One clean rebuild after this fix; do not loop rebuilds hoping it heals.

### 2026-10-22 — Containerization staged, not forced mid-rebuild

Qdrant warned that path mode is not ideal above 20k points. For the PoC we keep path mode for the current rebuild. Future stack is ready: Compose `qdrant` + `diligence-lens`, client reads `QDRANT_URL` when set, runbook in `docs/DOCKER.md`. Cutover = new store + re-index; do not flip while host Streamlit holds `.lock`.

### 2026-10-22 — Clear + optimize (timeout was Gemini)

**Symptom.** Review “Request timed out” (client-side) after long wait; index showed 99777 points.

**Order of ops.** Wipe path store → export **80** CUAD → CLI rebuild **7840/7840** match → raise Gemini timeout to 240s, evidence limit 6, clip text 600 chars, default threshold 0.60, stage timings in console/UI. Sidebar **Clear index** + `scripts/clear_index.py` / `rebuild_index.py`.

**Full 510 later (portfolio demo).** When review is stable on the staged room:

```powershell
# stop Streamlit first
python scripts/clear_index.py
python scripts/export_cuad.py --replace
python scripts/rebuild_index.py
# or Build in UI — expect ~49918 points, not 99k
# optional: docker compose up --build  (see docs/DOCKER.md)
```
