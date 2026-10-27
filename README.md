# Diligence Lens

Local-first multi-contract due-diligence RAG PoC. Index a CUAD slice, retrieve
with hybrid + MiniLM rerank + diversity, cut weak evidence, then ask Gemini for
a structured finding with citation checks. Cache and latency stay visible.
Optional approval gate pauses release for a human.

> **Not legal advice.** Engineering experiment only. See
> [`docs/TERMS.md`](docs/TERMS.md) and [`docs/PRIVACY.md`](docs/PRIVACY.md).

**Ship path:** local Python + host Ollama is the supported demo. Docker Compose
is optional packaging only ([`docs/DOCKER.md`](docs/DOCKER.md)) — not required.

## PoC knobs (frozen for this demo)

| Knob | Value |
|------|--------|
| Active room | first **80** CUAD contracts |
| Index size | ~**7783** points after a clean rebuild (= chunk count; confirm in sidebar) |
| Embeddings | Ollama `nomic-embed-text` (or `nomic-embed-text:v1.5`) on **host** |
| Generator | `GEMINI_MODEL` (default `gemini-3.5-flash-lite`) |
| Review cut | MiniLM `rerank_score` **≥ 4.0**, final evidence **8** |
| Empty pack | `insufficient_evidence` (no Gemini call) |
| Vector store | Qdrant **path mode** → `data/qdrant` (`QDRANT_URL` unset) |

## Prerequisites

- Python 3.12+ recommended
- [Ollama](https://ollama.com/) running locally
- Google AI API key for Gemini (`GEMINI_API_KEY`)
- ~disk for CUAD clone + embeds; GPU optional but speeds Ollama embeds

## Quick start (local — do this)

PowerShell from an empty working directory, then the repo root.

### 1. Clone and env

```powershell
git clone https://github.com/HGK7/diligence-lens.git
cd diligence-lens
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
# Edit .env → set GEMINI_API_KEY=...
```

Leave `QDRANT_URL` **unset** in `.env` for path mode.

### 2. Ollama embed model

```powershell
ollama pull nomic-embed-text
# Optional check:
# Invoke-WebRequest http://127.0.0.1:11434/api/tags
```

### 3. CUAD → 80-doc data room

```powershell
New-Item -ItemType Directory -Force -Path datasets | Out-Null
git clone https://github.com/The-Atticus-Project/cuad.git datasets/cuad
# If needed: extract datasets/cuad/data.zip so CUADv1.json exists
# Expect: datasets/cuad/data/CUADv1.json

python scripts/export_cuad.py --limit 80 --replace
# Expect: ~80 dataroom/cuad_*.txt + dataroom/MANIFEST.json
```

### 4. Index, then portal (one process rule)

Path-mode Qdrant allows **one** Python process on `data/qdrant` at a time.

**Either** rebuild from the CLI **or** from the UI — not both at once, and not
two Streamlit windows.

```powershell
# Option A — CLI index, then UI (recommended first time)
python scripts/rebuild_index.py
streamlit run portal.py

# Option B — empty store, build inside the sidebar
streamlit run portal.py
# → sidebar → Build or rebuild index
```

Open http://localhost:8501

**Verify**

| Check | Expect |
|-------|--------|
| Sidebar corpus files | ~80 |
| Sidebar indexed points | equals rebuild chunk count (~7783) |
| Mode caption | `path` · `data/qdrant` |

### 5. Smoke (honest for this room)

| Question | Expect |
|----------|--------|
| Default portal question (Vericel exclusive supply duration) | `pass` in a few seconds |
| *What is the diameter of Mars?* | `insufficient_evidence`, **0** sources |

Do **not** demo master-vs-side-letter conflicts on this slice — no side letters
in the 80-doc room. An abstain there is correct.

## Path-mode Qdrant (locks)

Embedded Qdrant locks `data/qdrant`. Symptoms: sidebar “Index locked / error”,
`already accessed by another instance`, or `WinError 32` on `.lock`.

**Avoid**

- Two `streamlit run portal.py` sessions on the same repo
- `python scripts/rebuild_index.py` / `clear_index.py` while Streamlit is up
- Deleting `data/qdrant/.lock` while a process is still alive

**Recover**

1. Sidebar → **Recover index connection** (after a hot-reload glitch).
2. If that fails: stop Streamlit (terminal Ctrl+C), confirm no other Python
   holds the folder, start `streamlit run portal.py` once.
3. CLI wipe only with Streamlit **stopped**:
   `python scripts/clear_index.py` then `python scripts/rebuild_index.py`.

Hot reload after editing `.py` files should reuse one client; if the sidebar
still shows a lock error, use **Recover** or a full Streamlit restart.

## Attribution

Contract texts may come from CUAD (CC BY 4.0, The Atticus Project). Cite
Hendrycks et al., NeurIPS 2021.
[`docs/ATTRIBUTION.md`](docs/ATTRIBUTION.md) · [`NOTICE`](NOTICE).

## Layout

```text
portal.py                 # Diligence Lens UI
approval_portal.py        # Human approval gate
retrieval.py              # Hybrid search orchestration
qdrant_store.py           # Path/URL client (reload-safe singleton)
risk_graph.py             # LangGraph review workflow
hybrid.py / reranking.py  # RRF + MiniLM
evaluation/               # Retrieval cases (~30)
scripts/export_cuad.py
scripts/rebuild_index.py
dataroom/                 # Active subset (gitignored)
datasets/cuad/            # Full CUAD dump (gitignored)
```

## Optional: Docker Compose

Not required for the demo. See [`docs/DOCKER.md`](docs/DOCKER.md) if you want
server Qdrant + a containerized UI later. Re-index after switching modes; the
path database does not migrate.

## Roadmap

- [x] Local Streamlit + CUAD export path
- [x] Hybrid BM25 / RRF, MiniLM rerank, diversity
- [x] Part 9 v1: rerank ≥4, limit 8, empty → abstain
- [x] Documented local happy path (primary ship)
- [ ] Portfolio WIP page after local path reproduces twice on a fresh clone
- [ ] Compose E2E (optional; deferred)

## License

Project code: MIT. See [`LICENSE`](LICENSE).  
CUAD contract files: **CC BY 4.0** (not covered by the MIT grant).
