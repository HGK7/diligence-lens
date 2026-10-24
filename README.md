# Diligence Lens

**Status: work in progress** — usable local demo; dense retrieval baseline measured; hybrid/rerank still on the roadmap.

Local-first **multi-contract due-diligence RAG** proof of concept.

Open a Streamlit portal, index a contract data room, retrieve passages above a similarity threshold, and produce a schema-validated finding (decision, severity, confidence, verified quotes) via a LangGraph workflow and Gemini. Cache and latency signals stay visible in the UI. An optional approval gate pauses release for a human reviewer.

> **Not legal advice.** Engineering demo only. See [`docs/TERMS.md`](docs/TERMS.md) and [`docs/PRIVACY.md`](docs/PRIVACY.md).

## Why this exists

Built as a portfolio-grade PoC: quality-gated retrieval, cross-document conflict framing, citation checks, telemetry, human-in-the-loop approval, and a path to containerized runs — not a thin ChatPDF clone.

## Baseline (dense-only, 80-contract slice)

Frozen eval set: [`evaluation/retrieval_cases.json`](evaluation/retrieval_cases.json) (10 manually labeled questions).

| Metric | Score |
|--------|-------|
| Recall@10 | ~0.80 |
| MRR@10 | ~0.80 |
| Index | ~80 CUAD contracts / 7840 chunks |
| Retriever | Qdrant + Ollama `nomic-embed-text:v1.5` |

```powershell
python scripts\evaluate_retrieval.py
```

How the harness was built (probe → label from real files → score → freeze):  
[`journal/retrieval-quality-learning-guide.md`](journal/retrieval-quality-learning-guide.md) Part 1.

## Stack

| Layer | Choice |
|-------|--------|
| UI | Streamlit |
| Orchestration | LangGraph |
| Generation | Google Gemini (structured JSON) |
| Embeddings | Ollama `nomic-embed-text` (local GPU-friendly) |
| Vector store | Qdrant (local path mode; Compose → server URL) |
| Primary corpus | [CUAD](https://github.com/The-Atticus-Project/cuad) subset in `dataroom/` |

## Attribution

Contract texts may come from **CUAD** (CC BY 4.0, The Atticus Project). Cite Hendrycks et al., NeurIPS 2021. Full notes: [`docs/ATTRIBUTION.md`](docs/ATTRIBUTION.md) · [`NOTICE`](NOTICE).

## Git hygiene

Datasets, `.env`, Qdrant/`audit` logs, and corpus exports are gitignored. See [`docs/GIT_HYGIENE.md`](docs/GIT_HYGIENE.md). Commit `.env.example` only — never real keys or `datasets/cuad/`.

## Quick start (local)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env   # add GEMINI_API_KEY
# Install + pull Ollama model: nomic-embed-text:v1.5
```

### Corpus: download all CUAD, index a slice

```powershell
git clone https://github.com/The-Atticus-Project/cuad.git datasets/cuad
# If contracts live in data.zip, extract it under datasets/cuad/

python scripts/export_cuad.py --replace
streamlit run portal.py
```

Exports all ~510 CUAD contracts from `CUADv1.json` into `dataroom/`.  
In the sidebar: **Build or rebuild index** (full embed may take a while on laptop GPU).

## Project layout (PoC)

```text
portal.py                       # Diligence Lens UI
approval_portal.py              # Human approval gate
retrieval.py                    # Chunking, Qdrant, semantic cache
risk_graph.py                   # LangGraph review workflow
evaluation/retrieval_cases.json # Frozen retrieval gold labels
scripts/evaluate_retrieval.py   # Recall@K / MRR harness
evidence_evals.py               # Quick top-k probe dump
dataroom/                       # Active indexed subset (generated)
datasets/cuad/                  # Full CUAD download (gitignored)
docs/                           # Attribution, terms, privacy
scripts/export_cuad.py
Dockerfile
docker-compose.yml              # app + Qdrant (future cutover)
docs/DOCKER.md                  # container runbook
```

## Containerization (ready as future work)

Stack is sketched in `Dockerfile` + `docker-compose.yml`. App code already accepts **`QDRANT_URL`** (server) or path mode when unset.

| Now (laptop) | Later (`docker compose up --build`) |
|--------------|-------------------------------------|
| Path Qdrant under `data/qdrant` | `qdrant` service + `QDRANT_URL=http://qdrant:6333` |
| Host Ollama + GPU | Same via `host.docker.internal` |
| Host Streamlit | `diligence-lens` image on `:8501` |

Cutover notes (migrate ≠ copy): [`docs/DOCKER.md`](docs/DOCKER.md). Re-index after flipping to server mode — path DB does not auto-migrate.

## Roadmap (PoC hardening)

- [x] Rebrand to Diligence Lens + CUAD export path
- [x] Steady section/paragraph chunking
- [x] Attribution / terms / privacy stubs
- [x] Compose + `QDRANT_URL` switch documented (cut over when ready)
- [x] Retrieval eval harness + dense baseline (~0.80 Recall@10 / MRR@10)
- [ ] Chunk diagnosis + hybrid BM25 / RRF (guide Parts 2–4)
- [ ] Smoke / scale ladder + threshold pass
- [ ] `docker compose up` end-to-end on a clean machine
- [ ] Public demo URL / portfolio write-up

## License

Project code: MIT — see [`LICENSE`](LICENSE).  
CUAD contract files: **CC BY 4.0** (not covered by the MIT grant).
