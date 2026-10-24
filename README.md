# Diligence Lens

A side project for hands-on practice with multi-contract due-diligence RAG.

The goal is a local-first app where you index a contract data room, retrieve passages above a similarity threshold, and produce a structured finding (decision, severity, confidence, verified quotes) through a LangGraph workflow and Gemini. Cache and latency stay visible in the UI. An optional approval gate can pause release for a human reviewer.

> **Not legal advice.** Engineering experiment only. See [`docs/TERMS.md`](docs/TERMS.md) and [`docs/PRIVACY.md`](docs/PRIVACY.md).

## What I'm building toward

- Quality-gated retrieval over many contracts in one room
- Cross-document conflict framing and citation checks against retrieved text
- Measurable retrieval improvements (evals in progress; hybrid search and reranking next)
- Clear attribution for research corpora and a path to containerized runs

## Stack

| Layer | Choice |
|-------|--------|
| UI | Streamlit |
| Orchestration | LangGraph |
| Generation | Google Gemini (structured JSON) |
| Embeddings | Ollama `nomic-embed-text` (local) |
| Vector store | Qdrant (local path mode; Compose can use a server URL) |
| Primary corpus | [CUAD](https://github.com/The-Atticus-Project/cuad) subset in `dataroom/` |

## Attribution

Contract texts may come from CUAD (CC BY 4.0, The Atticus Project). Cite Hendrycks et al., NeurIPS 2021. Full notes: [`docs/ATTRIBUTION.md`](docs/ATTRIBUTION.md) · [`NOTICE`](NOTICE).

## Quick start (local)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env   # add GEMINI_API_KEY
# Install + pull Ollama model: nomic-embed-text:v1.5
```

### Corpus: download CUAD, export into the data room

```powershell
git clone https://github.com/The-Atticus-Project/cuad.git datasets/cuad
# If contracts live in data.zip, extract it under datasets/cuad/

python scripts/export_cuad.py --replace
streamlit run portal.py
```

This exports CUAD contracts from `CUADv1.json` into `dataroom/`.  
In the sidebar, use **Build or rebuild index** (embedding can take a while on a laptop GPU).

Retrieval eval scripts live under `evaluation/` and `scripts/`.

## Layout

```text
portal.py                       # Diligence Lens UI
approval_portal.py              # Human approval gate
retrieval.py                    # Chunking, Qdrant, semantic cache
risk_graph.py                   # LangGraph review workflow
evaluation/                     # Retrieval eval cases
scripts/evaluate_retrieval.py   # Eval runner
evidence_evals.py               # Quick top-k probe dump
dataroom/                       # Active indexed subset (generated)
datasets/cuad/                  # Full CUAD download (gitignored)
docs/                           # Attribution, terms, privacy
scripts/export_cuad.py
Dockerfile
docker-compose.yml              # app + Qdrant (optional later)
docs/DOCKER.md                  # container notes
```

## Containers (optional later)

`Dockerfile` and `docker-compose.yml` sketch a portable stack. The app accepts `QDRANT_URL` for server mode, or path mode when that variable is unset. Cutover notes: [`docs/DOCKER.md`](docs/DOCKER.md). Re-index after switching to server mode; the path database does not migrate automatically.

## Roadmap

- [x] Local Streamlit app + CUAD export path
- [x] Section/paragraph chunking
- [x] Attribution / terms / privacy stubs
- [x] Compose + `QDRANT_URL` switch documented
- [ ] Retrieval evals and iterative retrieval quality work
- [ ] Hybrid BM25 / RRF and reranking
- [ ] Threshold and scale pass
- [ ] `docker compose up` end-to-end on a clean machine

## License

Project code: MIT. See [`LICENSE`](LICENSE).  
CUAD contract files: **CC BY 4.0** (not covered by the MIT grant).
