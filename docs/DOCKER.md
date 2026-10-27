# Diligence Lens: Docker Compose

Local path-mode Qdrant (`data/qdrant`) is the day-to-day default.  
Compose runs **server** Qdrant + the Streamlit app. Embeddings stay on **host
Ollama** (Windows/Mac GPU-friendly).

Prove the [README local happy path](../README.md) once before relying on Compose.

## Preconditions

1. Docker Desktop running (Windows: enable host networking helpers as usual).
2. Host Ollama with `nomic-embed-text` pulled; API on `http://127.0.0.1:11434`.
3. `.env` with `GEMINI_API_KEY` (from `.env.example`).
4. Host `dataroom/` already exported — Compose mounts it **read-only**:

```powershell
python scripts/export_cuad.py --limit 80 --replace
```

Stop host Streamlit first if it holds `data/qdrant/.lock`.

## Clone → env → up → review

```powershell
# From repo root (after clone + .venv optional; Compose builds its own image)
copy .env.example .env
# Edit .env → GEMINI_API_KEY=...

# Host embeddings (outside Compose)
ollama pull nomic-embed-text

# 80-doc room on the host (mounted into the container)
python scripts/export_cuad.py --limit 80 --replace

docker compose up --build
```

Open http://localhost:8501

| Service | URL |
|---------|-----|
| Portal | http://localhost:8501 |
| Qdrant HTTP | http://localhost:6333 |

### First run inside the UI

1. Sidebar → **Build or rebuild index** (path-mode vectors do **not** migrate).
2. Confirm indexed points ≈ chunk count; mode should be URL (`QDRANT_URL`).
3. First MiniLM download inside the container can be slow — wait once.

### Smoke (same as local)

| Question | Expect |
|----------|--------|
| Vericel exclusive purchase duration (default portal Q) | `pass` |
| *What is the diameter of Mars?* | `insufficient_evidence`, 0 sources |

## Env switches

| Variable | Local default | Compose |
|----------|---------------|---------|
| *(unset)* `QDRANT_URL` | path `data/qdrant` | `http://qdrant:6333` |
| `QDRANT_PATH` | `data/qdrant` | unused when URL set |
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | `http://host.docker.internal:11434` |
| `GEMINI_API_KEY` | `.env` | `.env` via `env_file` |
| `GEMINI_MODEL` | `gemini-3.5-flash-lite` | same |
| `REVIEW_RERANK_THRESHOLD` | `4.0` | same |
| `REVIEW_EVIDENCE_LIMIT` | `8` | same |

## Why re-index after Compose

Path mode and server mode are different stores. Starting Compose while host
Streamlit holds `data/qdrant/.lock` will conflict. Finish or stop the local app,
then cut over and rebuild in the container UI.

## Windows note

Host Ollama already uses the RTX. Putting Ollama in Compose on Windows is
optional and often worse for GPU. Keep embeddings on the host unless you move
to Linux with the NVIDIA Container Toolkit.

Optional in-compose Ollama service is commented out in `docker-compose.yml`.

## Image notes

- Image installs **CPU** `torch` (PyTorch CPU index) then `requirements.txt`.
  Rerank is tiny; GPU embeds stay on **host Ollama**. Avoids the huge CUDA wheel.
- `sentence-transformers` MiniLM weights download on first rerank inside the container.
- `dataroom/`, `datasets/`, and `data/` stay off the image (see `.dockerignore`);
  mount or generate on the host.
- Do not bake secrets into the image.
- First `docker compose build` can still take several minutes (CPU torch ~200MB).

## Troubleshooting

| Symptom | Likely fix |
|---------|------------|
| Embed / Ollama connection errors from app | Confirm host `ollama serve`; Compose `OLLAMA_HOST=http://host.docker.internal:11434`; `extra_hosts` present |
| Empty index after up | Rebuild in UI — server volume starts empty |
| ImportError for hybrid/rerank modules | Rebuild image (`docker compose build --no-cache`) — Dockerfile must COPY pipeline `.py` files |
| Huge first build | Normal: torch + sentence-transformers wheels |
| Lock / WinError on host path DB | Stop local Streamlit before Compose; do not share path DB with server mode |
