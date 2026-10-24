# Diligence Lens — containerization (future work)

Local path-mode Qdrant is the **default** for day-to-day laptop work.  
Compose + Qdrant **server** is ready when you want a portable stack and to leave the “>20k points in local mode” warning behind.

## Target shape

| Service | Role |
|---------|------|
| `diligence-lens` | Streamlit (`portal.py`) |
| `qdrant` | Official Qdrant image, HTTP on `:6333` |
| Ollama | Prefer **host** install + GPU (`host.docker.internal`); optional Compose service later |

CUAD full dump stays on the host under `datasets/` (gitignored). Export to `dataroom/` on the host; Compose mounts it read-only.

## Env switches

| Variable | Local default | Compose |
|----------|---------------|---------|
| *(unset)* `QDRANT_URL` | path `data/qdrant` | `http://qdrant:6333` |
| `QDRANT_PATH` | `data/qdrant` | unused when URL set |
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | `http://host.docker.internal:11434` |
| `GEMINI_API_KEY` | `.env` | `.env` via `env_file` |
| `EMBED_BATCH_SIZE` | `32` | same |

## When you flip the switch

1. Finish a local smoke index if you still need it (path DB).
2. `docker compose up --build`
3. Open http://localhost:8501
4. **Rebuild the index** inside the UI — path-mode vectors do **not** auto-migrate into the Qdrant volume.
5. Confirm sidebar: Indexed points match chunk count; mode is URL (`get_index_stats()["mode"] == "url"`).

## Why not change mid-rebuild today

Path → server is a new store. Starting Compose while a host Streamlit holds `data/qdrant/.lock` causes pain. One world at a time: finish local rebuild, then cut over.

## Windows GPU note

Host Ollama already uses the RTX. Putting Ollama in Compose on Windows is optional and often worse for GPU passthrough — keep embeddings on the host unless you move to Linux + NVIDIA Container Toolkit.
