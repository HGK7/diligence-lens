# Diligence Lens: containerization (future work)

Local path-mode Qdrant is the default for day-to-day laptop work.  
Compose with a Qdrant server is ready when you want a portable stack and to avoid the local-mode warning above ~20k points.

## Target shape

| Service | Role |
|---------|------|
| `diligence-lens` | Streamlit (`portal.py`) |
| `qdrant` | Official Qdrant image, HTTP on `:6333` |
| Ollama | Prefer host install + GPU (`host.docker.internal`); optional Compose service later |

Keep the full CUAD dump on the host under `datasets/` (gitignored). Export to `dataroom/` on the host. Compose mounts that folder read-only.

## Env switches

| Variable | Local default | Compose |
|----------|---------------|---------|
| *(unset)* `QDRANT_URL` | path `data/qdrant` | `http://qdrant:6333` |
| `QDRANT_PATH` | `data/qdrant` | unused when URL set |
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | `http://host.docker.internal:11434` |
| `GEMINI_API_KEY` | `.env` | `.env` via `env_file` |
| `EMBED_BATCH_SIZE` | `32` | same |

## When you flip the switch

1. Finish a local smoke index if you still need the path DB.
2. `docker compose up --build`
3. Open http://localhost:8501
4. Rebuild the index in the UI. Path-mode vectors do not move into the Qdrant volume on their own.
5. Confirm the sidebar: indexed points match chunk count; mode is URL (`get_index_stats()["mode"] == "url"`).

## Why not change mid-rebuild

Path mode and server mode are different stores. Starting Compose while host Streamlit holds `data/qdrant/.lock` will conflict. Finish the local rebuild, then cut over.

## Windows GPU note

Host Ollama already uses the RTX. Putting Ollama in Compose on Windows is optional and often worse for GPU passthrough. Keep embeddings on the host unless you move to Linux with the NVIDIA Container Toolkit.
