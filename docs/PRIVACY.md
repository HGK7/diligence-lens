# Privacy notice (stub)

**Status:** Draft stub for the Diligence Lens proof of concept. Review before exposing the app beyond your own machine.

## Local-first default

- Document indexing uses a **local** Qdrant store under `data/qdrant/` (path mode).
- Embeddings are intended to run via **local Ollama** (`nomic-embed-text`).
- Uploaded files and `dataroom/` contents stay on the host unless you copy them elsewhere.

## What may leave your machine

If you run a diligence review with Gemini (or another cloud LLM):

- The **question** and **retrieved evidence passages** are sent to that API
- Usage metadata may be logged by the provider under their privacy policy
- Configure keys only via environment / `.env` (never commit secrets)

Exact and semantic caches may store prior questions and report payloads on disk. Treat `data/` and `audit/` as sensitive if you indexed private contracts.

## Research corpora

Public datasets such as **CUAD** are third-party research materials. Using them does not grant this project rights to your private documents. See `docs/ATTRIBUTION.md`.

## Logs and audit trail

Successful reviews may append events to `audit/runs.jsonl` (question, finding, sources, timing). Delete or rotate that file if it contains sensitive text.

## Cookies / accounts

The default Streamlit PoC does **not** implement end-user accounts or analytics cookies. If you add auth, hosting, or tracking later, update this notice.

## Contact

For privacy questions about *your* deployment of this fork, contact the operator of that deployment (you, until published otherwise).
