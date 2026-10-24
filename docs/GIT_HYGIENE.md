# Git hygiene

## Never commit

| Path / pattern | Why |
|----------------|-----|
| `.env`, `.streamlit/secrets.toml`, `credentials*.json`, keys | API secrets |
| `datasets/*` (except `.gitkeep`) | Full CUAD / other corpus downloads |
| `dataroom/*` (except `.gitkeep`) | Active indexed subset (regenerate) |
| `data/`, `audit/` | Local Qdrant + review logs (may contain doc text) |
| `vendor_docs/contract-nli/`, `vendor_docs/cnli_*` | Large experiment dumps |
| `*.zip` | Dataset archives (e.g. CUAD `data.zip`) |
| `.venv/`, `__pycache__/` | Local Python env |

## Safe to commit

- Source: `portal.py`, `retrieval.py`, `risk_graph.py`, scripts, tests
- `docs/` (attribution, terms, privacy stubs)
- `.env.example` (placeholders only — no real keys)
- `datasets/.gitkeep`, `dataroom/.gitkeep`
- Optional: small synthetic seed markdown under `vendor_docs/0*.md` if you want demos without CUAD

## Before every push

```powershell
git status
git check-ignore -v datasets/cuad dataroom/.gitkeep .env
# Confirm .env is ignored and no contract dumps are staged
```

If something sensitive was staged by mistake:

```powershell
git reset HEAD -- path\to\file
# If already committed locally (not pushed): remove it from history before you publish
```

## Regenerating ignored corpora

```powershell
git clone https://github.com/The-Atticus-Project/cuad.git datasets/cuad
# extract data.zip if needed
python scripts/export_cuad.py --limit 10 --replace
```
