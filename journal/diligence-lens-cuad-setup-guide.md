# Guided lab: Diligence Lens + CUAD (download all, index a slice)

Goal: keep the **full** CUAD dump under `datasets/cuad/`, export a **small**
active set into `dataroom/`, rebuild the index, then raise `--limit` as you scale.

Source repo: https://github.com/The-Atticus-Project/cuad  
License: **CC BY 4.0** — keep [`docs/ATTRIBUTION.md`](../docs/ATTRIBUTION.md).

---

## Phase 0 — Legal / attribution checklist

- [x] Read `docs/ATTRIBUTION.md` (CUAD citation)
- [x] Skim `docs/TERMS.md` and `docs/PRIVACY.md` (stubs — not counsel-approved)
- [x] Confirm you will **not** commit `datasets/cuad/` (see `.gitignore`)
- [x] Plan to keep `dataroom/MANIFEST.json` with any frozen demo subset

Required paper citation when you publish:

> Hendrycks, Burns, Chen, Ball. CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review. NeurIPS 2021. https://arxiv.org/abs/2103.06268

---

## Phase 1 — Download everything once

From the project root:

```powershell
New-Item -ItemType Directory -Force -Path datasets | Out-Null
git clone https://github.com/The-Atticus-Project/cuad.git datasets/cuad
```

The GitHub repo includes `data.zip` (~18 MB). After extract you get JSON, not loose `.txt`:

```powershell
Get-ChildItem datasets\cuad\data
# Expect CUADv1.json (~40 MB) — 510 contracts inside
```

`Get-ChildItem … -Filter *.txt` under `datasets/cuad` returning nothing is normal.
Phase 2 materializes `.txt` files into `dataroom/`.

- [x] `datasets/cuad/data/CUADv1.json` present

---

## Phase 2 — Export the full CUAD room

GitHub `data.zip` gives `CUADv1.json` (510 contracts), not loose `.txt` files.
Export writes one `.txt` per contract into `dataroom/` (full corpus by default):

```powershell
.\.venv\Scripts\Activate.ps1
python scripts/export_cuad.py --replace
```

Expect:

- ~510 `dataroom/cuad_*.txt` files
- `dataroom/MANIFEST.json` (`limit: "all"`, attribution, source path)

Optional slice only if you need it later: `--limit 10`.

Confirm:

```powershell
(Get-ChildItem dataroom\cuad_*.txt).Count
```

- [x] Full export done (~510 files)

---

## Phase 3 — Staged index (optimize loop) — done

```powershell
# Stop Streamlit first if path-mode lock is held
python scripts/clear_index.py
python scripts/export_cuad.py --limit 80 --replace
python scripts/rebuild_index.py
streamlit run portal.py
```

Expect **~80** corpus files and **7840** indexed points (match embed count).

Review defaults: threshold **0.60**, evidence limit **6**, Gemini timeout **240s**.
Console prints `[review] retrieve / gemini / total`. Sidebar has **Clear index**.

- [x] Wiped bloated 99777-point store  
- [x] Staged 80-doc rebuild (7840 == 7840)  
- [x] Gemini timeout / payload / timings  

---

## Phase 4 — Full 510 rebuild (portfolio demo — later)

When staged review is stable:

```powershell
# Stop Streamlit
python scripts/clear_index.py
python scripts/export_cuad.py --replace
python scripts/rebuild_index.py
# Expect ~49918 points — never stack rebuilds without Clear
```

Do **not** delete `datasets/cuad/` between runs.

- [ ] Full 510 export + clean rebuild  
- [ ] Threshold / behavior pass on full room  

---

## Phase 5 — Containerization (after local smoke)

See `docs/DOCKER.md`. `QDRANT_URL` switch already in code.

```powershell
# Later:
# docker compose up --build
```

- [ ] Local staged smoke first  
- [ ] Compose + re-index (path DB does not migrate)  

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `Could not find CUAD .txt` | Expected on GitHub zip — use `CUADv1.json` via `export_cuad.py` |
| Index picks Northstar / vendor docs | Ensure `dataroom/` has CUAD files; `seed_documents()` stays commented |
| GPU OOM on embed | Close other GPU apps; staged `--limit 80` first |
| Points ≈ 2× chunks | Old+new ID schemes stacked — **Clear index**, one rebuild |
| Gemini client timeout | Raise `GEMINI_TIMEOUT_SECONDS`; check `[review] gemini` timing |
| `dial tcp … refused` / WinError 32 `.lock` | One Streamlit only; Clear index / wipe `data/qdrant` after stop |

---

## Recruiter one-liner

> Diligence Lens indexes a CC BY–attributed CUAD subset locally, quality-gates retrieval, and returns schema-validated diligence findings with citation checks, cache/latency telemetry, and an optional human approval gate — container stubs included for PoC packaging.
