# Diligence Lens — field notes

*Working draft in the same spirit as my [Profile Manager](https://hrishikeshgk.dev/projects/profile-manager) write-ups: what I am building, what I am solving, architecture choices, what shipped, what is still open.*

---

## The brief

Diligence Lens is a local-first Streamlit app for multi-contract due-diligence RAG. One data room holds the contracts I choose to index. Search quality-gates weak matches. A LangGraph review asks Gemini for a structured finding, then verifies citation quotes against retrieved text. Cache and latency stay visible. An approval portal can pause release for a human.

I am treating this as a **portfolio PoC**, not a practice throwaway: attribution, terms/privacy stubs, reproducible corpus export, chunking you can explain, and Docker stubs toward a packaged demo.

## What I'm solving

The earlier Nextwork path (Atlas Lens / vendor pack) taught the pipeline, but a tiny synthetic pack does not signal scale or corpus discipline. ContractNLI was useful as a load idea and weak as a product story (single-NDA NLI ≠ cross-document diligence).

I want a demo where I can say: I downloaded a real research corpus under a clear license, kept the full dump offline, indexed a controlled subset, and can scale the subset without rewriting the app.

## Corpus choice

**CUAD** — Contract Understanding Atticus Dataset (Atticus Project, CC BY 4.0).  
GitHub: https://github.com/The-Atticus-Project/cuad  
Paper: Hendrycks et al., NeurIPS 2021.

Pattern:

1. Clone / extract CUAD under `datasets/cuad/` (gitignored). GitHub `data.zip` → `CUADv1.json` (not loose `.txt`).  
2. `scripts/export_cuad.py --replace` writes **all ~510** contracts into `dataroom/` + `MANIFEST.json`.  
3. Rebuild Qdrant with batched Ollama embeds + shared local client (see narrative dump for the Windows failure story).

Long-form learning notes: [`diligence-lens-build-narrative.md`](diligence-lens-build-narrative.md).

Hands-on retrieval curriculum: [`retrieval-quality-learning-guide.md`](retrieval-quality-learning-guide.md).

## How it works

Sidebar rebuilds the index from `dataroom/` (+ optional uploads). Evidence search retrieves above a score threshold. Diligence review runs exact session cache → semantic cache → LangGraph retrieve/analyze/report/audit. Findings show badges, telemetry, and tabbed rationale / conflicts / citations / evidence. Attribution sits in an expander and in `docs/`.

Chunking is steady rather than naive fixed slices: prefer sections, pack paragraphs under a size budget, then sentence-aware windows for oversized blocks (`CHUNK_SIZE=800`, overlap `120`).

## Architecture choices

- **Local embeddings + path Qdrant** so the PoC runs on an RTX-class laptop without a hosted vector DB.  
- **Separate full dump vs active room** so “download everything” does not mean “embed everything on day one.”  
- **CC BY attribution as product chrome**, not a footnote I forget — `NOTICE`, `docs/ATTRIBUTION.md`, UI expander, export manifest.  
- **MIT for code**, CUAD texts remain CC BY — spelled out in `LICENSE`.  
- **Compose stubs early** (Profile Manager taught me to show packaging intent before polish) while local smoke stays the gate.

## What shipped (this pass)

- Cleared bloated index; staged **80** CUAD / **7840** points (matched)  
- Review tune: Gemini timeout 240s, evidence 6×600 chars, threshold 0.60, stage timings  
- **Clear index** UI + `scripts/clear_index.py` / `rebuild_index.py`  
- Full 510 + Compose cutover documented as later demo steps  
- Rebrand surfaces: Diligence Lens portal + approval gate copy  
- CUAD export script with seeded subsets + manifest  
- Attribution / terms / privacy stubs + NOTICE + LICENSE note  
- README PoC stub + Dockerfile / compose stubs  
- Setup guide: `journal/diligence-lens-cuad-setup-guide.md`  
- Steady chunking in `retrieval.py`  
- Streamlit theme tokens in `.streamlit/config.toml`  
- **Retrieval eval harness**: grounded `evaluation/retrieval_cases.json` (10), `scripts/evaluate_retrieval.py`, probe dump CLI  
- Learning guide Part 1 expanded with full probe → label → score → freeze loop  

## Still open

- [x] Staged ~80 CUAD index + dense baseline scored  
- [ ] Inspect 0-recall cases; Part 2 chunk diagnosis  
- [ ] Hybrid BM25 + RRF (only after frozen baseline)  
- [ ] Threshold tuning journal pass  
- [ ] Make Compose work with host Ollama end-to-end  
- [ ] Public GitHub baseline + portfolio WIP page with local demo path  

## Log

| Date | What I did | Result |
|------|------------|--------|
| | Wired Diligence Lens + CUAD export/attribution/Docker stubs | In repo |
| 2026-10-24 | Grounded eval set + dense harness; Part 1 guide rewrite | Recall@10 ≈ 0.80, MRR@10 ≈ 0.80 on 10 cases |
| | | |

## What I'd do again

Keep license obligations next to the download instructions, not in a slide I never write. Separate “corpus I own on disk” from “corpus I embed today.” Ship stubs for packaging while the retrieval path is still honest locally.
