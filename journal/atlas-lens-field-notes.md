# Field notes: Atlas Lens, then ContractNLI

*Working draft. I will fill this in as I go. Written from a Cursor session where I built and debugged the portal, then started learning how to load an external evaluation corpus.*

---

## Why I am writing this

I shipped Atlas Lens through a guided Nextwork project: a local-first vendor risk portal that indexes documents, retrieves passages above a similarity threshold, and runs a LangGraph review with Gemini for a structured finding (decision, severity, confidence, citations). It grew out of an earlier self-correcting RAG pipeline for Project Atlas runbooks.

I can click the buttons. I still cannot always explain the code out loud. These notes are how I force the story into my own words: what I built, what broke, what I am testing next, and how I load data I did not write myself.

---

## What Atlas Lens is supposed to do

A vendor pack lands as a pile of polished policies. Deadlines and commitments disagree across files. Atlas Lens is meant to:

1. Index synthetic vendor docs (and uploads: PDF, Markdown, text).
2. Quality-gate retrieval with a configurable score threshold so weak matches are not treated as evidence.
3. Produce a structured risk finding with quotes that must appear in the local passages.
4. Surface cache / latency / usage signals in Streamlit.
5. (Secret mission) pause for human approval before release.

My eight seed files live under `vendor_docs/` (overview, security policy, SLA, incident response, DPA, retention, BCP, access control). Those were enough to demo conflict questions like incident notification vs Priority 1 response times.

---

## Bugs I hit before I trusted the UI

### `'tuple' object has no attribute 'payload'`

Qdrant’s `query_points` returns a `QueryResponse`. The hits live on `response.points`. If you iterate the response object itself, Pydantic yields field tuples like `('points', [...])`, and `.payload` explodes. Fix: loop `response.points`, where each item is a `ScoredPoint`.

### Risk Review showed metrics but no results

After adding cache / quality telemetry, the review tab painted latency and “Evidence passages: N” and then stopped. The finding body, citations, retrieved passages, and download button never rendered. Restored that display block under the metrics so a review actually shows what was retrieved and decided.

Small side fix: the Markdown report generator had `## Citations` inside a loop, so the heading repeated for every section.

---

## Choosing something harder than my seed pack

The seed pack is synthetic and tiny on purpose. I wanted an external corpus people already use for contract understanding, so I can say “I tested retrieval and decisions against a known dataset,” not only against docs I already know by heart.

From browsing options (CUAD, LegalBench-RAG, Dropbox VSMC, CISA SCRM templates, various synthetic generators), I started with **ContractNLI**:

- Document-level NLI on NDAs.
- Fixed hypotheses labeled `Entailment` / `Contradiction` / `NotMentioned`.
- Evidence as spans in the contract text.
- Released under CC BY 4.0; cite Koreeda & Manning, Findings of EMNLP 2021.

I downloaded it under `vendor_docs/contract-nli/`. I have not finished loading it into the index yet.

---

## ContractNLI so far (README only)

I am learning the format before I write a loader. From their README:

- Top level: `documents` (the NDAs) and `labels` (the shared hypothesis bank).
- For indexing Atlas Lens, the field that matters first is each document’s `text`.
- Annotations live at `document['annotation_sets'][0]['annotations']`.
- Each annotation has `choice` and `spans` (indices into the document’s `spans` list of character offsets).
- `NotMentioned` means empty evidence spans — close in spirit to Atlas Lens abstaining with `insufficient_evidence`.

I am treating “load” as two different jobs:

1. **Corpus load** — get NDA `text` into files (or the indexer) so Search / Risk Review can retrieve.
2. **Eval load** — map `labels[*].hypothesis` + `choice` to questions and expected decisions later.

Checkpoint questions I still owe myself before touching `test.json` / train splits in earnest:

- [ ] Three NLI labels, and which Atlas decision is closest to `NotMentioned`
- [ ] Where hypothesis text lives (`labels` vs `documents`)
- [ ] Which field is the body text for indexing
- [ ] Whether annotation `spans: [12, 13, 91]` means character offsets or indices into the span list
- [ ] Why exceptions / negation make contracts nasty for NLI

---

## How I plan to load ContractNLI into Atlas Lens

*Open. Update this section as I actually do it.*

### Planned steps (draft)

1. Confirm which JSON files I have (train / dev / test) and how large they are.
2. Start with **one** NDA from the split I care about — export `text` to something `load_seed_records` / the uploader already understands (`.md` or `.txt`).
3. Rebuild the index from the sidebar (or a small script if seed globbing should ignore the nested `contract-nli` tree).
4. Ask one ContractNLI hypothesis in Evidence Search; check score threshold behavior.
5. Run the same hypothesis through Risk Review; compare the portal decision to gold `choice` informally.
6. Only then scale to more documents or wire a proper eval harness.

### Not decided yet

- Whether ContractNLI stays nested under `vendor_docs/` or moves to a separate `datasets/` folder so seed rebuild does not accidentally index the whole JSON zip every time.
- Whether I export many `.md` files or teach `retrieval.py` to read ContractNLI JSON directly.
- How strict I will be mapping `Entailment` / `Contradiction` / `NotMentioned` onto Atlas `pass` / `conflict` / `insufficient_evidence` (the schemas are related, not identical).

### Log of what I actually did

| Date | What I did | Result |
|------|------------|--------|
| *(today)* | Built / ran Atlas Lens; fixed payload + review display bugs | Portal usable again |
| *(today)* | Downloaded ContractNLI; read README with a guided walkthrough | Format understood at concept level |
| | | |
| | | |

---

## Questions I still cannot answer cleanly in an interview

*Paste fog here as I find it. Be honest.*

- [ ] Embeddings + Qdrant local path mode (what lives on disk, when to delete collections)
- [ ] LangGraph state: what each node reads/writes
- [ ] Gemini schema validation + local citation check
- [ ] Streamlit `st.cache_data` vs semantic cache in Qdrant
- [ ] Approval gate / interrupt resume
- [ ] …

---

## Resources I am citing for myself

- ContractNLI paper / dataset: Koreeda & Manning, Findings of EMNLP 2021; local copy under `vendor_docs/contract-nli/contract-nli/`; license CC BY 4.0.
- Dropbox Vendor Security Model Contract (considered, not loaded yet): https://github.com/dropbox/vsmc
- CUAD / LegalBench-RAG (parked for later stress tests).
- This learning thread: Cursor chat covering the Nextwork brief, the payload bug, the missing Risk Review render, dataset recommendations, and the README-only ContractNLI lesson.

---

## Next session

Continue the ContractNLI loading tutorial from the JSON on disk. Goal for the next sitting: export **one** NDA, rebuild the index without breaking the seed pack story, and write one paragraph here about what the portal retrieved.

---

*End of draft. Keep appending below rather than rewriting history.*
