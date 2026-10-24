import json
from pathlib import Path
from time import perf_counter

import streamlit as st

from retrieval import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DOCS_PATH,
    EMBED_BATCH_SIZE,
    build_index,
    clear_index,
    corpus_file_count,
    get_index_stats,
    load_seed_records,
    load_uploaded_records,
    search_evidence,
)
from risk_graph import DEFAULT_SCORE_THRESHOLD, run_review

# seed_documents disabled: Diligence Lens indexes ./dataroom (CUAD subset export).


@st.cache_data(
    ttl="15m",
    max_entries=32,
    scope="session",
    show_spinner="Checking caches or running the review...",
    show_time=True,
)
def cached_review(question: str, score_threshold: float) -> dict:
    return run_review(question, score_threshold)


def decision_badge(decision: str) -> None:
    colors = {
        "pass": "green",
        "concern": "orange",
        "conflict": "red",
        "insufficient_evidence": "gray",
    }
    label = decision.replace("_", " ")
    st.badge(
        label,
        icon=":material/gavel:",
        color=colors.get(decision, "blue"),
    )


def _plain_md(text: object) -> str:
    """Escape corpus/LLM text so Streamlit markdown does not treat $ as KaTeX."""
    return str(text or "").replace("$", r"\$")


def _write_plain(text: object) -> None:
    st.markdown(_plain_md(text))


def render_finding(review: dict, metrics: dict) -> None:
    usage = review.get("usage", {})
    semantic = review.get("semantic_cache", {})
    exact_hit = bool(metrics.get("cache_hit"))
    semantic_hit = bool(semantic.get("hit"))
    if exact_hit:
        cache_source = "Exact session hit"
    elif semantic_hit:
        cache_source = "Semantic hit"
    else:
        cache_source = "Miss to Gemini"

    with st.container(border=True):
        st.subheader("Run telemetry", anchor=False)
        cache_columns = st.columns(4)
        cache_columns[0].metric("Cache source", cache_source)
        score = semantic.get("similarity")
        cache_columns[1].metric(
            "Semantic score", "n/a" if score is None else f"{score:.3f}"
        )
        lookup = None if exact_hit else semantic.get("lookup_seconds")
        cache_columns[2].metric(
            "Cache lookup", "Skipped" if lookup is None else f"{lookup:.2f}s"
        )
        review_seconds = None if exact_hit else semantic.get("review_seconds")
        cache_columns[3].metric(
            "Miss workflow",
            "Skipped" if review_seconds is None else f"{review_seconds:.2f}s",
        )

        current_api_called = not (exact_hit or semantic_hit) and usage.get(
            "api_called", False
        )
        run_columns = st.columns(4)
        run_columns[0].metric(
            "Evidence passages", len(review.get("evidence", []))
        )
        run_columns[1].metric(
            "Gemini cached tokens",
            usage.get("total_cached_tokens", 0) if current_api_called else 0,
        )
        run_columns[2].metric(
            "Portal latency", f"{metrics.get('latency', 0):.2f}s"
        )
        run_columns[3].metric(
            "Gemini called", "Yes" if current_api_called else "No"
        )

        timings = review.get("timings") or {}
        if timings:
            stage_cols = st.columns(3)
            stage_cols[0].metric(
                "Retrieve",
                "n/a"
                if timings.get("retrieve_seconds") is None
                else f"{timings['retrieve_seconds']:.2f}s",
            )
            stage_cols[1].metric(
                "Gemini",
                "n/a"
                if timings.get("gemini_seconds") is None
                else f"{timings['gemini_seconds']:.2f}s",
            )
            stage_cols[2].metric(
                "Validate",
                "n/a"
                if timings.get("validate_seconds") is None
                else f"{timings['validate_seconds']:.2f}s",
            )

    finding = review.get("finding") or {}
    decision = finding.get("decision", "unknown")
    st.subheader("Finding", anchor=False)
    decision_badge(decision)
    st.caption(
        f"Severity `{finding.get('severity', 'n/a')}` · "
        f"Confidence `{finding.get('confidence', 'n/a')}`"
    )
    _write_plain(finding.get("summary") or "No summary recorded.")

    detail_tabs = st.tabs(
        [
            "Rationale",
            "Conflicts",
            "Gaps",
            "Actions",
            "Citations",
            "Evidence",
        ]
    )
    with detail_tabs[0]:
        _write_plain(finding.get("rationale") or "None recorded")
    with detail_tabs[1]:
        claims = finding.get("conflicting_claims") or ["None recorded"]
        for claim in claims if isinstance(claims, list) else [claims]:
            _write_plain(f"- {claim}")
    with detail_tabs[2]:
        gaps = finding.get("missing_evidence") or ["None recorded"]
        for gap in gaps if isinstance(gaps, list) else [gaps]:
            _write_plain(f"- {gap}")
    with detail_tabs[3]:
        actions = finding.get("recommended_actions") or ["None recorded"]
        for action in actions if isinstance(actions, list) else [actions]:
            _write_plain(f"- {action}")
    with detail_tabs[4]:
        citations = finding.get("citations") or []
        if not citations:
            st.info("No locally verified citations.", icon=":material/info:")
        else:
            for citation in citations:
                page = f" · page {citation['page']}" if citation.get("page") else ""
                with st.container(border=True):
                    st.markdown(
                        f"**{_plain_md(citation['source'])}** · "
                        f"{_plain_md(citation['section'])}{page}"
                    )
                    _write_plain(f'"{citation["quote"]}"')
    with detail_tabs[5]:
        evidence = review.get("evidence") or []
        if not evidence:
            st.warning(
                "No passages cleared the evidence score threshold.",
                icon=":material/warning:",
            )
        else:
            for number, item in enumerate(evidence, start=1):
                score = item.get("score")
                score_label = (
                    f"{score:.3f}" if isinstance(score, (int, float)) else "n/a"
                )
                label = (
                    f"{number}. {item.get('source', 'unknown')} · "
                    f"score {score_label}"
                )
                with st.expander(label, expanded=number <= 2):
                    meta = []
                    if item.get("section"):
                        meta.append(f"section: {_plain_md(item['section'])}")
                    if item.get("page"):
                        meta.append(f"page: {item['page']}")
                    if meta:
                        st.caption(" · ".join(meta))
                    _write_plain(item.get("text", ""))

    if review.get("report"):
        st.download_button(
            "Download Markdown report",
            data=review["report"],
            file_name="diligence-lens-report.md",
            mime="text/markdown",
            icon=":material/download:",
        )


st.set_page_config(
    page_title="Diligence Lens",
    page_icon=":material/policy:",
    layout="wide",
)
st.title("Diligence Lens")
st.caption(
    "Local-first multi-contract due-diligence RAG · quality-gated retrieval · "
    "schema-validated findings"
)
with st.expander("Data attribution & notices", expanded=False):
    st.markdown(
        """
Corpus subsets may include **CUAD** (Contract Understanding Atticus Dataset),
© The Atticus Project, licensed under **CC BY 4.0**.
Paper: Hendrycks et al., *CUAD: An Expert-Annotated NLP Dataset for Legal
Contract Review*, NeurIPS 2021 ([arXiv:2103.06268](https://arxiv.org/abs/2103.06268)).
Source: [GitHub](https://github.com/The-Atticus-Project/cuad) ·
[Atticus site](https://www.atticusprojectai.org/cuad/).

This app is an **engineering proof of concept**, not legal advice.
See `docs/ATTRIBUTION.md`, `docs/TERMS.md`, and `docs/PRIVACY.md`.
        """.strip()
    )

with st.sidebar:
    st.header("Data room", anchor=False)
    st.caption(
        f"Active corpus: `{DOCS_PATH}/` · chunk `{CHUNK_SIZE}` / "
        f"overlap `{CHUNK_OVERLAP}` · embed batch `{EMBED_BATCH_SIZE}`"
    )
    doc_count = corpus_file_count()
    st.metric("Corpus files", doc_count)
    index_stats = get_index_stats()
    st.metric("Indexed points", index_stats["points"])
    if index_stats["ready"]:
        st.badge("Index ready", icon=":material/check_circle:", color="green")
        st.caption(
            f"`{index_stats['collection']}` · {index_stats['points']} points · "
            f"mode `{index_stats.get('mode', '?')}` · `{index_stats['path']}`"
        )
    elif index_stats["error"]:
        st.badge("Index locked / error", icon=":material/warning:", color="orange")
        st.caption(index_stats["error"])
    else:
        st.badge("Index not built", icon=":material/pending:", color="orange")
        st.caption(
            "Embed-only does not count. Ready only after `[upsert]` + `[done]` "
            "in the console."
        )

    uploaded_files = st.file_uploader(
        "Add contracts",
        type=["pdf", "md", "txt"],
        accept_multiple_files=True,
        help="Optional uploads merged with ./dataroom on rebuild.",
    )
    if st.button(
        "Clear index",
        icon=":material/delete:",
        width="stretch",
        help="Wipes diligence + semantic-cache collections (or path store). "
        "Use before a clean rebuild — do not stack rebuilds on a bloated index.",
    ):
        try:
            result = clear_index()
            cached_review.clear()
            st.session_state["review_cache_keys"] = set()
            st.success(
                f"Index cleared ({result['mode']}). "
                f"Deleted: {', '.join(result['deleted']) or 'nothing'}.",
                icon=":material/check:",
            )
            st.rerun()
        except Exception as error:
            st.error(
                f"Clear failed: {error}. Stop other Streamlit processes and retry.",
                icon=":material/error:",
            )
    if st.button(
        "Build or rebuild index",
        type="primary",
        icon=":material/database:",
        width="stretch",
    ):
        progress_bar = st.progress(0.0, text="Starting index rebuild…")
        log_box = st.empty()
        log_lines: list[str] = []

        def on_progress(*, stage: str, current: int, total: int, message: str) -> None:
            fraction = 0.0 if total <= 0 else min(1.0, current / total)
            # Weight stages so the bar keeps moving across embed → upsert.
            stage_weight = {
                "prepare": 0.02,
                "embed": 0.75,
                "qdrant": 0.80,
                "upsert": 1.0,
                "done": 1.0,
            }
            if stage == "embed":
                shown = 0.05 + 0.70 * fraction
            elif stage == "upsert":
                shown = 0.80 + 0.20 * fraction
            else:
                shown = stage_weight.get(stage, fraction)
            progress_bar.progress(
                min(1.0, shown),
                text=f"{stage}: {message}",
            )
            line = f"`{stage}` {current}/{total} — {message}"
            log_lines.append(line)
            log_box.markdown(
                "#### Index log\n" + "\n".join(f"- {item}" for item in log_lines[-40:])
            )

        try:
            with st.status(
                "Building diligence index…", expanded=True
            ) as status:
                st.write(
                    "Loading `dataroom/` files. Console also prints `[embed]` / "
                    "`[upsert]` lines — watch `nvidia-smi` during embed batches."
                )
                st.write(
                    f"Embed batch=`{EMBED_BATCH_SIZE}` (env `EMBED_BATCH_SIZE`). "
                    "`ollama ps` should say **100% GPU**. Low nvidia-smi util is "
                    "normal for a small embed model — raise the batch for more "
                    "load. Keep one Streamlit only."
                )
                records = load_seed_records() + load_uploaded_records(uploaded_files)
                st.write(f"Loaded **{len(records)}** chunks from disk.")
                print(f"[prepare] loaded {len(records)} chunks", flush=True)
                summary = build_index(records, progress_callback=on_progress)
                cached_review.clear()
                st.session_state["review_cache_keys"] = set()
                status.update(label="Index rebuild complete", state="complete")
            progress_bar.progress(1.0, text="Done")
            final_stats = get_index_stats()
            st.success(
                f"Indexed {summary['records']} chunks · "
                f"{summary['dimensions']} dims · "
                f"{final_stats['points']} points on disk.",
                icon=":material/check:",
            )
            if final_stats["points"] != summary["records"]:
                st.warning(
                    f"Point count ({final_stats['points']}) ≠ embedded chunks "
                    f"({summary['records']}). Usually duplicate chunk IDs "
                    f"(fixed in current code) — one clean rebuild needed, not "
                    f"a loop. Do not keep rebuilding hoping it self-heals.",
                    icon=":material/warning:",
                )
        except Exception as error:
            progress_bar.progress(0.0, text="Failed")
            st.error(str(error), icon=":material/error:")
            print(f"[error] index rebuild failed: {error}", flush=True)

search_tab, review_tab, audit_tab, quality_tab = st.tabs(
    [
        ":material/search: Evidence search",
        ":material/gavel: Diligence review",
        ":material/history: Audit trail",
        ":material/science: Quality lab",
    ]
)

with search_tab:
    st.subheader("Retrieve across the data room", anchor=False)
    with st.form("evidence_search_form"):
        search_question = st.text_input(
            "Diligence question",
            value="Where do contracts leave liability uncapped?",
        )
        search_threshold = st.slider(
            "Minimum match score", 0.0, 1.0, DEFAULT_SCORE_THRESHOLD, 0.05
        )
        search_submitted = st.form_submit_button(
            "Search evidence",
            icon=":material/search:",
        )

    if search_submitted:
        try:
            results = search_evidence(
                search_question,
                limit=8,
                score_threshold=search_threshold,
            )
            if not results:
                st.warning(
                    "No evidence cleared the current match threshold. "
                    "Lower the threshold to view looser matches.",
                    icon=":material/warning:",
                )
            else:
                metric_row = st.columns(2)
                metric_row[0].metric("Accepted passages", len(results))
                metric_row[1].metric(
                    "Top match score", f"{results[0]['score']:.3f}"
                )
                for number, result in enumerate(results, start=1):
                    label = (
                        f"{number}. {result['source']} · "
                        f"score {result['score']:.3f}"
                    )
                    with st.expander(label, expanded=number <= 2):
                        if result.get("section"):
                            st.caption(
                                f"section: {_plain_md(result['section'])}"
                            )
                        _write_plain(result["text"])
        except Exception as error:
            st.error(str(error), icon=":material/error:")

with review_tab:
    st.subheader("Structured diligence finding", anchor=False)
    with st.form("risk_review_form"):
        review_question = st.text_area(
            "Diligence question",
            value=(
                "Do any master agreements conflict with side letters on "
                "liability caps or change-of-control terms?"
            ),
            height=100,
        )
        review_threshold = st.slider(
            "Minimum evidence score", 0.0, 1.0, DEFAULT_SCORE_THRESHOLD, 0.05
        )
        review_submitted = st.form_submit_button(
            "Run diligence review",
            type="primary",
            icon=":material/play_arrow:",
        )

    if review_submitted:
        cache_key = (review_question.strip(), review_threshold)
        known_keys = st.session_state.setdefault("review_cache_keys", set())
        cache_hit = cache_key in known_keys
        started = perf_counter()
        try:
            with st.status(
                "Running the diligence RAG pipeline...",
                expanded=True,
            ) as status:
                st.write("Checking exact and semantic caches.")
                st.write("A miss continues to retrieval and Gemini.")
                review_result = cached_review(*cache_key)
                status.update(
                    label="Review complete",
                    state="complete",
                )
            st.session_state["atlas_review"] = review_result
            known_keys.add(cache_key)
            st.session_state["review_metrics"] = {
                "cache_hit": cache_hit,
                "latency": perf_counter() - started,
            }
        except Exception as error:
            st.error(str(error), icon=":material/error:")

    review = st.session_state.get("atlas_review")
    if review:
        render_finding(review, st.session_state.get("review_metrics", {}))

with audit_tab:
    audit_path = Path("audit/runs.jsonl")
    if not audit_path.exists():
        st.info(
            "Run a diligence review to create the audit trail.",
            icon=":material/info:",
        )
    else:
        events = [
            json.loads(line)
            for line in audit_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        st.caption(f"Showing the latest {min(10, len(events))} of {len(events)} runs.")
        for event in reversed(events[-10:]):
            decision = event.get("finding", {}).get("decision", "unknown")
            with st.expander(
                f"{event.get('timestamp_utc', '?')} · {decision}"
            ):
                st.json(event)

with quality_tab:
    st.subheader("Boundary checks", anchor=False)
    st.caption(
        "Cases are diligence-shaped. Recalibrate after swapping corpora or thresholds."
    )
    cases = [
        {
            "case": "Cross-document conflict probe",
            "question": (
                "Do any contracts conflict on liability caps or "
                "change-of-control consent?"
            ),
            "expected": "conflict",
        },
        {
            "case": "Missing clause / weak support",
            "question": (
                "Which agreements omit a change-of-control clause entirely?"
            ),
            "expected": "insufficient_evidence",
        },
        {
            "case": "Unrelated abstention",
            "question": "What is the diameter of Mars?",
            "expected": "insufficient_evidence",
        },
    ]
    st.dataframe(cases, hide_index=True, width="stretch", alt="Quality lab cases")
    selected = st.selectbox(
        "Quality case",
        range(len(cases)),
        format_func=lambda index: cases[index]["case"],
    )
    quality_threshold = st.slider(
        "Quality-test threshold", 0.0, 1.0, DEFAULT_SCORE_THRESHOLD, 0.05
    )

    if st.button(
        "Run selected quality check",
        icon=":material/science:",
    ):
        case = cases[selected]
        result = cached_review(case["question"], quality_threshold)
        observed = result["finding"]["decision"]
        st.write({"expected": case["expected"], "observed": observed})
        if observed == case["expected"]:
            st.success(
                "Observed decision matches the expected boundary.",
                icon=":material/check_circle:",
            )
        else:
            st.warning(
                "Recalibrate the threshold or inspect retrieved evidence.",
                icon=":material/warning:",
            )
