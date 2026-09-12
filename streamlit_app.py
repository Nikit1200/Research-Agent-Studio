"""
Modern Streamlit UI for the multi-agent research pipeline.

Run with:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import contextlib
import io
import json
import re
import time
from datetime import datetime
from typing import Any

import streamlit as st

from pipeline import PipelineStageError, run_research_pipeline


st.set_page_config(
    page_title="Research Agent Studio",
    page_icon=":material/travel_explore:",
    layout="wide",
    initial_sidebar_state="expanded",
)


def initialize_state() -> None:
    st.session_state.setdefault("latest_result", None)
    st.session_state.setdefault("latest_log", "")
    st.session_state.setdefault("run_history", [])
    st.session_state.setdefault("active_history_index", None)


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def extract_score(feedback: str) -> str:
    match = re.search(r"Score:\s*([0-9.]+\s*/\s*10)", feedback, flags=re.IGNORECASE)
    return match.group(1).replace(" ", "") if match else "Pending"


def run_pipeline_with_log(topic: str, progress_callback: Any = None) -> tuple[dict[str, Any], str, float]:
    started = time.perf_counter()
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        result = run_research_pipeline(topic, progress=progress_callback)
    elapsed = time.perf_counter() - started
    return result, stream.getvalue(), elapsed


def as_markdown_report(topic: str, result: dict[str, Any], log: str) -> str:
    report = clean_text(result.get("report"))
    feedback = clean_text(result.get("feedback"))
    search = clean_text(result.get("search_result"))
    scraped = clean_text(result.get("scraped_content"))
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return f"""# Research report: {topic}

Generated: {generated_at}

## Final report

{report}

## Critic feedback

{feedback}

## Search results

```text
{search}
```

## Scraped content

```text
{scraped}
```

## Pipeline log

```text
{log}
```
"""


def render_theme() -> None:
    st.markdown(
        """
        <style>
            :root {
                --studio-border: rgba(148, 163, 184, 0.28);
                --studio-surface: rgba(255, 255, 255, 0.055);
                --studio-surface-strong: rgba(255, 255, 255, 0.09);
                --studio-text-muted: rgba(226, 232, 240, 0.72);
                --studio-accent: #14b8a6;
                --studio-accent-2: #f59e0b;
            }

            .stApp {
                background:
                    linear-gradient(120deg, rgba(20, 184, 166, 0.12), transparent 34%),
                    linear-gradient(250deg, rgba(245, 158, 11, 0.11), transparent 36%),
                    #111827;
            }

            [data-testid="stSidebar"] {
                background: rgba(15, 23, 42, 0.95);
                border-right: 1px solid var(--studio-border);
            }

            .studio-hero {
                padding: 1.35rem 1.5rem;
                border: 1px solid var(--studio-border);
                border-radius: 8px;
                background: linear-gradient(135deg, rgba(15, 23, 42, 0.96), rgba(31, 41, 55, 0.82));
                margin-bottom: 1rem;
            }

            .studio-eyebrow {
                color: var(--studio-accent);
                font-size: 0.82rem;
                letter-spacing: 0;
                font-weight: 700;
                margin-bottom: 0.25rem;
            }

            .studio-title {
                font-size: clamp(2rem, 4vw, 3.4rem);
                line-height: 1.03;
                font-weight: 780;
                margin: 0;
            }

            .studio-subtitle {
                max-width: 58rem;
                color: var(--studio-text-muted);
                font-size: 1.02rem;
                margin: 0.7rem 0 0;
            }

            .agent-strip {
                display: grid;
                grid-template-columns: repeat(4, minmax(0, 1fr));
                gap: 0.75rem;
                margin: 0.75rem 0 1rem;
            }

            .agent-card {
                border: 1px solid var(--studio-border);
                border-radius: 8px;
                background: var(--studio-surface);
                padding: 0.9rem;
                min-height: 6rem;
            }

            .agent-card strong {
                display: block;
                font-size: 0.95rem;
                margin-bottom: 0.25rem;
            }

            .agent-card span {
                color: var(--studio-text-muted);
                font-size: 0.86rem;
            }

            .result-shell {
                border: 1px solid var(--studio-border);
                border-radius: 8px;
                background: rgba(15, 23, 42, 0.56);
                padding: 1rem;
            }

            @media (max-width: 900px) {
                .agent-strip {
                    grid-template-columns: repeat(2, minmax(0, 1fr));
                }
            }

            @media (max-width: 560px) {
                .agent-strip {
                    grid-template-columns: 1fr;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> tuple[str, str, bool, bool]:
    with st.sidebar:
        st.title("Research Agent Studio")
        st.caption("Search, read, write, and critique from one focused workspace.")

        with st.form("research_form", border=False):
            topic = st.text_area(
                "Research topic",
                placeholder="Example: Recent advances in AI agents for financial research",
                height=120,
                key="topic_input",
            )
            depth = st.segmented_control(
                "Workspace mode",
                options=["Balanced", "Deep review", "Fast scan"],
                default="Balanced",
                key="workspace_mode",
            )
            submitted = st.form_submit_button(
                "Run research pipeline",
                icon=":material/play_arrow:",
                type="primary",
                width="stretch",
            )

        clear_clicked = st.button(
            "Clear current result",
            icon=":material/refresh:",
            width="stretch",
        )

        st.caption(
            "Mode is saved with each run for organization. The current pipeline executes the same terminal workflow."
        )

        st.subheader("Recent runs")
        if not st.session_state.run_history:
            st.caption("Your completed runs will appear here.")
        else:
            for index, item in enumerate(reversed(st.session_state.run_history[-8:])):
                real_index = len(st.session_state.run_history) - 1 - index
                label = f"{item['created_at']} - {item['topic'][:34]}"
                if st.button(label, key=f"history_{real_index}", width="stretch"):
                    st.session_state.latest_result = item["result"]
                    st.session_state.latest_log = item["log"]
                    st.session_state.active_history_index = real_index
                    st.rerun()

    return clean_text(topic), clean_text(depth), submitted, clear_clicked


def render_header() -> None:
    st.markdown(
        """
        <div class="studio-hero">
            <div class="studio-eyebrow">Multi-agent research command center</div>
            <h1 class="studio-title">Research Agent Studio</h1>
            <p class="studio-subtitle">
                Launch your existing terminal pipeline from a polished workspace, inspect each agent output,
                download the report, and keep a short session history for comparison.
            </p>
        </div>
        <div class="agent-strip">
            <div class="agent-card"><strong>Search agent</strong><span>Finds recent and reliable source candidates.</span></div>
            <div class="agent-card"><strong>Reader agent</strong><span>Selects relevant URLs and gathers deeper page content.</span></div>
            <div class="agent-card"><strong>Writer chain</strong><span>Turns collected evidence into a structured report.</span></div>
            <div class="agent-card"><strong>Critic chain</strong><span>Scores the output and highlights improvement areas.</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_state() -> None:
    left, right = st.columns([1.2, 0.8], gap="large", vertical_alignment="top")
    with left:
        with st.container(border=True):
            st.subheader("Start a research run", anchor=False)
            st.write(
                "Enter a topic in the sidebar and run the pipeline. Results will appear here as a report, "
                "critic review, raw search evidence, scraped content, and captured terminal log."
            )
    with right:
        with st.container(border=True):
            st.metric("Agents", "4")
            st.metric("Outputs", "5")
            st.metric("History", f"{len(st.session_state.run_history)} runs")


def render_result(topic: str, mode: str, result: dict[str, Any], log: str, elapsed: float | None = None) -> None:
    report = clean_text(result.get("report"))
    feedback = clean_text(result.get("feedback"))
    search = clean_text(result.get("search_result"))
    scraped = clean_text(result.get("scraped_content"))
    score = extract_score(feedback)

    metric_cols = st.columns(4)
    metric_cols[0].metric("Critic score", score)
    metric_cols[1].metric("Report length", f"{len(report.split()):,} words")
    metric_cols[2].metric("Source text", f"{len(search.split()):,} words")
    metric_cols[3].metric("Scraped text", f"{len(scraped.split()):,} words")

    if elapsed is not None:
        st.caption(f"Completed in {elapsed:.1f}s using {mode.lower()} workspace mode.")

    tabs = st.tabs(
        [
            "Report",
            "Critic feedback",
            "Search evidence",
            "Scraped content",
            "Console log",
            "Export",
        ]
    )

    with tabs[0]:
        with st.container(border=True):
            st.subheader("Final report", anchor=False)
            st.markdown(report or "_No report returned._")

    with tabs[1]:
        with st.container(border=True):
            st.subheader("Critic feedback", anchor=False)
            st.markdown(feedback or "_No critic feedback returned._")

    with tabs[2]:
        sources = result.get("sources", [])
        if sources:
            st.dataframe(sources, hide_index=True, key="research_sources")
        st.text_area("Search evidence", value=search, height=320, key="search_evidence")

    with tabs[3]:
        st.text_area("Scraped content", value=scraped, height=420)

    with tabs[4]:
        st.code(log or "No console output captured.", language="text")

    with tabs[5]:
        markdown_payload = as_markdown_report(topic or "research-topic", result, log)
        json_payload = json.dumps(result, indent=2, ensure_ascii=False)
        with st.container(horizontal=True):
            st.download_button(
                "Download report",
                data=markdown_payload,
                file_name="research_report.md",
                mime="text/markdown",
                icon=":material/download:",
                width="content",
            )
            st.download_button(
                "Download raw JSON",
                data=json_payload,
                file_name="research_pipeline_result.json",
                mime="application/json",
                icon=":material/data_object:",
                width="content",
            )


initialize_state()
render_theme()
topic, mode, submitted, clear_clicked = render_sidebar()
render_header()

if clear_clicked:
    st.session_state.latest_result = None
    st.session_state.latest_log = ""
    st.session_state.active_history_index = None
    st.rerun()

if submitted:
    if not topic:
        st.warning("Enter a research topic before starting the pipeline.", icon=":material/warning:")
    else:
        with st.status("Running the multi-agent pipeline", expanded=True) as status:
            progress_slot = st.empty()
            completed_stages: list[str] = []

            def show_progress(stage: str, state: str, message: str) -> None:
                if state == "complete" and stage not in completed_stages:
                    completed_stages.append(stage)
                lines = [f"✓ {item} completed" for item in completed_stages]
                prefix = "❌" if state == "error" else "…"
                lines.append(f"{prefix} {stage}: {message}")
                progress_slot.markdown("  \n".join(lines))

            try:
                result, log, elapsed = run_pipeline_with_log(topic, show_progress)
            except PipelineStageError as exc:
                status.update(label=f"{exc.stage} failed", state="error", expanded=True)
                st.error(f"{exc.stage}: {exc}", icon=":material/error:")
            except Exception:
                status.update(label="Unexpected pipeline error", state="error", expanded=True)
                st.error("An unexpected error occurred. Review the console log and try again.", icon=":material/error:")
            else:
                status.update(label="Pipeline complete", state="complete", expanded=False)
                st.session_state.latest_result = result
                st.session_state.latest_log = log
                st.session_state.active_history_index = len(st.session_state.run_history)
                st.session_state.run_history.append(
                    {
                        "topic": topic,
                        "mode": mode,
                        "created_at": datetime.now().strftime("%H:%M"),
                        "elapsed": elapsed,
                        "result": result,
                        "log": log,
                    }
                )
                st.success("Research run completed.", icon=":material/check_circle:")

active_item = None
if st.session_state.active_history_index is not None:
    if 0 <= st.session_state.active_history_index < len(st.session_state.run_history):
        active_item = st.session_state.run_history[st.session_state.active_history_index]

if active_item:
    render_result(
        active_item["topic"],
        active_item["mode"],
        active_item["result"],
        active_item["log"],
        active_item["elapsed"],
    )
elif st.session_state.latest_result:
    render_result(topic, mode, st.session_state.latest_result, st.session_state.latest_log)
else:
    render_empty_state()
