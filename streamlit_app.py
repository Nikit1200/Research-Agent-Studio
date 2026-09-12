"""Professional Streamlit workspace for the Research Agent Studio pipeline."""
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

st.set_page_config(page_title="Research Agent Studio", page_icon=":material/travel_explore:", layout="wide")
STAGES = ("Search Agent", "Reader Agent", "Writer Agent", "Critic Agent")
ACADEMIC_SOURCES = {"OpenAlex", "arXiv"}


def initialize_state() -> None:
    for key, value in {
        "latest_result": None, "latest_log": "", "latest_elapsed": None,
        "run_history": [], "active_history_index": None,
        "agent_status": {stage: ("waiting", "Waiting") for stage in STAGES},
    }.items():
        st.session_state.setdefault(key, value)


def text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def score(feedback: str) -> str:
    found = re.search(r"Score:\s*([0-9.]+\s*/\s*10)", feedback, re.I)
    return found.group(1).replace(" ", "") if found else "Not scored"


def run_pipeline(topic: str, callback: Any) -> tuple[dict[str, Any], str, float]:
    started, output = time.perf_counter(), io.StringIO()
    with contextlib.redirect_stdout(output):
        result = run_research_pipeline(topic, progress=callback)
    return result, output.getvalue(), time.perf_counter() - started


def resource_kind(item: dict[str, Any]) -> str:
    return "Academic" if item.get("source") in ACADEMIC_SOURCES else "Web"


def matching_resources(items: list[dict[str, Any]], query: str, kind: str, ordering: str) -> list[dict[str, Any]]:
    query = query.casefold().strip()
    filtered = []
    for item in items:
        haystack = " ".join([item.get("title", ""), item.get("source", ""), item.get("abstract", ""), " ".join(item.get("authors", []))]).casefold()
        if (not query or query in haystack) and (kind == "All" or resource_kind(item) == kind):
            filtered.append(item)
    if ordering == "Newest":
        return sorted(filtered, key=lambda item: item.get("year") or 0, reverse=True)
    if ordering == "Most cited":
        return sorted(filtered, key=lambda item: item.get("citations") or 0, reverse=True)
    return filtered


def render_sidebar() -> bool:
    with st.sidebar:
        st.title("Research Agent Studio")
        st.caption("Multi-agent AI research assistant")
        st.divider()
        st.subheader("Workspace")
        st.caption("Search, read, write, and critique evidence in one workflow.")
        clear = st.button("Clear current result", icon=":material/refresh:", width="stretch")
        st.divider()
        st.subheader("Recent research")
        if not st.session_state.run_history:
            st.caption("Completed research runs will appear here.")
        for position, item in enumerate(reversed(st.session_state.run_history[-8:])):
            index = len(st.session_state.run_history) - 1 - position
            if st.button(f"{item['created_at']} · {item['topic'][:32]}", key=f"history_{index}", width="stretch"):
                st.session_state.active_history_index = index
                st.session_state.latest_result = item["result"]
                st.session_state.latest_log = item["log"]
                st.session_state.latest_elapsed = item["elapsed"]
                st.rerun()
        st.divider()
        st.caption("Uses Gemini, OpenAlex, arXiv, and optional Tavily.")
    return clear


def render_header() -> tuple[str, bool]:
    st.title("Research Agent Studio", anchor=False)
    st.caption("AI-powered multi-agent research assistant")
    st.markdown("Search. Analyze. Synthesize. Critique.")
    with st.form("research_form", border=True):
        topic = st.text_area("Research topic", placeholder="Enter a research topic, e.g. Impact of Generative AI on Education", height=100, key="topic_input")
        submit = st.form_submit_button("Generate research", icon=":material/auto_awesome:", type="primary", width="stretch")
    return text(topic), submit


def render_pipeline() -> None:
    st.subheader("Research pipeline", anchor=False)
    icons = {"waiting": ":material/radio_button_unchecked:", "running": ":material/progress_activity:", "complete": ":material/check_circle:", "error": ":material/error:"}
    for column, stage in zip(st.columns(4), STAGES):
        state, message = st.session_state.agent_status.get(stage, ("waiting", "Waiting"))
        with column.container(border=True):
            st.markdown(f"{icons[state]} **{stage}**")
            st.caption(message)


def render_resources(result: dict[str, Any]) -> None:
    items = result.get("sources") or []
    st.subheader(f"Collected resources ({len(items)})", anchor=False)
    st.caption("Filtering happens locally; it does not make another research request.")
    if not items:
        st.info("No research resources collected yet. Run a research query to discover relevant sources.", icon=":material/library_books:")
        return
    a, b, c = st.columns([0.5, 0.25, 0.25])
    query = a.text_input("Search collected resources", placeholder="Title, author, source…", key="resource_query")
    kind = b.segmented_control("Resource type", ["All", "Academic", "Web"], default="All", key="resource_kind") or "All"
    ordering = c.selectbox("Sort by", ["Research order", "Newest", "Most cited"], key="resource_sort")
    visible = matching_resources(items, query, kind, ordering)
    st.caption(f"Showing {len(visible)} of {len(items)} resources")
    if not visible:
        st.info("No collected resources match these filters.", icon=":material/filter_alt_off:")
    for item in visible:
        with st.container(border=True):
            left, right = st.columns([0.8, 0.2], vertical_alignment="top")
            with left:
                st.markdown(f"#### {text(item.get('title')) or 'Untitled resource'}")
                st.markdown(f":blue-badge[{resource_kind(item)}] :gray-badge[{text(item.get('source')) or 'Unknown source'}]")
            with right:
                if item.get("url"):
                    st.link_button("Open source", item["url"], icon=":material/open_in_new:", width="stretch")
            facts = []
            if item.get("authors"): facts.append(f"**Authors:** {', '.join(item['authors'])}")
            if item.get("year"): facts.append(f"**Published:** {item['year']}")
            if item.get("citations") is not None: facts.append(f"**Citations:** {item['citations']}")
            if facts: st.caption(" · ".join(facts))
            if item.get("abstract"): st.write(text(item["abstract"]))
            with st.expander("View source details", icon=":material/article:"):
                if item.get("doi"): st.markdown(f"**DOI:** {item['doi']}")
                if item.get("url"): st.markdown(f"**Original URL:** {item['url']}")


def render_report(result: dict[str, Any], topic: str, log: str, elapsed: float | None) -> None:
    report, feedback, items = text(result.get("report")), text(result.get("feedback")), result.get("sources") or []
    metrics = st.columns(4)
    metrics[0].metric("Sources collected", len(items))
    metrics[1].metric("Sources analyzed", min(len(items), 3))
    metrics[2].metric("Research score", score(feedback))
    metrics[3].metric("Status", "Complete")
    if elapsed is not None: st.caption(f"Completed in {elapsed:.1f} seconds.")
    report_tab, resource_tab, critic_tab, details_tab, export_tab = st.tabs(["Research report", "Collected resources", "Critic review", "Research details", "Export"])
    with report_tab:
        st.subheader("Final research report", anchor=False)
        parts = list(re.finditer(r"^##\s+(.+)$", report, re.M))
        if not parts: st.markdown(report or "_No report returned._")
        for index, match in enumerate(parts):
            end = parts[index + 1].start() if index + 1 < len(parts) else len(report)
            with st.expander(match.group(1), expanded=index == 0): st.markdown(report[match.end():end].strip())
    with resource_tab: render_resources(result)
    with critic_tab:
        st.subheader("Critic review", anchor=False)
        with st.container(border=True): st.markdown(feedback or "_No critic feedback returned._")
    with details_tab:
        st.text_area("Reader output", text(result.get("scraped_content")), height=250, key="reader_output")
        with st.expander("Console log"): st.code(log or "No console output captured.", language="text")
    with export_tab:
        payload = f"# Research report: {topic}\n\n{report}\n\n## Critic feedback\n\n{feedback}"
        st.download_button("Download report", payload, "research_report.md", "text/markdown", icon=":material/download:")
        st.download_button("Download raw data", json.dumps(result, indent=2), "research_result.json", "application/json", icon=":material/data_object:")


initialize_state()
clear_clicked = render_sidebar()
topic, submitted = render_header()
if clear_clicked:
    st.session_state.latest_result, st.session_state.latest_log, st.session_state.latest_elapsed = None, "", None
    st.session_state.active_history_index = None
    st.session_state.agent_status = {stage: ("waiting", "Waiting") for stage in STAGES}
    st.rerun()
render_pipeline()
if submitted:
    if not topic:
        st.warning("Enter a research topic before generating a report.", icon=":material/warning:")
    else:
        st.session_state.agent_status = {stage: ("waiting", "Waiting") for stage in STAGES}
        with st.status("Generating research", expanded=True) as status:
            slot = st.empty()
            def progress(stage: str, state: str, message: str) -> None:
                st.session_state.agent_status[stage] = (state, message)
                slot.markdown(f"**{stage}:** {message}")
            try:
                result, log, elapsed = run_pipeline(topic, progress)
            except PipelineStageError as exc:
                status.update(label=f"{exc.stage} failed", state="error", expanded=True)
                st.error(f"{exc.stage}: {exc}", icon=":material/error:")
            except Exception:
                status.update(label="Unexpected pipeline error", state="error", expanded=True)
                st.error("An unexpected error occurred. Review the pipeline log and try again.", icon=":material/error:")
            else:
                status.update(label="Research complete", state="complete", expanded=False)
                st.session_state.latest_result, st.session_state.latest_log, st.session_state.latest_elapsed = result, log, elapsed
                st.session_state.active_history_index = len(st.session_state.run_history)
                st.session_state.run_history.append({"topic": topic, "created_at": datetime.now().strftime("%H:%M"), "elapsed": elapsed, "result": result, "log": log})
                st.rerun()
active = st.session_state.active_history_index
if active is not None and 0 <= active < len(st.session_state.run_history):
    item = st.session_state.run_history[active]
    render_report(item["result"], item["topic"], item["log"], item["elapsed"])
elif st.session_state.latest_result:
    render_report(st.session_state.latest_result, topic, st.session_state.latest_log, st.session_state.latest_elapsed)
else:
    with st.container(border=True):
        st.subheader("Start a research run", anchor=False)
        st.write("Enter a topic above to collect academic sources, draft a report, and receive a critic review.")
