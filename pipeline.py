"""Four-stage, failure-aware research pipeline."""
from __future__ import annotations
import random, time
from typing import Any, Callable
from agents import GroqConfigurationError, get_critic_chain, get_writer_chain
from tools import ResearchServiceError, research_search, scrape_url
ProgressCallback = Callable[[str, str, str], None]
class PipelineStageError(RuntimeError):
    def __init__(self, stage: str, message: str): super().__init__(message); self.stage = stage
def _compact(value: str, limit: int) -> str:
    value = str(value).strip(); return value[:limit].rstrip() + ("\n[Truncated]" if len(value) > limit else "")
def _invoke_groq(runnable: Any, payload: dict[str, Any]) -> str:
    for attempt in range(3):
        try: return str(runnable.invoke(payload))
        except Exception as exc:
            text = str(exc).lower()
            if any(x in text for x in ("401", "authentication", "invalid api key")): raise PipelineStageError("Groq", "Groq authentication failed. Check GROQ_API_KEY.") from exc
            if "model" in text and any(x in text for x in ("not found", "404", "decommissioned")): raise PipelineStageError("Groq", "Groq model is unavailable. Check GROQ_MODEL.") from exc
            transient = any(x in text for x in ("429", "rate limit", "timeout", "connection", "remote disconnected", "503", "502"))
            if not transient or attempt == 2:
                if "quota" in text or "tokens per day" in text: raise PipelineStageError("Groq", "Groq quota reached. Wait for reset or change plan.") from exc
                raise PipelineStageError("Groq", "Groq request failed. Please retry shortly.") from exc
            time.sleep(min(8, 2 ** attempt + random.random()))
    raise AssertionError("unreachable")
def _format_sources(sources: list[dict[str, Any]]) -> str:
    records=[]
    for item in sources[:5]:
        records.append(f"Title: {item.get('title')}\nAuthors: {', '.join(item.get('authors', [])[:4]) or 'Unknown authors'}\nYear: {item.get('year') or 'n.d.'}\nSource: {item.get('source')}\nURL: {item.get('url')}\nDOI: {item.get('doi') or 'N/A'}\nAbstract: {_compact(item.get('abstract', ''), 600)}")
    return "\n\n---\n\n".join(records)
def run_research_pipeline(topic: str, progress: ProgressCallback | None = None) -> dict[str, Any]:
    def update(stage: str, state: str, message: str) -> None:
        print(f"[{stage}] {state}: {message}")
        if progress: progress(stage, state, message)
    topic = topic.strip()
    if not topic: raise PipelineStageError("Search Agent", "Enter a research topic before starting.")
    state: dict[str, Any] = {"topic": topic, "sources": [], "warnings": []}
    update("Search Agent", "running", "Searching OpenAlex, arXiv, and optional Tavily.")
    try: sources, warnings = research_search(topic)
    except ResearchServiceError as exc:
        update("Search Agent", "error", str(exc)); raise PipelineStageError("Search Agent", str(exc)) from exc
    state["sources"], state["warnings"], state["search_result"] = sources, warnings, _format_sources(sources)
    update("Search Agent", "complete", f"Found {len(sources)} structured sources.")
    update("Reader Agent", "running", "Ranking metadata and reading at most one supplementary webpage.")
    ranked = sorted(sources, key=lambda s: (s.get("citations") or 0, bool(s.get("abstract"))), reverse=True)
    primary = ranked[0]; web_text = scrape_url(primary["url"]) if not primary.get("abstract") and primary.get("url") else ""
    state["scraped_content"] = _compact(web_text or "Reader used research API abstracts and metadata; no webpage scrape was required.", 1800)
    state["reader_notes"] = _format_sources(ranked[:3]); update("Reader Agent", "complete", "Selected the most relevant structured sources.")
    update("Writer Agent", "running", "Drafting a sourced research report.")
    try: state["report"] = _invoke_groq(get_writer_chain(), {"topic": topic, "research": _compact(state["reader_notes"], 4500)})
    except (GroqConfigurationError, PipelineStageError) as exc:
        update("Writer Agent", "error", str(exc)); raise PipelineStageError("Writer Agent", str(exc)) from exc
    update("Writer Agent", "complete", "Drafted report.")
    update("Critic Agent", "running", "Checking the report for clarity and citation quality.")
    try: state["feedback"] = _invoke_groq(get_critic_chain(), {"report": _compact(state["report"], 7000)})
    except (GroqConfigurationError, PipelineStageError) as exc:
        update("Critic Agent", "error", str(exc)); raise PipelineStageError("Critic Agent", str(exc)) from exc
    update("Critic Agent", "complete", "Completed quality review.")
    return state
