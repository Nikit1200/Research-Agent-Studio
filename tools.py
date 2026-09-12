"""Resilient research retrieval and optional HTML extraction utilities."""
from __future__ import annotations
import html
import logging
import os
from dataclasses import asdict, dataclass
from typing import Any
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()
LOGGER = logging.getLogger(__name__)
REQUEST_TIMEOUT = (5, 15)

def _config_value(name: str, default: str = "") -> str:
    value = os.getenv(name, "").strip()
    if value:
        return value
    try:
        import streamlit as st
        return str(st.secrets.get(name, default)).strip()
    except Exception:
        return default

@dataclass
class ResearchSource:
    title: str
    authors: list[str]
    year: int | None
    abstract: str
    url: str
    doi: str | None = None
    citations: int | None = None
    source: str = "Unknown"
    def to_dict(self) -> dict[str, Any]: return asdict(self)

class ResearchServiceError(RuntimeError):
    """An external research service failed with a safe user-facing message."""

def _compact(value: str, limit: int = 700) -> str:
    value = " ".join(html.unescape(value or "").split())
    return value[:limit].rstrip() + ("…" if len(value) > limit else "")

def _request_json(url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    try:
        response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT, headers={"User-Agent": "Research-Agent-Studio/1.0"})
        response.raise_for_status()
        return response.json()
    except requests.Timeout as exc: raise ResearchServiceError("Research API timed out. Please try again.") from exc
    except requests.ConnectionError as exc: raise ResearchServiceError("Research API could not be reached. Check your connection and try again.") from exc
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "unknown"
        raise ResearchServiceError("Research API rate limit reached. Please try again shortly." if status == 429 else f"Research API returned HTTP {status}.") from exc
    except ValueError as exc: raise ResearchServiceError("Research API returned an invalid response.") from exc

def search_openalex(query: str) -> list[ResearchSource]:
    params: dict[str, Any] = {"search": query, "per-page": 4, "select": "title,authorships,publication_year,abstract_inverted_index,doi,primary_location,cited_by_count"}
    if key := _config_value("OPENALEX_API_KEY"): params["api_key"] = key
    try:
        payload = _request_json("https://api.openalex.org/works", params)
    except ResearchServiceError as exc:
        if not any(f"HTTP {status}" in str(exc) for status in (400, 401, 403)) or "api_key" not in params:
            raise
        params.pop("api_key")
        payload = _request_json("https://api.openalex.org/works", params)
    sources = []
    for work in payload.get("results", []):
        inverted = work.get("abstract_inverted_index") or {}
        words = sorted((position, word) for word, positions in inverted.items() for position in positions)
        location = work.get("primary_location") or {}
        sources.append(ResearchSource(work.get("title") or "Untitled work", [a.get("author", {}).get("display_name", "Unknown") for a in work.get("authorships", [])[:6]], work.get("publication_year"), _compact(" ".join(word for _, word in words)), location.get("landing_page_url") or work.get("doi") or "", work.get("doi"), work.get("cited_by_count"), "OpenAlex"))
    return sources

def search_arxiv(query: str) -> list[ResearchSource]:
    try:
        response = requests.get("https://export.arxiv.org/api/query", params={"search_query": f"all:{query}", "start": 0, "max_results": 4}, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "xml")
    except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as exc: raise ResearchServiceError("arXiv is currently unavailable.") from exc
    sources = []
    for entry in soup.find_all("entry"):
        published = entry.find("published")
        sources.append(ResearchSource(_compact(entry.title.get_text(" ", strip=True), 240), [a.get_text(" ", strip=True) for a in entry.find_all("author")[:6]], int(published.text[:4]) if published and published.text[:4].isdigit() else None, _compact(entry.summary.get_text(" ", strip=True) if entry.summary else ""), entry.id.get_text(strip=True) if entry.id else "", source="arXiv"))
    return sources

def search_tavily(query: str) -> list[ResearchSource]:
    key = _config_value("TAVILY_API_KEY")
    if not key: raise ResearchServiceError("Tavily API key is missing (optional; academic fallbacks are still available).")
    try:
        from tavily import TavilyClient
        payload = TavilyClient(api_key=key).search(query=query, max_results=3, search_depth="basic")
    except Exception as exc:
        message = str(exc).lower()
        if "rate" in message or "429" in message: raise ResearchServiceError("Tavily rate limit reached.") from exc
        if "timeout" in message or "connection" in message or "remote" in message: raise ResearchServiceError("Tavily could not be reached; using academic fallbacks.") from exc
        raise ResearchServiceError("Tavily search failed; using academic fallbacks.") from exc
    return [ResearchSource(r.get("title", "Untitled result"), [], None, _compact(r.get("content", ""), 420), r.get("url", ""), source="Tavily") for r in payload.get("results", [])]

def research_search(query: str) -> tuple[list[dict[str, Any]], list[str]]:
    sources: list[ResearchSource] = []; warnings: list[str] = []
    for provider in (search_openalex, search_arxiv, search_tavily):
        try: sources.extend(provider(query))
        except ResearchServiceError as exc:
            LOGGER.info("%s failed: %s", provider.__name__, exc); warnings.append(str(exc))
    seen: set[str] = set(); unique = []
    for source in sources:
        identity = (source.doi or source.url or source.title).lower()
        if identity and identity not in seen: seen.add(identity); unique.append(source.to_dict())
    if not unique: raise ResearchServiceError("No research sources are available right now. " + " ".join(warnings))
    return unique[:6], warnings

def scrape_url(url: str) -> str:
    """Best-effort HTML extraction; a failed URL never raises into the pipeline."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc: return "Website could not be accessed: invalid URL."
    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT, headers={"User-Agent": "Mozilla/5.0 (compatible; Research-Agent-Studio/1.0)"})
        response.raise_for_status()
        if "html" not in response.headers.get("content-type", "").lower(): return "Website could not be accessed: the URL is not an HTML page."
        response.encoding = response.encoding or response.apparent_encoding or "utf-8"
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]): tag.decompose()
        return _compact(soup.get_text(" ", strip=True), 1800) or "Website could not be accessed: no readable text was returned."
    except requests.Timeout: return "Website could not be accessed: request timed out."
    except requests.ConnectionError: return "Website could not be accessed: connection failed or the site blocks automated requests."
    except requests.HTTPError as exc: return f"Website could not be accessed: HTTP {exc.response.status_code if exc.response else 'error'}."
    except requests.RequestException as exc:
        LOGGER.info("Scrape failed for %s: %s", url, exc); return "Website could not be accessed due to a network error."

def web_search(query: str) -> str:
    sources, warnings = research_search(query)
    text = "\n\n".join(f"Title: {s['title']}\nSource: {s['source']}\nURL: {s['url']}\nAbstract: {s['abstract']}" for s in sources)
    return text + ("\n\nWarnings: " + " | ".join(warnings) if warnings else "")
