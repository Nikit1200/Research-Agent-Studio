"""Gemini-backed writer and critic chains for the research pipeline."""
from __future__ import annotations
from functools import lru_cache
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).with_name(".env"))
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"

def _config_value(name: str, default: str = "") -> str:
    value = os.getenv(name, "").strip()
    if value:
        return value
    try:
        import streamlit as st
        return str(st.secrets.get(name, default)).strip()
    except Exception:
        return default

class GeminiConfigurationError(RuntimeError): pass
def _env_int(name: str, default: int) -> int:
    try: return int(os.getenv(name, str(default)))
    except ValueError: return default
def _env_float(name: str, default: float) -> float:
    try: return float(os.getenv(name, str(default)))
    except ValueError: return default
@lru_cache(maxsize=1)
def get_llm():
    key = _config_value("GOOGLE_API_KEY")
    if not key: raise GeminiConfigurationError("Gemini API key is missing. Add GOOGLE_API_KEY to Streamlit Cloud Secrets, then reboot the app.")
    model = _config_value("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
    if not model: raise GeminiConfigurationError("Gemini model is missing. Add GEMINI_MODEL to .env.")
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(model=model, api_key=key, temperature=_env_float("GEMINI_TEMPERATURE", 0.0), max_tokens=_env_int("GEMINI_MAX_OUTPUT_TOKENS", 1200), request_timeout=_env_int("GEMINI_TIMEOUT", 30), retries=0)
def get_writer_chain():
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    prompt = ChatPromptTemplate.from_messages([("system", "You are a careful academic research writer. Use only supplied sources; never invent citations."), ("human", """Write a concise report about {topic} from the source evidence below.
Required structure exactly:
# Research Report
## 1. Introduction
## 2. Background
## 3. Key Findings
### Finding 1
### Finding 2
### Finding 3
## 4. Advantages / Opportunities
## 5. Challenges / Limitations
## 6. Future Scope
## 7. Conclusion
## 8. References
In References list only titles and URLs/DOIs in the supplied evidence. Qualify uncertainty.
SOURCE EVIDENCE:\n{research}""")])
    return prompt | get_llm() | StrOutputParser()
def get_critic_chain():
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    prompt = ChatPromptTemplate.from_messages([("system", "You are a strict research editor. Evaluate only the supplied report and sources."), ("human", """Review this report for factual consistency, source quality, completeness, clarity, logical structure, unsupported claims, and citation quality.
Return exactly:
Score: X/10
Strengths:
- ...
Areas to Improve:
- ...
Citation Issues:
- ...
Verdict: ...
REPORT:\n{report}""")])
    return prompt | get_llm() | StrOutputParser()
