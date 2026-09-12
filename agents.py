"""Groq-backed writer and critic chains for the research pipeline."""
from __future__ import annotations
import os
from dotenv import load_dotenv
load_dotenv()
DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"
class GroqConfigurationError(RuntimeError): pass
def _env_int(name: str, default: int) -> int:
    try: return int(os.getenv(name, str(default)))
    except ValueError: return default
def get_llm():
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key: raise GroqConfigurationError("Groq API key missing. Add GROQ_API_KEY to .env.")
    from langchain_groq import ChatGroq
    return ChatGroq(model=os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL), api_key=key, temperature=float(os.getenv("GROQ_TEMPERATURE", "0.2")), max_tokens=_env_int("GROQ_MAX_TOKENS", 700), timeout=_env_int("GROQ_TIMEOUT", 30), max_retries=0)
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
