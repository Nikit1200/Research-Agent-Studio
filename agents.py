from tools import web_search, scrape_url
from dotenv import load_dotenv
import os

load_dotenv()

_llm = None
_writer_chain = None
_critic_chain = None

# Keep the provider model configurable. The previous default,
# `llama-3.3-70b-versatile`, is no longer available to this Groq account.
DEFAULT_GROQ_MODEL = "qwen/qwen3.8-27b"
# The free Groq on-demand tier for this account permits 1,000 output tokens per
# minute. Keep each model turn deliberately small so a complete four-stage run
# stays below that ceiling.
DEFAULT_GROQ_MAX_TOKENS = 120


def get_llm():
    global _llm
    if _llm is None:
        from langchain_groq import ChatGroq

        _llm = ChatGroq(
            model=os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL),
            temperature=0,
            max_tokens=int(os.getenv("GROQ_MAX_TOKENS", DEFAULT_GROQ_MAX_TOKENS)),
        )
    return _llm


def build_search_agent():
    from langchain.agents import create_agent
    from tools import get_search_tool

    return create_agent(
        model=get_llm(),
        tools=[get_search_tool()],
    )


def build_reader_agent():
    from langchain.agents import create_agent
    from tools import get_scrape_tool

    return create_agent(
        model=get_llm(),
        tools=[get_scrape_tool()],
    )


def get_writer_chain():
    global _writer_chain
    if _writer_chain is None:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser

        writer_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "you are an expert research writer. Write clear, structured and insightful reports.",
            ),
            (
                "human",
                """Write a detailed research report on the topic below.
    
    Topic: {topic}
    Research Gathered: {research}

    Structure the report as:
    -Introduction
    -Key Findings (minimum 3 well- explained points)
    -conclusion
    -Sources (list all URLs found in the research),
    Be detailed , factual and professional.""",
            ),
        ])

        _writer_chain = writer_prompt | get_llm().bind(max_tokens=220) | StrOutputParser()
    return _writer_chain


def get_critic_chain():
    global _critic_chain
    if _critic_chain is None:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser

        critic_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "you are a sharp and constructive research critic. Be honest and specific.",
            ),
            (
                "human",
                """Review the research report below and evaluate it strictly.
    
    Report: {report}

    Respond in this exact format:

    Score: X/10
    Strengths:
    - ...
    - ...

    Areas to Improve:
    - ...
    - ...

    One line verdict:
    ...""",
            ),
        ])

        _critic_chain = critic_prompt | get_llm().bind(max_tokens=100) | StrOutputParser()
    return _critic_chain
