class DailyGroqQuotaExceeded(RuntimeError):
    """Raised when Groq's account-wide daily token allowance is exhausted."""


def compact_text(value: str, max_characters: int) -> str:
    """Bound inter-agent context so source pages cannot exhaust the token quota."""
    text = str(value).strip()
    if len(text) <= max_characters:
        return text
    return f"{text[:max_characters].rstrip()}\n[Content truncated to conserve the Groq quota.]"


def invoke_with_rate_limit_retry(runnable, payload: dict, attempts: int = 3):
    """Retry only Groq's transient output-token-per-minute limit."""
    import re
    import time

    for attempt in range(attempts):
        try:
            return runnable.invoke(payload)
        except Exception as exc:
            message = str(exc)
            if "tokens per day" in message.lower():
                raise DailyGroqQuotaExceeded(
                    "Groq's daily token quota is exhausted. Wait for its reset or upgrade the Groq plan."
                ) from exc
            is_rate_limit = "rate_limit_exceeded" in message or "Rate limit reached" in message
            if not is_rate_limit or attempt == attempts - 1:
                raise

            match = re.search(r"try again in\s+([0-9.]+)s", message, flags=re.IGNORECASE)
            delay = float(match.group(1)) + 1 if match else 20
            print(f"Groq output limit reached; retrying in {delay:.1f}s.")
            time.sleep(delay)


def run_research_pipeline(topic: str) -> dict:

    from agents import build_search_agent, build_reader_agent, get_writer_chain, get_critic_chain

    state = {}

    # search agent working
    print("\n"+" ="*50)
    print("step 1 - search agent is working ...")
    print("="*50)

    search_agent = build_search_agent()
    search_result = invoke_with_rate_limit_retry(search_agent, {
        "messages":[("user", f"Find recent and reliable information about: {topic}")]
    })

    state["search_result"] = compact_text(search_result['messages'][-1].content, 1_200)

    print("\n search result", state["search_result"])


    # step-2 - reader agent 

    print("\n"+" ="*50)
    print("step 2 - Reader agent is scraping top resources ...")
    print("="*50)

    reader_agent = build_reader_agent()
    reader_result = invoke_with_rate_limit_retry(reader_agent, {
        "messages":[("user",
                     f"Based on the following search results about '{topic}',"
                     f"choose one primary URL and scrape it for concise supporting facts.\n\n"
                     f"SEARCH RESULTS:\n{state['search_result']}"
        )]
    })

    state['scraped_content'] = compact_text(reader_result['messages'][-1].content, 1_800)

    print("\n scraped content:\n", state['scraped_content'])

    # step 3 - writer chain

    print("\n"+" ="*50)
    print("step 3 - Writer is drafting the report ...")
    print("="*50)

    research_combined = (
        f"SEARCH RESULTS:\n{state['search_result']}\n\n"
        f"DETAILED SCRAPED CONTENT:\n{state['scraped_content']}"
    )

    state["report"] = invoke_with_rate_limit_retry(get_writer_chain(), {
        "topic": topic,
        "research": research_combined,
    })

    print("\n final Report\n", state["report"])

    # step 4 - critic report

    print("\n"+" ="*50)
    print("step 4 - critic is reviewing the report ...")
    print("="*50)


    state["feedback"] = invoke_with_rate_limit_retry(get_critic_chain(), {
        "report": state["report"],
    })

    print("\n critic report \n", state["feedback"])

    return state


if __name__ == "__main__":
    topic = input("\n Enter a research topic:")
    run_research_pipeline(topic)
