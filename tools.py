import requests
from bs4 import BeautifulSoup
from tavily import TavilyClient
import os
from dotenv import load_dotenv
load_dotenv()
from rich import print

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


def web_search(query: str) -> str:
    """Search the web for recent and reliable information on a topic. Returns Titles, URLs and snippets"""

    # Keep tool output compact: every character is later sent back to the LLM.
    results = tavily.search(query=query, max_results=3)

    out = []
    for r in results["results"]:
        out.append(
            f"Title: {r['title']}\nURL: {r['url']}\nSnippet: {r['content'][:160]}"
        )

    return "\n----\n".join(out)


def scrape_url(url: str) -> str:
    """Scrape and return clean text content from a given URL for deeper reading."""
    try:
        resp = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        return soup.get_text(separator=" ", strip=True)[:1200]
    except Exception as e:
        return f"Could not scrape URL: {str(e)}"


def get_search_tool():
    from langchain.tools import tool

    return tool(web_search)


def get_scrape_tool():
    from langchain.tools import tool

    return tool(scrape_url)


if __name__ == "__main__":
    print(
        scrape_url.invoke(
            "https://www.thehindu.com/news/national/centre-proposes-bill-to-strengthen-anti-cheating-law-in-bid-to-curb-exam-malpractices/article71265665.ece"
        )
    )
