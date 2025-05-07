import math
import os
import requests
import sys
import asyncio

from fastmcp import FastMCP, Context
from loguru import logger
from typing import TypedDict, Optional, Any
from playwright.async_api import async_playwright, Playwright
from inscriptis import get_text


class SearchResult(TypedDict):
    title: str
    url: str
    contains_file: bool


logger.remove()
logger.level("SEARCH", no=15, color="<blue>", icon="🧐")
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level.icon} {level.name: <8}</level> | <level>{message}</level>",
)


BRAVE_API_KEY = os.environ["BRAVE_API_KEY"]


async def fetch_website_data(
    playwright: Playwright, url: str
) -> tuple[str | None, str | None, str | None]:
    """
    Se connecte à une URL donnée avec Playwright et retourne le titre et le contenu HTML de la page.

    Args:
        url: L'URL du site web à visiter.

    Returns:
        Un tuple contenant le titre de la page et son contenu HTML.
        Retourne (None, None) si une erreur survient.
    """
    chromium = playwright.chromium
    browser = await chromium.launch()
    page = await browser.new_page()
    try:
        print(f"Connexion à {url}...")
        # Effectue la requête GET sur l'URL
        # Augmentation du timeout par défaut si nécessaire pour les pages lentes
        await page.goto(url, timeout=60000)  # Timeout de 60 secondes

        # Récupère le titre de la page
        title = await page.title()
        print(f"Le titre de la page est : '{title}'")

        # Récupère le contenu HTML complet de la page
        content = await page.content()
        # print(f"Contenu de la page :\n{content[:500]}...") # Affiche les 500 premiers caractères

        text = get_text(content)
        return title, content, text
    except Exception as e:
        print(f"Une erreur est survenue lors de la connexion à {url}: {e}")
        return None, None, None
    finally:
        # Ferme le navigateur
        await browser.close()


def search(
    question: str,
    country: Optional[str] = None,
    search_lang: Optional[str] = None,
    time_range: Optional[tuple[str, str]] = None,
    pdf: bool = False,
    website: Optional[str] = None,
) -> list[SearchResult]:
    question_uri_format = question.replace(" ", "+").lower()
    time_range_parameter = ""
    if time_range:
        if type(time_range) != tuple:
            raise Exception("time_range format should be a tuple of strings")
        time_range_parameter = f"?freshness={time_range[0]}to{time_range[1]}"

    operators = []
    if pdf:
        operators.append("filetype:pdf")
    if website:
        operators.append(f"site:{website}")
    full_search = question + " AND ".join(operators)

    country_parameter = ""
    if country:
        country_parameter = f"?country={country_parameter.upper()}"

    search_lang_parameter = ""
    if search_lang:
        search_lang_parameter = f"?search_lang={search_lang.lower()}"

    url = f"https://api.search.brave.com/res/v1/web/search?q={full_search}{country_parameter}{search_lang_parameter}"
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",  # requests handles gzip decompression automatically
        "X-Subscription-Token": BRAVE_API_KEY,
    }

    logger.log("SEARCH", f"Searching for {url}")
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    json_response = response.json()
    search_results: list[SearchResult] = []
    if "web" not in json_response:
        return []
    for result in json_response["web"]["results"]:
        search_result: SearchResult = {
            "title": result["title"],
            "url": result["url"],
            "contains_file": True if ".pdf" in result["url"] else False,
        }
        search_results.append(search_result)
    return search_results


# instantiate an MCP server client
mcp = FastMCP("OnlineSearch Server")


@mcp.tool()
def web_search(search_question: str) -> str:
    """Performs a web search and results a list of potentially relevant titles & urls."""
    search_results = search(search_question)
    search_results_string = ""
    for search_result in search_results:
        if search_result["contains_file"]:
            continue
        search_results_string += f"- {search_result['title']}: {search_result['url']}\n"
    return search_results_string


@mcp.tool()
async def read_url(url: str) -> str:
    """Reads the content of a webpage url"""
    async with async_playwright() as playwright:
        title, html_content, text_content = await fetch_website_data(playwright, url)
    return f"{title}\n===\n{text_content}"


@mcp.tool()
def final_answer(answer: Any) -> str:
    """provides the user with your final answer, ends the conversation."""
    return str(answer)


# execute and return the stdio output
if __name__ == "__main__":
    mcp.run(transport="stdio")
