# search_tool.py - Refactored
import asyncio
import logging
from typing import List
from urllib.parse import quote_plus, urljoin
from langchain_core.tools import tool
from playwright.async_api import async_playwright, Error
from schema import EvidenceItem, SourceType

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def _scrape_brave_search_links(query: str, max_results: int = 10) -> List[dict]:
    results = []
    encoded_query = quote_plus(query)
    search_url = f"https://search.brave.com/search?q={encoded_query}"

    async with async_playwright() as p:
        page = None
        browser = None
        try:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')
            page = await context.new_page()
            
            logging.info(f"Navigating to {search_url}")
            await page.goto(search_url, wait_until="domcontentloaded", timeout=25000)

            result_selector = "div.snippet"
            logging.info(f"Waiting for selector: '{result_selector}'")
            await page.wait_for_selector(result_selector, timeout=20000)
            
            result_nodes = await page.query_selector_all(result_selector)
            logging.info(f"Found {len(result_nodes)} search results on the page.")

            for node in result_nodes[:max_results]:
                link_node = await node.query_selector("a.svelte-7ipt5e")
                title_node = await node.query_selector("div.title")
                snippet_node = await node.query_selector("div.snippet-description")

                if link_node and title_node and snippet_node:
                    title = await title_node.inner_text()
                    link = await link_node.get_attribute("href")
                    snippet = await snippet_node.inner_text()
                    if link:
                        link = urljoin(search_url, link)

                    results.append({"title": title.strip(), "url": link, "snippet": snippet.strip()})
        except Error as e:
            logging.error(f"A Playwright error occurred during search: {e}")
            if page:
                try:
                    content = await page.content()
                    with open("debug_page.html", "w", encoding="utf-8") as f:
                        f.write(content)
                    logging.info("Saved failing page HTML to debug_page.html for analysis.")
                except Exception as save_e:
                    logging.error(f"Could not save page content for debugging: {save_e}")
            return [{"error": f"Failed to scrape search results due to a browser error: {e}"}]
        except Exception as e:
            logging.error(f"An unexpected error occurred during search: {e}")
            return [{"error": f"An unexpected error occurred: {e}"}]
        finally:
            if browser:
                await browser.close()
    return results

@tool
def browser_search(query: str, max_results: int = 10) -> List[EvidenceItem]:
    """
    Performs a web search using a headless browser to scrape Brave Search results.
    Returns a list of EvidenceItem objects.
    """
    logging.info(f"Starting browser search for query: '{query}'")
    try:
        scraped_data = asyncio.run(_scrape_brave_search_links(query, max_results))
        evidence_items = []
        for result in scraped_data:
            if "error" in result:
                evidence_items.append(EvidenceItem(
                    source_type=SourceType.WEB_SEARCH,
                    content=result["error"],
                    timestamp=datetime.now()
                ))
            else:
                evidence_items.append(EvidenceItem(
                    source_type=SourceType.WEB_SEARCH,
                    url=result["url"],
                    content=f"{result['title']}: {result['snippet']}",
                    timestamp=datetime.now(),
                    raw_data={"title": result["title"]}
                ))
        return evidence_items
    except Exception as e:
        logging.error(f"Failed to run browser_search: {e}")
        return [EvidenceItem(
            source_type=SourceType.WEB_SEARCH,
            content=f"An error occurred while trying to run the browser search: {e}",
            timestamp=datetime.now()
        )]

if __name__ == "__main__":
    print("--- Running Test 1: Simple Search ---")
    results1 = browser_search.invoke({"query": "what is langgraph"})
    for item in results1:
        print(item.model_dump_json(indent=2))
    print("-" * 50)

    print("\n--- Running Test 2: Search for a specific person ---")
    results2 = browser_search.invoke({"query": "Narendra Modi"})
    for item in results2:
        print(item.model_dump_json(indent=2))
    print("-" * 50)