# search_tool.py (v4 - Corrected Brave Search selectors)
import asyncio
import logging
from typing import List, Dict, Any
from urllib.parse import quote_plus, urljoin
from langchain_core.tools import tool
from playwright.async_api import async_playwright, Error

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Functions ---

async def _scrape_brave_search_links(query: str, max_results: int = 10) -> List[Dict[str, str]]:
    """
    Uses Playwright to perform a search on Brave Search and scrape the results.
    This version uses updated selectors based on direct HTML analysis.
    """
    results = []
    encoded_query = quote_plus(query)
    search_url = f"https://search.brave.com/search?q={encoded_query}"

    async with async_playwright() as p:
        page = None
        browser = None
        try:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            )
            page = await context.new_page()
            
            logging.info(f"Navigating to {search_url}")
            await page.goto(search_url, wait_until="domcontentloaded", timeout=25000)

            # FIX: Updated selectors to match the HTML provided in the debug output.
            # The main container for each result is a div with class "snippet".
            result_selector = "div.snippet"
            logging.info(f"Waiting for selector: '{result_selector}'")
            await page.wait_for_selector(result_selector, timeout=20000)
            
            result_nodes = await page.query_selector_all(result_selector)
            logging.info(f"Found {len(result_nodes)} search results on the page.")

            for node in result_nodes[:max_results]:
                # Using selectors derived from the debug HTML.
                link_node = await node.query_selector("a.svelte-7ipt5e")
                title_node = await node.query_selector("div.title")
                snippet_node = await node.query_selector("div.snippet-description")

                if link_node and title_node and snippet_node:
                    title = await title_node.inner_text()
                    link = await link_node.get_attribute("href")
                    snippet = await snippet_node.inner_text()
                    
                    # FIX: Links can be relative, so we join them with the base URL to make them absolute.
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

def _format_results(results: List[Dict[str, str]]) -> str:
    """Formats the scraped results into a clean, readable string."""
    if not results:
        return "Search completed, but no results were found."
    if "error" in results[0]:
        return f"Search failed. Details: {results[0].get('error', 'Unknown error')}"
        
    formatted_string = ""
    for result in results:
        formatted_string += f"Title: {result.get('title')}\n"
        formatted_string += f"URL: {result.get('url')}\n"
        formatted_string += f"Snippet: {result.get('snippet')}\n\n"
    return formatted_string

# --- Main Tool Definition ---

@tool
def browser_search(query: str, max_results: int = 10) -> str:
    """
    Performs a web search using a headless browser to scrape Brave Search results.
    This is a reliable method for gathering search links.
    
    Args:
        query: The search query.
        max_results: The maximum number of results to return.
    """
    logging.info(f"Starting browser search for query: '{query}'")
    try:
        scraped_data = asyncio.run(_scrape_brave_search_links(query, max_results))
        return _format_results(scraped_data)
    except Exception as e:
        logging.error(f"Failed to run browser_search: {e}")
        return f"An error occurred while trying to run the browser search: {e}"

# --- Testing Block ---
if __name__ == "__main__":
    print("--- Running Test 1: Simple Search ---")
    results1 = browser_search.invoke({"query": "what is langgraph"})
    print(results1)
    print("-" * 50)

    print("\n--- Running Test 2: Search for a specific person ---")
    results2 = browser_search.invoke({"query": "Narendra Modi"})
    print(results2)
    print("-" * 50)
