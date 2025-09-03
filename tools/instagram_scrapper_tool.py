import aiohttp
import asyncio
import json
import logging
import os
import re
from urllib.parse import urljoin
from datetime import datetime

from langchain_core.tools import tool
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError

# --- CONFIGURATION ---
IG_AUTH_FILE = "ig_auth_state.json"
SCREENSHOT_DIR = "screenshots"
PICTURES_DIR = "pictures"
DEBUG_DIR = "debug"
os.makedirs(SCREENSHOT_DIR, exist_ok=True)
os.makedirs(PICTURES_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# --- Instagram Tools ---

@tool
async def search_and_scrape_instagram(query: str, max_results: int = 5) -> str:
    """
    Searches for a hashtag or profile on Instagram and scrapes posts.
    Requires a valid ig_auth_state.json file.
    """
    if not os.path.exists(IG_AUTH_FILE):
        return json.dumps([{"error": f"Authentication file '{IG_AUTH_FILE}' not found."}])

    logger.info(f"Starting authenticated Instagram search for: {query}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_query = re.sub(r'[^a-zA-Z0-9]', '', query)
    debug_html_path = os.path.join(DEBUG_DIR, f"instagram_{safe_query}_{timestamp}.html")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(storage_state=IG_AUTH_FILE)
        page = await context.new_page()

        try:
            if query.startswith("#"):
                url = f"https://www.instagram.com/explore/tags/{query.strip('#')}/"
            else:
                url = f"https://www.instagram.com/{query.strip('@')}/"

            logger.info(f"Navigating to {url}")
            await page.goto(url, wait_until="domcontentloaded", timeout=90000)

            main_content_selector = "main[role='main']"
            await page.wait_for_selector(main_content_selector, timeout=30000)
            logger.info("Main content area is visible.")

            try:
                not_now_button = page.get_by_role("button", name="Not now")
                if await not_now_button.is_visible(timeout=5000):
                    logger.info("Dismissing 'Not now' login prompt.")
                    await not_now_button.click()
                    await asyncio.sleep(2)
            except PlaywrightTimeoutError:
                logger.info("No 'Not now' login prompt found.")


            for i in range(3):
                logger.info(f"Scrolling down... (Attempt {i+1})")
                await page.mouse.wheel(0, 2000)
                await asyncio.sleep(3)

            post_link_selector = "a[href*='/p/']"
            
            await page.locator(post_link_selector).first.wait_for(timeout=30000)
            
            with open(debug_html_path, "w", encoding="utf-8") as f:
                f.write(await page.content())
            logger.info(f"Saved debug HTML to {debug_html_path}")

            post_links = await page.locator(post_link_selector).all()
            if not post_links:
                logger.warning("No post links found.")
                return json.dumps([{"error": "No posts found."}])

            logger.info(f"Found {len(post_links)} post links on the page.")
            
            scraped_data = []
            async with aiohttp.ClientSession() as session:
                for i, link_locator in enumerate(post_links[:max_results]):
                    post_url = ""
                    try:
                        href = await link_locator.get_attribute('href')
                        if not href: continue
                        post_url = urljoin("https://www.instagram.com", href)

                        await link_locator.click()

                        dialog_selector = "div[role='dialog']"
                        await page.locator(dialog_selector).first.wait_for(timeout=30000)
                        dialog_locator = page.locator(dialog_selector).first
                        
                        caption_el = await dialog_locator.locator("h1").first.element_handle(timeout=7000)
                        caption = (await caption_el.inner_text() if caption_el else "No caption").strip()

                        author_href_el = dialog_locator.locator("header a[role='link']").first
                        author_href = await author_href_el.get_attribute("href")
                        author = author_href.strip('/') if author_href else "Unknown"

                        image_url = ""
                        try:
                            image_el_src = await dialog_locator.locator("article img[srcset]").first.get_attribute("src", timeout=7000)
                            image_url = image_el_src if image_el_src else ""
                        except PlaywrightTimeoutError:
                            video_el_poster = await dialog_locator.locator("article video[poster]").first.get_attribute("poster", timeout=3000)
                            image_url = video_el_poster if video_el_poster else ""

                        post_id = post_url.strip('/').split('/')[-1]
                        safe_author = re.sub(r'[^a-zA-Z0-9]', '', author)
                        
                        screenshot_filename = f"ig_{safe_author}_{post_id}.png"
                        screenshot_path = os.path.join(SCREENSHOT_DIR, screenshot_filename)
                        await dialog_locator.screenshot(path=screenshot_path)

                        downloaded_image_path = ""
                        if image_url:
                            image_filename = f"ig_{safe_author}_{post_id}.jpg"
                            image_filepath = os.path.join(PICTURES_DIR, image_filename)
                            async with session.get(image_url) as resp:
                                if resp.status == 200:
                                    with open(image_filepath, 'wb') as f:
                                        f.write(await resp.read())
                                    downloaded_image_path = image_filepath

                        scraped_data.append({
                            "url": post_url, "author": author, "content": caption,
                            "media_urls": [image_url] if image_url else [],
                            "screenshot_path": screenshot_path,
                            "downloaded_image_path": downloaded_image_path,
                            "source": "instagram.com"
                        })
                    
                    except Exception as e:
                        logger.warning(f"Failed to scrape post #{i+1} at {post_url}: {e}")
                    
                    finally:
                        close_button = page.locator("svg[aria-label='Close']")
                        if await close_button.count() > 0:
                            await close_button.click()
                        await asyncio.sleep(1)

            return json.dumps(scraped_data, indent=2)

        except Exception as e:
            error_message = f"An unexpected error occurred: {e}"
            logger.error(error_message, exc_info=True)
            try:
                with open(debug_html_path, "w", encoding="utf-8") as f:
                    f.write(await page.content())
                logger.info(f"Saved debug HTML on error to {debug_html_path}")
            except Exception:
                pass
            return json.dumps([{"error": error_message}])
        finally:
            if 'browser' in locals() and browser.is_connected():
                await browser.close()

# --- CLI Test Block ---
async def main():
    print("\n--- Testing Instagram scraper for hashtag ---")
    results_str = await search_and_scrape_instagram.ainvoke({"query": "#VerisDataLeak", "max_results": 2})
    print("\n--- INSTAGRAM SCRAPER RESULT ---")
    try:
        print(json.dumps(json.loads(results_str), indent=2))
    except (json.JSONDecodeError, TypeError) as e:
        print(f"Could not parse result: {e}\nRaw output: {results_str}")

if __name__ == "__main__":
    asyncio.run(main())

