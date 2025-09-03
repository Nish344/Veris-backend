# playwright_tool.py - Refactored
import asyncio
import os
import re
import aiohttp
import aiofiles
from datetime import datetime
from urllib.parse import urljoin, quote

from langchain_core.tools import tool
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
from schema import EvidenceItem, MediaItem, SourceType

# --- CONFIGURATION ---
X_AUTH_FILE = "x_auth_state.json"
SCREENSHOT_DIR = "screenshots"
PICTURES_DIR = "screenshots"  # Changed to screenshots folder

# Create directories if they don't exist
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

# Configure logging
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

async def download_image(session, image_url: str, filename: str) -> bool:
    """Download an image from a URL and save it to the screenshots directory."""
    try:
        async with session.get(image_url) as response:
            if response.status == 200:
                content = await response.read()
                filepath = os.path.join(PICTURES_DIR, filename)
                async with aiofiles.open(filepath, 'wb') as f:
                    await f.write(content)
                logger.info(f"Downloaded image: {filename}")
                return True
            else:
                logger.warning(f"Failed to download image {image_url}: Status {response.status}")
                return False
    except Exception as e:
        logger.error(f"Error downloading image {image_url}: {e}")
        return False

def generate_timestamp():
    """Generate a timestamp string for file naming."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

@tool
async def search_and_scrape_x(query: str, max_results: int = 10) -> list[EvidenceItem]:
    """
    Searches for a query on X.com and scrapes the top results including ALL images.
    This tool requires a one-time login setup by running the `setup_x_auth.py` script.
    Returns a list of EvidenceItem objects.
    """
    if not os.path.exists(X_AUTH_FILE):
        return [EvidenceItem(content=f"Authentication file '{X_AUTH_FILE}' not found. Please run the `setup_x_auth.py` script first.", source_type=SourceType.TWITTER)]

    logger.info(f"Starting authenticated X search for query: '{query}'")
    timestamp = generate_timestamp()
    
    evidence_items = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(storage_state=X_AUTH_FILE)
        page = await context.new_page()

        try:
            encoded_query = quote(query)
            search_url = f"https://x.com/search?q={encoded_query}&src=typed_query"
            logger.info(f"Navigating to X search page: {search_url}")
            await page.goto(search_url, wait_until="domcontentloaded", timeout=45000)

            logger.info("Waiting for tweets to load...")
            await page.locator('article[data-testid="tweet"]').first.wait_for(timeout=45000)

            # Take initial screenshot
            clean_query = query.replace('#', '').replace('@', '').replace(' ', '_')
            initial_screenshot = f"x_search_initial_{timestamp}_{clean_query}.png"
            await page.screenshot(path=os.path.join(SCREENSHOT_DIR, initial_screenshot))
            logger.info(f"Saved initial search screenshot: {initial_screenshot}")

            # Scroll to load more results
            for scroll_num in range(3):
                await page.mouse.wheel(0, 1500)
                await asyncio.sleep(2)
                if scroll_num == 1:
                    scroll_screenshot = f"x_search_scrolled_{timestamp}_{clean_query}.png"
                    await page.screenshot(path=os.path.join(SCREENSHOT_DIR, scroll_screenshot))
                    logger.info(f"Saved scroll screenshot: {scroll_screenshot}")

            tweets = await page.locator('article[data-testid="tweet"]').all()
            logger.info(f"Found {len(tweets)} tweet locators on the page.")
            
            async with aiohttp.ClientSession() as session:
                image_counter = 0
                global_downloaded_urls = set()
                for i, tweet in enumerate(tweets[:max_results]):
                    try:
                        tweet_url = await tweet.locator('a[href*="/status/"]').first.get_attribute('href')
                        if tweet_url:
                            tweet_url = urljoin("https://x.com", tweet_url)
                        author_name = await tweet.locator('span').filter(has_text=re.compile(r'^@')).first.text_content() or "Unknown"
                        tweet_text = await tweet.locator('div[data-testid="tweetText"]').first.text_content() or ""

                        # Handle images
                        tweet_images = []
                        img_elements = tweet.locator('img')
                        for img_idx in range(await img_elements.count()):
                            img_element = img_elements.nth(img_idx)
                            img_src = await img_element.get_attribute('src')
                            if img_src and img_src not in global_downloaded_urls:
                                global_downloaded_urls.add(img_src)
                                image_counter += 1
                                filename = f"x_image_{timestamp}_{image_counter:03d}.jpg"
                                if await download_image(session, img_src, filename):
                                    tweet_images.append(MediaItem(media_type="IMAGE", url=img_src, local_path=os.path.join(SCREENSHOT_DIR, filename)))

                        evidence_items.append(EvidenceItem(
                            source_type=SourceType.TWITTER,
                            url=tweet_url,
                            content=tweet_text,
                            timestamp=datetime.now(),
                            author_id=author_name,
                            media=tweet_images,
                            raw_data={"scraped_at": datetime.now().isoformat()}
                        ))
                    except Exception as e:
                        logger.warning(f"Could not parse tweet #{i}: {e}")
                        continue

            final_screenshot = f"x_search_final_{timestamp}_{clean_query}.png"
            await page.screenshot(path=os.path.join(SCREENSHOT_DIR, final_screenshot))
            logger.info(f"Saved final search screenshot: {final_screenshot}")

            logger.info(f"Successfully scraped {len(evidence_items)} tweets with {image_counter} total unique images downloaded")
            return evidence_items

        except PlaywrightTimeoutError as e:
            logger.error(f"Timeout while waiting for X.com content: {e}")
            error_screenshot = f"error_x_search_{timestamp}.png"
            await page.screenshot(path=os.path.join(SCREENSHOT_DIR, error_screenshot))
            logger.info(f"Saved error screenshot: {error_screenshot}")
            return [EvidenceItem(content=f"Timeout while waiting for X.com content: {e}", source_type=SourceType.TWITTER)]
        except Exception as e:
            logger.error(f"An unexpected error occurred during X scrape: {e}", exc_info=True)
            error_screenshot = f"error_x_search_{timestamp}.png"
            try:
                await page.screenshot(path=os.path.join(SCREENSHOT_DIR, error_screenshot))
                logger.info(f"Saved error screenshot: {error_screenshot}")
            except:
                pass
            return [EvidenceItem(content=f"An unexpected error occurred during X scrape: {e}", source_type=SourceType.TWITTER)]
        finally:
            await browser.close()

@tool
async def scrape_single_page(url: str) -> list[EvidenceItem]:
    """
    Scrapes the full text content of a single web page and takes a screenshot.
    Returns a list of EvidenceItem objects.
    """
    logger.info(f"Scraping single page: {url}")
    timestamp = generate_timestamp()
    
    evidence_items = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()
        try:
            await page.goto(url, wait_until="networkidle", timeout=45000)
            
            safe_url = re.sub(r'[^\w\-_.]', '_', url.replace('https://', '').replace('http://', ''))
            screenshot_filename = f"page_scrape_{timestamp}_{safe_url[:50]}.png"
            await page.screenshot(path=os.path.join(SCREENSHOT_DIR, screenshot_filename))
            logger.info(f"Saved page screenshot: {screenshot_filename}")
            
            body_text = await page.evaluate("() => document.body.innerText")
            content = body_text.strip() if body_text else "No text content found."

            evidence_items.append(EvidenceItem(
                source_type=SourceType.WEB_PAGE,
                url=url,
                content=content,
                timestamp=datetime.now(),
                author_id="unknown",
                raw_data={"screenshot": screenshot_filename, "scraped_at": datetime.now().isoformat()}
            ))
            return evidence_items
        except Exception as e:
            logger.error(f"Error scraping page {url}: {e}")
            return [EvidenceItem(content=f"Failed to scrape {url}: {e}", source_type=SourceType.WEB_PAGE)]
        finally:
            await browser.close()

# --- Main Execution Block for Testing ---
async def main():
    """Main asynchronous function to test the tools."""
    print("--- 1. TESTING 'search_and_scrape_x' TOOL ---")
    print("NOTE: This test requires the 'x_auth_state.json' file to be present.\n")
    try:
        x_results = await search_and_scrape_x.ainvoke({"query": "#Python", "max_results": 3})
        print("--- X Scraper Result ---")
        for item in x_results:
            print(item.model_dump_json(indent=2))
    except Exception as e:
        print(f"An error occurred during X scraper test: {e}")

    print("\n" + "="*50 + "\n")

    print("--- 2. TESTING 'scrape_single_page' TOOL ---")
    try:
        page_results = await scrape_single_page.ainvoke({"url": "https://example.com/"})
        print("--- Single Page Scraper Result ---")
        for item in page_results:
            print(item.model_dump_json(indent=2))
    except Exception as e:
        print(f"An error occurred during single page scraper test: {e}")

if __name__ == "__main__":
    asyncio.run(main())