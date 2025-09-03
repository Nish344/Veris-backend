import asyncio
import json
import logging
import os
import re
import aiohttp
import aiofiles
from datetime import datetime
from urllib.parse import urljoin, quote

from langchain_core.tools import tool
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError

# --- CONFIGURATION ---
X_AUTH_FILE = "x_auth_state.json"
SCREENSHOT_DIR = "screenshots"
# Save everything in screenshots folder as requested
PICTURES_DIR = "screenshots"  # Changed to screenshots folder

# Create directories if they don't exist
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

# Configure logging for better debugging
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


# --- Tools ---

@tool
async def search_and_scrape_x(query: str, max_results: int = 10) -> str:
    """
    Searches for a query on X.com (formerly Twitter) and scrapes the top results including ALL images.
    This tool requires a one-time login setup by running the `setup_x_auth.py` script.
    
    Args:
        query (str): The search query (e.g., a hashtag '#VerisProject' or a mention '@VerisTruth').
        max_results (int): The maximum number of tweets to scrape.
    """
    if not os.path.exists(X_AUTH_FILE):
        return json.dumps([{
            "error": f"Authentication file '{X_AUTH_FILE}' not found. "
                     f"Please run the `setup_x_auth.py` script first."
        }])

    logger.info(f"Starting authenticated X search for query: '{query}'")
    timestamp = generate_timestamp()
    
    async with async_playwright() as p:
        # For debugging, you might use headless=False, but for production, headless=True is better.
        browser = await p.chromium.launch(headless=False) 
        context = await browser.new_context(storage_state=X_AUTH_FILE)
        page = await context.new_page()

        try:
            # URL encode the query properly to handle special characters like '#'
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
            for scroll_num in range(3): # Increased scrolls for more results
                await page.mouse.wheel(0, 1500)
                await asyncio.sleep(2)
                
                # Take screenshot during scrolling
                if scroll_num == 1:  # Take screenshot after first scroll
                    scroll_screenshot = f"x_search_scrolled_{timestamp}_{clean_query}.png"
                    await page.screenshot(path=os.path.join(SCREENSHOT_DIR, scroll_screenshot))
                    logger.info(f"Saved scroll screenshot: {scroll_screenshot}")

            # Get tweets for processing
            tweets = await page.locator('article[data-testid="tweet"]').all()
            logger.info(f"Found {len(tweets)} tweet locators on the page.")
            
            scraped_data = []
            image_counter = 0
            global_downloaded_urls = set()  # Global tracker to prevent duplicates across all images
            
            # Create aiohttp session for downloading images
            async with aiohttp.ClientSession() as session:
                
                # DOWNLOAD ALL IMAGES ON THE PAGE FIRST
                logger.info("Starting to download ALL images visible on the page...")
                
                # Get ALL images on the entire page
                all_page_images = page.locator('img')
                total_images = await all_page_images.count()
                logger.info(f"Found {total_images} total images on the page")
                
                # Download all images first
                for img_idx in range(total_images):
                    try:
                        img_element = all_page_images.nth(img_idx)
                        img_src = await img_element.get_attribute('src')
                        img_alt = await img_element.get_attribute('alt') or f"Image {img_idx}"
                        
                        # Skip if we've already processed this image URL (prevent duplicates)
                        if img_src in global_downloaded_urls:
                            continue
                        
                        # Download ALL images from Twitter/X domains EXCEPT small profile pics
                        if (img_src and 
                            ('pbs.twimg.com' in img_src or 'twimg.com' in img_src or 'abs.twimg.com' in img_src)):
                            
                            # Check image dimensions to filter out 48x48 profile pics
                            try:
                                # Get the bounding box to check dimensions
                                bbox = await img_element.bounding_box()
                                if bbox:
                                    width = bbox['width']
                                    height = bbox['height']
                                    
                                    # Skip 48x48 profile pictures and other tiny images
                                    if width <= 48 and height <= 48:
                                        logger.info(f"Skipping small profile image ({width}x{height}): {img_src[:50]}...")
                                        continue
                                    
                                    logger.info(f"Processing image ({width}x{height}): {img_src[:50]}...")
                                
                            except Exception as size_error:
                                # If we can't get dimensions, check URL patterns for profile pics
                                if ('profile_images' in img_src or 
                                    '/profile_images/' in img_src or
                                    'profile_banners' in img_src or
                                    'avatars' in img_src or
                                    '48x48' in img_src):
                                    logger.info(f"Skipping likely profile image by URL pattern: {img_src[:50]}...")
                                    continue
                            
                            # Mark this URL as processed to prevent duplicates
                            global_downloaded_urls.add(img_src)
                            
                            # Generate filename for the image
                            image_counter += 1
                            # Determine extension from URL or default to jpg
                            if 'format=png' in img_src or '.png' in img_src:
                                img_extension = 'png'
                            elif 'format=webp' in img_src or '.webp' in img_src:
                                img_extension = 'webp'
                            elif '.gif' in img_src:
                                img_extension = 'gif'
                            else:
                                img_extension = 'jpg'
                            
                            # Create descriptive filename
                            img_filename = f"x_image_{timestamp}_{image_counter:03d}.{img_extension}"
                            
                            # Download the image to screenshots folder
                            success = await download_image(session, img_src, img_filename)
                            if success:
                                logger.info(f"Downloaded image #{image_counter}: {img_filename}")
                            else:
                                # If download failed, remove from global tracker so it can be retried
                                global_downloaded_urls.discard(img_src)
                                        
                    except Exception as img_error:
                        logger.warning(f"Could not process image {img_idx}: {img_error}")
                
                logger.info(f"Completed downloading {image_counter} unique images from the page")
                
                # Now process individual tweets for text content and metadata
                for i, tweet_locator in enumerate(tweets[:max_results]):
                    try:
                        # Get tweet text
                        # Get tweet text
                        tweet_text_element = tweet_locator.locator('div[data-testid="tweetText"]')
                        tweet_text = await tweet_text_element.first.inner_text() if await tweet_text_element.count() > 0 else "No text content"

                        # Finding the permanent link to the tweet which contains author and status ID
                        timestamp_link = tweet_locator.locator('a[href*="/status/"]').first

                        tweet_url = ""
                        author_name = "Unknown Author"
                        if await timestamp_link.count() > 0:
                            href = await timestamp_link.get_attribute('href')
                            tweet_url = urljoin("https://x.com", href)
                            # Extract author from the URL, e.g., /author/status/123 -> author
                            match = re.search(r'/(.*?)/status/', href)
                            if match:
                                author_name = f"@{match.group(1)}"

                        # Find which downloaded images belong to this specific tweet
                        tweet_images = []
                        tweet_img_elements = tweet_locator.locator('img')
                        tweet_img_count = await tweet_img_elements.count()
                        
                        for img_idx in range(tweet_img_count):
                            try:
                                img_element = tweet_img_elements.nth(img_idx)
                                img_src = await img_element.get_attribute('src')
                                
                                if (img_src and img_src in global_downloaded_urls):
                                    # Find the filename that corresponds to this image
                                    for counter in range(1, image_counter + 1):
                                        # Check all possible extensions
                                        for ext in ['jpg', 'png', 'webp', 'gif']:
                                            potential_filename = f"x_image_{timestamp}_{counter:03d}.{ext}"
                                            if os.path.exists(os.path.join(SCREENSHOT_DIR, potential_filename)):
                                                # Verify this is the right image by checking if URLs would match
                                                tweet_images.append({
                                                    "filename": potential_filename,
                                                    "original_url": img_src,
                                                    "local_path": os.path.join(SCREENSHOT_DIR, potential_filename)
                                                })
                                                break
                                            
                            except Exception as img_error:
                                logger.warning(f"Could not link image {img_idx} in tweet #{i}: {img_error}")

                        tweet_data = {
                            "url": tweet_url,
                            "author": author_name,
                            "content": tweet_text.strip(),
                            "images": tweet_images,
                            "source": "x.com",
                            "scraped_at": datetime.now().isoformat()
                        }
                        
                        scraped_data.append(tweet_data)
                        
                    except Exception as e:
                        logger.warning(f"Could not parse tweet #{i}: {e}")
                        continue

            # Take final screenshot showing results
            final_screenshot = f"x_search_final_{timestamp}_{clean_query}.png"
            await page.screenshot(path=os.path.join(SCREENSHOT_DIR, final_screenshot))
            logger.info(f"Saved final search screenshot: {final_screenshot}")

            logger.info(f"Successfully scraped {len(scraped_data)} tweets with {image_counter} total unique images downloaded")
            return json.dumps(scraped_data, indent=2)

        except PlaywrightTimeoutError as e:
            error_message = f"Timeout while waiting for X.com content: {e}"
            logger.error(error_message)
            error_screenshot = f"error_x_search_{timestamp}.png"
            await page.screenshot(path=os.path.join(SCREENSHOT_DIR, error_screenshot))
            logger.info(f"Saved error screenshot: {error_screenshot}")
            return json.dumps([{"error": error_message}])
        except Exception as e:
            error_message = f"An unexpected error occurred during X scrape: {e}"
            logger.error(error_message, exc_info=True)
            error_screenshot = f"error_x_search_{timestamp}.png"
            try:
                await page.screenshot(path=os.path.join(SCREENSHOT_DIR, error_screenshot))
                logger.info(f"Saved error screenshot: {error_screenshot}")
            except:
                pass
            return json.dumps([{"error": error_message}])
        finally:
            await browser.close()


@tool
async def scrape_single_page(url: str) -> str:
    """
    Scrapes the full text content of a single web page and takes a screenshot.
    
    Args:
        url (str): The URL to scrape.
    """
    logger.info(f"Scraping single page: {url}")
    timestamp = generate_timestamp()
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()
        try:
            await page.goto(url, wait_until="networkidle", timeout=45000)
            
            # Take screenshot of the page
            safe_url = re.sub(r'[^\w\-_.]', '_', url.replace('https://', '').replace('http://', ''))
            screenshot_filename = f"page_scrape_{timestamp}_{safe_url[:50]}.png"
            await page.screenshot(path=os.path.join(SCREENSHOT_DIR, screenshot_filename))
            logger.info(f"Saved page screenshot: {screenshot_filename}")
            
            body_text = await page.evaluate("() => document.body.innerText")
            
            result = {
                "url": url,
                "content": body_text.strip() if body_text else "No text content found.",
                "screenshot": screenshot_filename,
                "scraped_at": datetime.now().isoformat()
            }
            return json.dumps(result, indent=2)
        except Exception as e:
            logger.error(f"Error scraping page {url}: {e}")
            return json.dumps({"error": f"Failed to scrape {url}: {e}"})
        finally:
            await browser.close()


# --- Main Execution Block for Testing ---
async def main():
    """Main asynchronous function to test the tools."""
    
    print("--- 1. TESTING 'search_and_scrape_x' TOOL ---")
    print("NOTE: This test requires the 'x_auth_state.json' file to be present.\n")
    try:
        # Using a common hashtag for a reliable test that is properly encoded
        x_results_str = await search_and_scrape_x.ainvoke({"query": "#Python", "max_results": 3})
        print("--- X Scraper Result ---")
        # Pretty-print the JSON output
        print(json.dumps(json.loads(x_results_str), indent=2))
    except Exception as e:
        print(f"An error occurred during X scraper test: {e}")

    print("\n" + "="*50 + "\n")

    print("--- 2. TESTING 'scrape_single_page' TOOL ---")
    try:
        # Using a simple and reliable website for scraping test
        page_results_str = await scrape_single_page.ainvoke({"url": "https://example.com/"})
        print("--- Single Page Scraper Result ---")
        # Pretty-print the JSON output
        print(json.dumps(json.loads(page_results_str), indent=2))
    except Exception as e:
        print(f"An error occurred during single page scraper test: {e}")


if __name__ == "__main__":
    # This allows the async main function to be run from the command line
    asyncio.run(main())
