# browser_tool.py - Refactored
import os
import uuid
from datetime import datetime
from langchain.tools import tool
from playwright.sync_api import sync_playwright
from schema import EvidenceItem, SourceType

import logging
logger = logging.getLogger(__name__)

class BrowserManager:
    """Enhanced browser management with better error handling."""
    
    def __init__(self):
        self.screenshot_dir = "evidence_screenshots"
        os.makedirs(self.screenshot_dir, exist_ok=True)
    
    def take_screenshot(self, url: str, full_page: bool = True, wait_for: str = 'networkidle') -> list[EvidenceItem]:
        """Take a screenshot with enhanced options and return EvidenceItem."""
        screenshot_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_{timestamp}_{screenshot_id}.png"
        file_path = os.path.join(self.screenshot_dir, filename)
        
        evidence_items = []
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                context = browser.new_context(viewport={'width': 1280, 'height': 720})
                page = context.new_page()
                
                for attempt in range(3):
                    try:
                        page.goto(url, wait_until=wait_for, timeout=30000)
                        break
                    except Exception as e:
                        if attempt == 2:
                            raise e
                        logger.warning(f"Attempt {attempt + 1} failed for {url}, retrying...")
                
                page.wait_for_load_state('domcontentloaded')
                page.screenshot(path=file_path, full_page=full_page)
                
                title = page.title()
                final_url = page.url
                
                evidence_items.append(EvidenceItem(
                    source_type=SourceType.WEB_PAGE,
                    url=final_url,
                    content=title,
                    timestamp=datetime.now(),
                    author_id="unknown",
                    raw_data={"file_path": file_path, "message": f"Screenshot saved successfully to: {file_path}"}
                ))
                browser.close()
                return evidence_items
                
        except Exception as e:
            logger.error(f"Screenshot failed for {url}: {e}")
            return [EvidenceItem(content=f"Error taking screenshot for {url}: {str(e)}", source_type=SourceType.WEB_PAGE)]

browser_manager = BrowserManager()

@tool
def enhanced_screenshot(url: str, full_page: bool = True) -> list[EvidenceItem]:
    """
    Takes an enhanced screenshot of a webpage with better error handling and metadata extraction.
    Returns a list of EvidenceItem objects.
    """
    return browser_manager.take_screenshot(url, full_page)

@tool
def screenshot_with_interaction(url: str, interactions: str = "") -> list[EvidenceItem]:
    """
    Takes a screenshot after performing basic interactions like clicking or scrolling.
    interactions: comma-separated actions like "scroll_down", "click:.button", "wait:3"
    Returns a list of EvidenceItem objects.
    """
    evidence_items = []
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, wait_until='networkidle', timeout=30000)
            
            if interactions:
                actions = [action.strip() for action in interactions.split(',')]
                for action in actions:
                    if action.startswith('scroll_down'):
                        page.keyboard.press('PageDown')
                    elif action.startswith('click:'):
                        selector = action.split(':', 1)[1]
                        try:
                            page.click(selector, timeout=5000)
                        except:
                            logger.warning(f"Could not click on {selector}")
                    elif action.startswith('wait:'):
                        seconds = int(action.split(':', 1)[1])
                        page.wait_for_timeout(seconds * 1000)
            
            screenshot_id = str(uuid.uuid4())
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"interactive_screenshot_{timestamp}_{screenshot_id}.png"
            file_path = os.path.join(browser_manager.screenshot_dir, filename)
            
            page.screenshot(path=file_path, full_page=True)
            browser.close()
            
            evidence_items.append(EvidenceItem(
                source_type=SourceType.WEB_PAGE,
                url=url,
                content="Interactive screenshot taken",
                timestamp=datetime.now(),
                author_id="unknown",
                raw_data={"file_path": file_path, "message": f"Interactive screenshot saved to: {file_path}"}
            ))
            return evidence_items
            
    except Exception as e:
        return [EvidenceItem(content=f"Interactive screenshot failed for {url}: {str(e)}", source_type=SourceType.WEB_PAGE)]