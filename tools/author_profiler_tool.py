# author_profiler_tool.py - Corrected
import aiohttp
import asyncio
import os
from datetime import datetime
from urllib.parse import urljoin
from typing import Optional, List
import logging

from langchain_core.tools import tool
from playwright.async_api import async_playwright
from pydantic import BaseModel, Field, HttpUrl
from schema import EvidenceItem, MediaItem, SourceType

# --- MODELS ---
class PostMedia(BaseModel):
    media_type: str = "IMAGE"  # default uppercase
    url: HttpUrl

class RecentPost(BaseModel):
    post_id: str
    post_url: HttpUrl
    text: str
    created_at: datetime
    like_count: int
    reply_count: int
    retweet_count: int
    view_count: int
    media: List[MediaItem] = Field(default_factory=list)   # use schema.MediaItem here
    mentioned_accounts: List[str] = Field(default_factory=list)
    hashtags: List[str] = Field(default_factory=list)

class AuthorProfile(BaseModel):
    author_id: str
    username: str
    display_name: str
    bio: Optional[str]
    followers_count: int
    following_count: int
    posts_count: str
    account_created_at: datetime
    is_verified: bool
    profile_url: HttpUrl
    location: Optional[str]
    recent_posts: List[RecentPost] = Field(default_factory=list)

# --- CONFIG ---
X_AUTH_FILE = "x_auth_state.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def parse_engagement_string(text: Optional[str]) -> int:
    """Accepts strings like '1,234 Likes', '10.5K', '2M Views', '12 replies', or None. Returns int (0 on failure)."""
    try:
        if not text:
            return 0
        token = str(text).strip().split()[0].lower().replace(',', '')
        if token.endswith('k'):
            return int(float(token[:-1]) * 1_000)
        if token.endswith('m'):
            return int(float(token[:-1]) * 1_000_000)
        return int(float(token))
    except Exception:
        return 0

async def safe_text(locator, default: Optional[str] = None) -> Optional[str]:
    try:
        if await locator.count() == 0:
            return default
        val = await locator.first.text_content()
        return val.strip() if val else default
    except Exception:
        return default

async def safe_attr(locator, attr: str, default: Optional[str] = None) -> Optional[str]:
    try:
        if await locator.count() == 0:
            return default
        val = await locator.first.get_attribute(attr)
        return val if val is not None else default
    except Exception:
        return default

@tool("get_author_profile")
async def get_author_profile(username: str, max_posts: int = 4) -> list[EvidenceItem]:
    """
    Scrape an X profile and recent posts.
    Returns a list of EvidenceItem objects with profile details.
    Requires a valid Playwright storage state.
    """
    if not os.path.exists(X_AUTH_FILE):
        return [EvidenceItem(content=f"Authentication file '{X_AUTH_FILE}' not found.", source_type=SourceType.TWITTER)]

    logger.info(f"Scraping profile: @{username}")
    profile_url = f"https://x.com/{username}"

    evidence_items = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(storage_state=X_AUTH_FILE)
        page = await context.new_page()

        try:
            await page.goto(profile_url, wait_until="domcontentloaded", timeout=120_000)
            await page.wait_for_selector('div[data-testid="primaryColumn"]', timeout=30_000)

            header = page.locator('div[data-testid="primaryColumn"]')

            display_name = await safe_text(header.locator('div[data-testid="UserName"] h2'), default=username)
            user_handle = await safe_text(header.locator('div[data-testid="UserName"] div[dir="ltr"]'), default=f"@{username}")
            bio = await safe_text(header.locator('div[data-testid="UserDescription"]'), default=None)

            following_text = await safe_text(header.locator('a[href$="/following"]'))
            followers_text = await safe_text(header.locator('a[href$="/followers"]'))
            following_count = parse_engagement_string(following_text)
            followers_count = parse_engagement_string(followers_text)

            location = await safe_text(header.locator('span[data-testid="UserLocation"]'), default=None)
            join_date_str = await safe_text(header.locator('span[data-testid="UserJoinDate"]'), default=None)
            join_date = datetime.now()
            if join_date_str and join_date_str.startswith("Joined "):
                try:
                    join_date = datetime.strptime(join_date_str.replace("Joined ", ""), "%B %Y")
                except Exception:
                    join_date = datetime.now()

            is_verified = (await header.locator('svg[data-testid="icon-verified"]').count()) > 0
            posts_count_text = await safe_attr(page.locator('div[role="navigation"] a[href$="/posts"]'), 'title', default="")

            # Scroll more reliably to load posts
            for _ in range(3):
                await page.mouse.wheel(0, 3000)
                await asyncio.sleep(1.5)

            tweet_locator = page.locator('article[data-testid="tweet"]')
            total = await tweet_locator.count()
            limit = min(total, max_posts)

            scraped_posts = []
            for i in range(limit):
                t = tweet_locator.nth(i)
                try:
                    raw_text = await safe_text(t.locator('div[data-testid="tweetText"]'), default="")
                    text = raw_text.replace("\u200d", "").strip() if raw_text else ""

                    href = await safe_attr(t.locator('a[href*="/status/"]'), 'href', default=None)
                    if not href:
                        continue
                    tweet_url = urljoin("https://x.com", href)
                    try:
                        tweet_id = href.split("/status/")[1].split("?")[0]
                    except Exception:
                        tweet_id = href.rsplit("/", 1)[-1].split("?")[0]

                    replies = parse_engagement_string(await safe_attr(t.locator('div[data-testid="reply"]'), 'aria-label'))
                    retweets = parse_engagement_string(await safe_attr(t.locator('div[data-testid="retweet"]'), 'aria-label'))
                    likes = parse_engagement_string(await safe_attr(t.locator('div[data-testid="like"]'), 'aria-label'))
                    views = parse_engagement_string(await safe_attr(t.locator('a[href$="/analytics"]'), 'aria-label'))

                    time_str = await safe_text(t.locator('time'), default=None)
                    created_at = datetime.now()
                    if time_str:
                        try:
                            created_at = datetime.strptime(time_str, "%I:%M %p · %b %d, %Y")
                        except Exception:
                            pass

                    post_media_items = []
                    imgs = t.locator('div[data-testid="tweetPhoto"] img')
                    img_count = await imgs.count()
                    for img_idx in range(img_count):
                        src = await imgs.nth(img_idx).get_attribute('src')
                        if src:
                            post_media_items.append(MediaItem(media_type="IMAGE", url=src))

                    mentioned_accounts = [word for word in text.split() if word.startswith('@')]
                    hashtags = [word.rstrip('",.') for word in text.split() if word.startswith('#')]

                    scraped_posts.append(RecentPost(
                        post_id=tweet_id,
                        post_url=tweet_url,
                        text=text,
                        created_at=created_at,
                        like_count=likes,
                        reply_count=replies,
                        retweet_count=retweets,
                        view_count=views,
                        media=post_media_items,
                        mentioned_accounts=mentioned_accounts,
                        hashtags=hashtags
                    ))
                except Exception as e:
                    logger.warning(f"Skipping tweet {i+1}/{limit}: {e}")

            profile = AuthorProfile(
                author_id=user_handle or f"@{username}",
                username=(user_handle or f"@{username}").replace("@", ""),
                display_name=display_name or username,
                bio=bio,
                followers_count=followers_count,
                following_count=following_count,
                posts_count=posts_count_text or "",
                account_created_at=join_date,
                is_verified=is_verified,
                profile_url=profile_url,
                location=location,
                recent_posts=scraped_posts
            )

            content = f"Profile: @{profile.username}"
            if display_name and display_name != f"@{profile.username}":
                content += f" ({display_name})"
            if bio:
                content += f" - {bio}"
            content += f" | Followers: {followers_count}, Following: {following_count}, Posts: {posts_count_text or 'N/A'}"
            if scraped_posts:
                content += f" | Recent activity: {len(scraped_posts)} recent posts"

            evidence_items.append(EvidenceItem(
                source_type=SourceType.PROFILE_ANALYSIS,
                url=profile_url,
                content=content,
                timestamp=join_date,
                author_id=profile.author_id,
                mentioned_accounts=sum((p.mentioned_accounts for p in scraped_posts), []),
                hashtags=sum((p.hashtags for p in scraped_posts), []),
                media=sum((p.media for p in scraped_posts), []),
                raw_data=profile.model_dump()
            ))
            return evidence_items

        except Exception as e:
            logger.exception(f"Error while scraping @{username}: {e}")
            return [EvidenceItem(content=f"Unexpected error scraping '{username}': {e}", source_type=SourceType.TWITTER)]
        finally:
            await browser.close()

# Example run
async def main():
    target_user = "elonmusk"
    results = await get_author_profile.ainvoke({"username": target_user})
    print(results)
    for item in results:
        print(item.model_dump_json(indent=2))

if __name__ == "__main__":
    asyncio.run(main())
