# reddit_tool.py - Refactored
from typing import List
import praw
from langchain_core.tools import tool
from datetime import datetime
import os
from dotenv import load_dotenv
from schema import EvidenceItem, SourceType

load_dotenv()

REDDIT_CLIENT_ID = "DJrzVtzLeN7ZLTiKOU7WNA"
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = "Veris by u/Lucky_Initiative8899"

import logging
logger = logging.getLogger(__name__)

REDDIT_CONFIG = {"client_id": REDDIT_CLIENT_ID, "client_secret": REDDIT_CLIENT_SECRET, "user_agent": REDDIT_USER_AGENT}

class RedditSearcher:
    def __init__(self):
        self.reddit = None
        self._initialize_client()

    def _initialize_client(self):
        try:
            if all(val != f"YOUR_{key.upper()}" for key, val in REDDIT_CONFIG.items()):
                self.reddit = praw.Reddit(**REDDIT_CONFIG)
                self.reddit.user.me()
                logger.info("Reddit client initialized successfully")
            else:
                logger.warning("Reddit API credentials not configured")
        except Exception as e:
            logger.error(f"Failed to initialize Reddit client: {e}")
            self.reddit = None

    def search_submissions(self, query: str, limit: int = 10, subreddit: str = "all") -> List[dict]:
        if not self.reddit:
            return []
        try:
            target_subreddit = self.reddit.subreddit(subreddit)
            submissions = target_subreddit.search(query, limit=limit, sort='relevance')
            results = []
            for submission in submissions:
                submission.comments.replace_more(limit=0)
                top_comments = [{"author": str(c.author) if c.author else 'Unknown', "body": c.body[:200], "score": c.score} for c in submission.comments.list()[:3] if hasattr(c, 'body') and len(c.body) > 10]
                results.append({
                    'id': submission.id,
                    'title': submission.title,
                    'url': f"https://reddit.com{submission.permalink}",
                    'full_url': submission.url,
                    'author': str(submission.author) if submission.author else 'Unknown',
                    'timestamp': datetime.utcfromtimestamp(submission.created_utc).isoformat(),
                    'content': submission.selftext,
                    'subreddit': str(submission.subreddit),
                    'score': submission.score,
                    'num_comments': submission.num_comments,
                    'top_comments': top_comments,
                    'flair': submission.link_flair_text or ''
                })
            return results
        except Exception as e:
            logger.error(f"Reddit search failed: {e}")
            return []

reddit_searcher = RedditSearcher()

@tool
def enhanced_reddit_search(query: str, limit: int = 10, subreddit: str = "all") -> List[EvidenceItem]:
    """
    Enhanced Reddit search with better content extraction.
    Returns a list of EvidenceItem objects.
    """
    if not reddit_searcher.reddit:
        return [EvidenceItem(
            source_type=SourceType.REDDIT,
            content="Reddit API not configured. Please set up credentials in the configuration.",
            timestamp=datetime.now()
        )]
    try:
        results = reddit_searcher.search_submissions(query, limit, subreddit)
        evidence_items = []
        for post in results:
            evidence_items.append(EvidenceItem(
                source_type=SourceType.REDDIT,
                url=post["url"],
                content=f"{post['title']}: {post['content'][:300] if post['content'] else 'No text content'}",
                timestamp=datetime.fromisoformat(post["timestamp"]),
                author_id=post["author"],
                mentioned_accounts=[c["author"] for c in post["top_comments"]] if post["top_comments"] else [],
                raw_data={
                    "subreddit": post["subreddit"],
                    "score": post["score"],
                    "num_comments": post["num_comments"],
                    "top_comments": post["top_comments"],
                    "flair": post["flair"]
                }
            ))
        return evidence_items if evidence_items else [EvidenceItem(
            source_type=SourceType.REDDIT,
            content=f"No results found for query: {query}",
            timestamp=datetime.now()
        )]
    except Exception as e:
        return [EvidenceItem(
            source_type=SourceType.REDDIT,
            content=f"Reddit search error: {str(e)}",
            timestamp=datetime.now()
        )]

@tool
def reddit_subreddit_analysis(subreddit_name: str, limit: int = 20) -> List[EvidenceItem]:
    """
    Analyzes a specific subreddit for trending topics and discussions.
    Returns a list of EvidenceItem objects.
    """
    if not reddit_searcher.reddit:
        return [EvidenceItem(
            source_type=SourceType.REDDIT,
            content="Reddit API not configured.",
            timestamp=datetime.now()
        )]
    try:
        subreddit = reddit_searcher.reddit.subreddit(subreddit_name)
        hot_posts = list(subreddit.hot(limit=limit))
        evidence_items = []
        for i, post in enumerate(hot_posts, 1):
            evidence_items.append(EvidenceItem(
                source_type=SourceType.REDDIT,
                url=f"https://reddit.com{post.permalink}",
                content=f"{i}. {post.title}",
                timestamp=datetime.utcfromtimestamp(post.created_utc),
                author_id=str(post.author) if post.author else 'Unknown',
                raw_data={
                    "subreddit": subreddit_name,
                    "subscribers": subreddit.subscribers,
                    "score": post.score,
                    "num_comments": post.num_comments
                }
            ))
        return evidence_items if evidence_items else [EvidenceItem(
            source_type=SourceType.REDDIT,
            content=f"No posts found for subreddit: r/{subreddit_name}",
            timestamp=datetime.now()
        )]
    except Exception as e:
        return [EvidenceItem(
            source_type=SourceType.REDDIT,
            content=f"Subreddit analysis failed: {str(e)}",
            timestamp=datetime.now()
        )]

if __name__ == "__main__":
    print(enhanced_reddit_search.invoke({
        "query": "Why do Indians like Modi so much? Why did the previous Indian leaders not get the reception overseas?",
        "limit": 5,
        "subreddit": "technology"
    }))