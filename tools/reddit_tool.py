from typing import List, Union
import praw
from langchain.tools import tool
from datetime import datetime
import os
from dotenv import load_dotenv
load_dotenv()

# --- IMPORTANT ---
# Fill in your Reddit API credentials here.
# You can get these for free by creating a "script" app on your Reddit account.
REDDIT_CLIENT_ID = "DJrzVtzLeN7ZLTiKOU7WNA"
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = "Veris by u/Lucky_Initiative8899"

import praw
from langchain.tools import tool
from datetime import datetime
from typing import List, Dict, Union, Optional
import logging
import json

logger = logging.getLogger(__name__)

# Configuration - these should be set via environment variables in production
REDDIT_CONFIG = {
    "client_id": REDDIT_CLIENT_ID,
    "client_secret": REDDIT_CLIENT_SECRET, 
    "user_agent": REDDIT_USER_AGENT
}

class RedditSearcher:
    """Enhanced Reddit search functionality."""
    
    def __init__(self):
        self.reddit = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Reddit client with proper error handling."""
        try:
            if all(val != f"YOUR_{key.upper()}" for key, val in REDDIT_CONFIG.items()):
                self.reddit = praw.Reddit(**REDDIT_CONFIG)
                # Test connection
                self.reddit.user.me()
                logger.info("Reddit client initialized successfully")
            else:
                logger.warning("Reddit API credentials not configured")
        except Exception as e:
            logger.error(f"Failed to initialize Reddit client: {e}")
            self.reddit = None
    
    def search_submissions(self, query: str, limit: int = 10, subreddit: str = "all", 
                          time_filter: str = "all") -> List[Dict]:
        """Enhanced submission search with better filtering."""
        if not self.reddit:
            return []
        
        try:
            target_subreddit = self.reddit.subreddit(subreddit)
            submissions = target_subreddit.search(
                query, 
                limit=limit, 
                time_filter=time_filter,
                sort='relevance'
            )
            
            results = []
            for submission in submissions:
                # Get top comments for more context
                submission.comments.replace_more(limit=0)
                top_comments = []
                for comment in submission.comments.list()[:3]:  # Top 3 comments
                    if hasattr(comment, 'body') and len(comment.body) > 10:
                        top_comments.append({
                            'author': str(comment.author) if comment.author else 'Unknown',
                            'body': comment.body[:200],
                            'score': comment.score
                        })
                
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
def enhanced_reddit_search(query: str, limit: int = 10, subreddit: str = "all", 
                          include_comments: bool = True) -> str:
    """
    Enhanced Reddit search with better content extraction and comment inclusion.
    """
    if not reddit_searcher.reddit:
        return "Reddit API not configured. Please set up credentials in the configuration."
    
    try:
        results = reddit_searcher.search_submissions(query, limit, subreddit)
        
        if not results:
            return f"No results found for query: {query}"
        
        formatted_results = []
        for post in results:
            post_info = f"""
Title: {post['title']}
Subreddit: r/{post['subreddit']}
Author: u/{post['author']}
Score: {post['score']} | Comments: {post['num_comments']}
Posted: {post['timestamp']}
URL: {post['url']}
{f"Flair: {post['flair']}" if post['flair'] else ""}

Content: {post['content'][:300] if post['content'] else 'No text content'}

{f"Top Comments:" if include_comments and post['top_comments'] else ""}
"""
            
            if include_comments and post['top_comments']:
                for i, comment in enumerate(post['top_comments'], 1):
                    post_info += f"\n  Comment {i} (Score: {comment['score']}) by u/{comment['author']}: {comment['body']}\n"
            
            formatted_results.append(post_info)
        
        return "\n" + "="*80 + "\n".join(formatted_results)
        
    except Exception as e:
        return f"Reddit search error: {str(e)}"

@tool
def reddit_subreddit_analysis(subreddit_name: str, limit: int = 20) -> str:
    """
    Analyzes a specific subreddit for trending topics and discussions.
    """
    if not reddit_searcher.reddit:
        return "Reddit API not configured."
    
    try:
        subreddit = reddit_searcher.reddit.subreddit(subreddit_name)
        
        # Get hot posts
        hot_posts = list(subreddit.hot(limit=limit))
        
        analysis = f"Subreddit Analysis: r/{subreddit_name}\n"
        analysis += f"Subscribers: {subreddit.subscribers:,}\n"
        analysis += f"Description: {subreddit.public_description}\n\n"
        analysis += "Hot Posts:\n" + "="*50 + "\n"
        
        for i, post in enumerate(hot_posts, 1):
            analysis += f"{i}. {post.title}\n"
            analysis += f"   Score: {post.score} | Comments: {post.num_comments}\n"
            analysis += f"   Author: u/{post.author}\n"
            analysis += f"   URL: https://reddit.com{post.permalink}\n\n"
        
        return analysis
        
    except Exception as e:
        return f"Subreddit analysis failed: {str(e)}"

if __name__ == "__main__":
    # Correct LangChain call
    print(REDDIT_CLIENT_SECRET)
    print(enhanced_reddit_search.invoke({
        "query": "Why do Indians like Modi so much? Why did the previous Indian leaders not get the reception overseas?",
        "limit": 5,
        "subreddit": "technology",
        "include_comments": True
    }))