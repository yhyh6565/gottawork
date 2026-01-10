"""Twitter/X search utilities."""
from typing import List, Optional
from datetime import datetime

from langchain.tools import Tool
from pydantic import BaseModel, Field

from config import config

try:
    import tweepy
    TWEEPY_AVAILABLE = True
except ImportError:
    TWEEPY_AVAILABLE = False


class Tweet(BaseModel):
    """Tweet data."""

    id: str
    text: str
    author: str
    created_at: datetime
    likes: int = 0
    retweets: int = 0
    url: str


class TwitterSearcher:
    """Twitter/X search wrapper."""

    def __init__(self):
        self.bearer_token = config.search.twitter_bearer_token
        self.client = None

        if TWEEPY_AVAILABLE and self.bearer_token:
            try:
                self.client = tweepy.Client(bearer_token=self.bearer_token)
            except Exception as e:
                print(f"Failed to initialize Twitter client: {e}")

    def search(
        self, query: str, max_results: int = 10, language: str = "ko"
    ) -> List[Tweet]:
        """Search Twitter for tweets.

        Args:
            query: The search query
            max_results: Maximum number of tweets to return
            language: Language filter (default: Korean)

        Returns:
            List of Tweet objects
        """
        if not TWEEPY_AVAILABLE:
            raise ImportError("tweepy is not installed. Install with: pip install tweepy")

        if not self.client:
            raise ValueError(
                "Twitter API not configured. Set TWITTER_BEARER_TOKEN in .env file"
            )

        tweets = []

        try:
            # Search recent tweets
            response = self.client.search_recent_tweets(
                query=f"{query} lang:{language}",
                max_results=min(max_results, 100),
                tweet_fields=["created_at", "public_metrics", "author_id"],
                user_fields=["username"],
                expansions=["author_id"],
            )

            if not response.data:
                return tweets

            # Create user lookup
            users = {user.id: user.username for user in response.includes.get("users", [])}

            for tweet in response.data:
                metrics = tweet.public_metrics or {}
                author = users.get(tweet.author_id, "unknown")

                tweets.append(
                    Tweet(
                        id=str(tweet.id),
                        text=tweet.text,
                        author=author,
                        created_at=tweet.created_at,
                        likes=metrics.get("like_count", 0),
                        retweets=metrics.get("retweet_count", 0),
                        url=f"https://twitter.com/{author}/status/{tweet.id}",
                    )
                )

        except Exception as e:
            print(f"Twitter search error: {e}")

        return tweets

    def search_formatted(self, query: str) -> str:
        """Search Twitter and return formatted results.

        Args:
            query: The search query

        Returns:
            Formatted string of tweets
        """
        if not TWEEPY_AVAILABLE:
            return "Twitter search is not available. Install tweepy: pip install tweepy"

        if not self.client:
            return "Twitter API not configured. Set TWITTER_BEARER_TOKEN in .env file"

        tweets = self.search(query)

        if not tweets:
            return f"No tweets found for: {query}"

        output = [f"Twitter search results for '{query}':\n"]

        for i, tweet in enumerate(tweets, 1):
            output.append(f"{i}. @{tweet.author} ({tweet.created_at.strftime('%Y-%m-%d %H:%M')})")
            output.append(f"   {tweet.text}")
            output.append(
                f"   💚 {tweet.likes} | 🔁 {tweet.retweets} | {tweet.url}"
            )
            output.append("")

        return "\n".join(output)


def create_twitter_search_tool() -> Tool:
    """Create a LangChain tool for Twitter search."""
    searcher = TwitterSearcher()

    return Tool(
        name="twitter_search",
        description=(
            "Search Twitter/X for tweets about a topic. Input should be a search query. "
            "Returns recent tweets with engagement metrics. Useful for finding fan reactions, "
            "trending discussions, and community sentiment about characters or content."
        ),
        func=searcher.search_formatted,
    )
