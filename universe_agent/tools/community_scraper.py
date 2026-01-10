"""Community and fan site scraping utilities."""
from typing import List, Optional
from urllib.parse import quote

from langchain.tools import Tool
from pydantic import BaseModel

from tools.web_scraper import WebScraper


class CommunityPost(BaseModel):
    """Community post data."""

    title: str
    url: str
    content: str
    source: str  # e.g., "namu.wiki", "reddit", etc.


class CommunityScraper:
    """Scraper for various community sites and fan communities."""

    def __init__(self):
        self.scraper = WebScraper()

    def search_namuwiki(self, query: str) -> Optional[CommunityPost]:
        """Search and scrape from Namu Wiki (Korean community wiki).

        Args:
            query: The search term (character name, work title, etc.)

        Returns:
            CommunityPost with wiki content or None
        """
        encoded_query = quote(query)
        url = f"https://namu.wiki/w/{encoded_query}"

        try:
            page = self.scraper.scrape(url)

            if "존재하지 않는" in page.content or "error" in page.metadata:
                return None

            return CommunityPost(
                title=page.title or query,
                url=url,
                content=page.content[:3000],  # Limit content length
                source="namu.wiki",
            )
        except Exception as e:
            print(f"Namu Wiki scraping error: {e}")
            return None

    def search_communities(self, query: str) -> str:
        """Search multiple community sources for information.

        Args:
            query: The search term

        Returns:
            Formatted string with community information
        """
        results = []

        # Try Namu Wiki
        namu_result = self.search_namuwiki(query)
        if namu_result:
            results.append(namu_result)

        if not results:
            return f"No community information found for: {query}"

        output = [f"Community search results for '{query}':\n"]

        for i, post in enumerate(results, 1):
            output.append(f"{i}. [{post.source}] {post.title}")
            output.append(f"   URL: {post.url}")
            output.append(f"   Content preview:")
            # Show first 500 characters of content
            preview = post.content[:500].replace("\n", "\n   ")
            output.append(f"   {preview}...")
            output.append("")

        return "\n".join(output)


def create_community_search_tool() -> Tool:
    """Create a LangChain tool for community searching."""
    scraper = CommunityScraper()

    return Tool(
        name="community_search",
        description=(
            "Search fan communities and wikis for character and content information. "
            "Input should be a character name or work title. "
            "Returns information from community sources like Namu Wiki. "
            "Useful for finding detailed character profiles, background information, "
            "and fan-created content."
        ),
        func=scraper.search_communities,
    )
