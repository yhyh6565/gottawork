"""Google search utilities."""
from typing import List, Optional

from langchain.tools import Tool
from pydantic import BaseModel, Field

from config import config

try:
    from googlesearch import search as google_search_lib
    GOOGLESEARCH_AVAILABLE = True
except ImportError:
    GOOGLESEARCH_AVAILABLE = False


class SearchResult(BaseModel):
    """Search result data."""

    url: str
    title: Optional[str] = None
    snippet: Optional[str] = None


class GoogleSearcher:
    """Google search wrapper."""

    def __init__(self):
        self.max_results = config.search.max_search_results

    def search(self, query: str, num_results: Optional[int] = None) -> List[SearchResult]:
        """Search Google for a query.

        Args:
            query: The search query
            num_results: Number of results to return (defaults to config value)

        Returns:
            List of SearchResult objects
        """
        if not GOOGLESEARCH_AVAILABLE:
            raise ImportError(
                "googlesearch-python is not installed. "
                "Install with: pip install googlesearch-python"
            )

        num = num_results or self.max_results
        results = []

        try:
            # Use the free googlesearch library
            urls = google_search_lib(query, num_results=num, lang="ko")

            for url in urls:
                results.append(SearchResult(url=url))

        except Exception as e:
            print(f"Google search error: {e}")

        return results

    def search_with_snippets(self, query: str) -> str:
        """Search and return formatted results with snippets.

        Args:
            query: The search query

        Returns:
            Formatted string of search results
        """
        results = self.search(query)

        if not results:
            return f"No results found for: {query}"

        output = [f"Search results for '{query}':\n"]
        for i, result in enumerate(results, 1):
            output.append(f"{i}. {result.url}")
            if result.title:
                output.append(f"   Title: {result.title}")
            if result.snippet:
                output.append(f"   {result.snippet}")
            output.append("")

        return "\n".join(output)


def create_google_search_tool() -> Tool:
    """Create a LangChain tool for Google search."""
    searcher = GoogleSearcher()

    return Tool(
        name="google_search",
        description=(
            "Search Google for information. Input should be a search query string. "
            "Returns a list of relevant URLs and snippets. Useful for finding recent information, "
            "news, and general web content about characters, web novels, or entertainment sources."
        ),
        func=searcher.search_with_snippets,
    )
