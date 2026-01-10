"""Web scraping utilities."""
import time
from typing import Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from langchain.tools import Tool
from pydantic import BaseModel, Field

from config import config


class WebPage(BaseModel):
    """Scraped web page data."""

    url: str
    title: Optional[str] = None
    content: str
    metadata: dict = Field(default_factory=dict)


class WebScraper:
    """Web scraper for extracting content from URLs."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": config.search.user_agent})

    def scrape(self, url: str) -> WebPage:
        """Scrape content from a URL.

        Args:
            url: The URL to scrape

        Returns:
            WebPage object with extracted content
        """
        try:
            time.sleep(config.search.scraping_delay)
            response = self.session.get(url, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, "html.parser")

            # Remove script and style elements
            for element in soup(["script", "style", "nav", "footer", "header"]):
                element.decompose()

            # Extract title
            title = None
            if soup.title:
                title = soup.title.string
            elif soup.find("h1"):
                title = soup.find("h1").get_text(strip=True)

            # Extract main content
            content_tags = ["article", "main", "div[class*='content']", "div[class*='post']"]
            content = None

            for tag in content_tags:
                if "class" in tag:
                    element = soup.select_one(tag)
                else:
                    element = soup.find(tag)

                if element:
                    content = element.get_text(separator="\n", strip=True)
                    break

            # Fallback to body if no content found
            if not content:
                content = soup.get_text(separator="\n", strip=True)

            # Clean up excessive whitespace
            lines = [line.strip() for line in content.split("\n") if line.strip()]
            content = "\n".join(lines)

            # Extract metadata
            metadata = {
                "domain": urlparse(url).netloc,
                "description": self._extract_meta(soup, "description"),
                "keywords": self._extract_meta(soup, "keywords"),
            }

            return WebPage(url=url, title=title, content=content, metadata=metadata)

        except Exception as e:
            return WebPage(
                url=url,
                content=f"Error scraping {url}: {str(e)}",
                metadata={"error": str(e)},
            )

    def _extract_meta(self, soup: BeautifulSoup, property_name: str) -> Optional[str]:
        """Extract meta tag content."""
        meta = soup.find("meta", attrs={"name": property_name}) or soup.find(
            "meta", attrs={"property": f"og:{property_name}"}
        )
        return meta.get("content") if meta else None


def create_web_scraper_tool() -> Tool:
    """Create a LangChain tool for web scraping."""
    scraper = WebScraper()

    def scrape_wrapper(url: str) -> str:
        """Scrape a web page and return its content."""
        result = scraper.scrape(url)
        return f"Title: {result.title}\n\nContent:\n{result.content[:2000]}"

    return Tool(
        name="web_scraper",
        description="Scrape content from a web page URL. Returns the page title and main content.",
        func=scrape_wrapper,
    )
