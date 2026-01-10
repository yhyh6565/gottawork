"""Configuration management for the research agent."""
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()


class LLMConfig(BaseModel):
    """LLM configuration."""

    anthropic_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY"))
    openai_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    default_model: str = Field(default_factory=lambda: os.getenv("DEFAULT_MODEL", "claude-3-5-sonnet-20241022"))
    temperature: float = Field(default_factory=lambda: float(os.getenv("TEMPERATURE", "0.7")))
    max_tokens: int = Field(default_factory=lambda: int(os.getenv("MAX_TOKENS", "4096")))


class SearchConfig(BaseModel):
    """Search and scraping configuration."""

    google_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("GOOGLE_API_KEY"))
    google_cse_id: Optional[str] = Field(default_factory=lambda: os.getenv("GOOGLE_CSE_ID"))
    twitter_bearer_token: Optional[str] = Field(default_factory=lambda: os.getenv("TWITTER_BEARER_TOKEN"))
    max_search_results: int = Field(default_factory=lambda: int(os.getenv("MAX_SEARCH_RESULTS", "10")))
    scraping_delay: float = Field(default_factory=lambda: float(os.getenv("SCRAPING_DELAY", "1.0")))
    user_agent: str = Field(
        default_factory=lambda: os.getenv(
            "USER_AGENT", "Mozilla/5.0 (compatible; ResearchAgent/1.0)"
        )
    )


class VectorStoreConfig(BaseModel):
    """Vector store configuration."""

    chroma_persist_dir: Path = Field(
        default_factory=lambda: Path(os.getenv("CHROMA_PERSIST_DIR", "./data/chroma"))
    )
    collection_name: str = Field(
        default_factory=lambda: os.getenv("COLLECTION_NAME", "character_knowledge")
    )


class Config(BaseModel):
    """Main configuration."""

    llm: LLMConfig = Field(default_factory=LLMConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    outputs_dir: Path = Path("./outputs")

    def __init__(self, **data):
        super().__init__(**data)
        # Create necessary directories
        self.vector_store.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)


# Global config instance
config = Config()
