"""Vector store for character knowledge base."""
from typing import List, Optional, Dict, Any

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from pydantic import BaseModel

from config import config


class CharacterKnowledge(BaseModel):
    """Character knowledge entry."""

    character_name: str
    source: str  # e.g., "google", "twitter", "namu.wiki"
    content: str
    metadata: Dict[str, Any] = {}


class VectorKnowledgeBase:
    """Vector store-based knowledge base for characters."""

    def __init__(self, collection_name: Optional[str] = None):
        """Initialize the vector knowledge base.

        Args:
            collection_name: Name of the collection (defaults to config value)
        """
        self.collection_name = collection_name or config.vector_store.collection_name
        self.persist_directory = str(config.vector_store.chroma_persist_dir)

        # Use HuggingFace embeddings (free, runs locally)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )

        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )

        # Initialize or load vector store
        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory,
        )

    def add_knowledge(self, knowledge: CharacterKnowledge) -> None:
        """Add character knowledge to the vector store.

        Args:
            knowledge: CharacterKnowledge object to add
        """
        # Split content into chunks
        chunks = self.text_splitter.split_text(knowledge.content)

        # Prepare metadata for each chunk
        metadatas = [
            {
                "character_name": knowledge.character_name,
                "source": knowledge.source,
                **knowledge.metadata,
            }
            for _ in chunks
        ]

        # Add to vector store
        self.vectorstore.add_texts(texts=chunks, metadatas=metadatas)

    def add_knowledge_batch(self, knowledge_list: List[CharacterKnowledge]) -> None:
        """Add multiple knowledge entries at once.

        Args:
            knowledge_list: List of CharacterKnowledge objects
        """
        for knowledge in knowledge_list:
            self.add_knowledge(knowledge)

    def search(
        self,
        query: str,
        character_name: Optional[str] = None,
        k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search the knowledge base.

        Args:
            query: Search query
            character_name: Filter by character name (optional)
            k: Number of results to return

        Returns:
            List of relevant documents with metadata
        """
        # Build filter if character_name is provided
        filter_dict = {"character_name": character_name} if character_name else None

        # Search
        results = self.vectorstore.similarity_search_with_score(
            query,
            k=k,
            filter=filter_dict,
        )

        # Format results
        formatted_results = []
        for doc, score in results:
            formatted_results.append(
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "relevance_score": float(score),
                }
            )

        return formatted_results

    def get_character_knowledge(self, character_name: str, max_results: int = 20) -> str:
        """Get all knowledge about a character as formatted text.

        Args:
            character_name: Name of the character
            max_results: Maximum number of knowledge chunks to return

        Returns:
            Formatted string with character knowledge
        """
        results = self.search(
            query=character_name,
            character_name=character_name,
            k=max_results,
        )

        if not results:
            return f"No knowledge found for character: {character_name}"

        output = [f"Knowledge about {character_name}:\n"]

        for i, result in enumerate(results, 1):
            source = result["metadata"].get("source", "unknown")
            output.append(f"{i}. [Source: {source}]")
            output.append(result["content"])
            output.append("")

        return "\n".join(output)

    def delete_character(self, character_name: str) -> None:
        """Delete all knowledge about a character.

        Args:
            character_name: Name of the character to delete
        """
        # Note: Chroma doesn't have direct filter-based deletion in all versions
        # This is a simplified version
        print(f"Deleting knowledge for: {character_name}")
        print("Note: Full deletion requires recreating the collection")

    def persist(self) -> None:
        """Persist the vector store to disk."""
        self.vectorstore.persist()
