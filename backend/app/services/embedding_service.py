"""Gemini embedding service for generating text embeddings."""

import logging
from typing import List

import google.generativeai as genai

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating text embeddings using Gemini."""

    MODEL = "models/text-embedding-004"
    DIMENSIONS = 768  # Gemini embedding dimensions

    def __init__(self, api_key: str):
        """Initialize the embedding service.

        Args:
            api_key: Gemini API key
        """
        genai.configure(api_key=api_key)

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            List of embedding dimensions
        """
        result = genai.embed_content(
            model=self.MODEL,
            content=text,
            task_type="retrieval_document",
        )
        return result["embedding"]

    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a query (optimized for retrieval).

        Args:
            text: Query text to embed

        Returns:
            List of embedding dimensions
        """
        result = genai.embed_content(
            model=self.MODEL,
            content=text,
            task_type="retrieval_query",
        )
        return result["embedding"]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embeddings
        """
        if not texts:
            return []

        embeddings = []
        for text in texts:
            embedding = self.embed_text(text)
            embeddings.append(embedding)

        return embeddings
