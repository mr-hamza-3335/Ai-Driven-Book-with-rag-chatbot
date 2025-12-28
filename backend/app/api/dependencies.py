"""Dependency injection for API routes."""

from functools import lru_cache
from typing import Generator

from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.database import get_db as db_get_db
from app.services.embedding_service import EmbeddingService
from app.services.vector_service import VectorService
from app.services.llm_service import LLMService
from app.services.chunking_service import ChunkingService


def get_db() -> Generator[Session, None, None]:
    """Get database session dependency."""
    yield from db_get_db()


@lru_cache
def get_embedding_service() -> EmbeddingService:
    """Get embedding service singleton."""
    settings = get_settings()
    return EmbeddingService(api_key=settings.gemini_api_key)


@lru_cache
def get_vector_service() -> VectorService:
    """Get vector service singleton."""
    settings = get_settings()
    return VectorService(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        collection_name=settings.qdrant_collection_name,
    )


@lru_cache
def get_llm_service() -> LLMService:
    """Get LLM service singleton."""
    settings = get_settings()
    return LLMService(api_key=settings.gemini_api_key)


@lru_cache
def get_chunking_service() -> ChunkingService:
    """Get chunking service singleton."""
    return ChunkingService()


def get_settings_dependency() -> Settings:
    """Get settings dependency."""
    return get_settings()
