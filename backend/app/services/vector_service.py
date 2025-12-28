"""Qdrant vector database service for similarity search."""

import logging
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    PayloadSchemaType,
)

logger = logging.getLogger(__name__)


class VectorService:
    """Service for Qdrant vector database operations."""

    VECTOR_SIZE = 768  # Gemini text-embedding-004 dimensions

    def __init__(self, url: str, api_key: str, collection_name: str):
        """Initialize the vector service.

        Args:
            url: Qdrant Cloud URL
            api_key: Qdrant API key
            collection_name: Name of the collection
        """
        self.client = QdrantClient(url=url, api_key=api_key)
        self.collection_name = collection_name

    def ensure_collection(self) -> None:
        """Create collection if it doesn't exist."""
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]

        if self.collection_name not in collection_names:
            logger.info(f"Creating collection: {self.collection_name}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.VECTOR_SIZE,
                    distance=Distance.COSINE,
                ),
            )
            # Create payload index for chapter_id filtering
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="chapter_id",
                field_schema=PayloadSchemaType.KEYWORD,
            )
            logger.info("Created payload index for chapter_id")

    def upsert_chunks(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]],
    ) -> int:
        """Upsert chunks with their embeddings.

        Args:
            chunks: List of chunk data with payload
            embeddings: List of embedding vectors

        Returns:
            Number of points upserted
        """
        self.ensure_collection()

        points = []
        for chunk, embedding in zip(chunks, embeddings):
            # Generate UUID from chunk_id for consistent IDs
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk["chunk_id"]))
            point = PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "chunk_id": chunk["chunk_id"],
                    "chapter_id": chunk["chapter_id"],
                    "section_title": chunk.get("section_title", ""),
                    "content": chunk["content"],
                    "chunk_index": chunk.get("chunk_index", 0),
                    "word_count": len(chunk["content"].split()),
                    "created_at": datetime.utcnow().isoformat(),
                },
            )
            points.append(point)

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

        logger.info(f"Upserted {len(points)} chunks to {self.collection_name}")
        return len(points)

    def search(
        self,
        query_vector: List[float],
        limit: int = 5,
        chapter_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar chunks.

        Args:
            query_vector: Query embedding vector
            limit: Maximum number of results
            chapter_filter: Optional chapter ID to filter by

        Returns:
            List of matching chunks with scores
        """
        query_filter = None
        if chapter_filter:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="chapter_id",
                        match=MatchValue(value=chapter_filter),
                    )
                ]
            )

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
            query_filter=query_filter,
        )

        return [
            {
                "chunk_id": hit.payload.get("chunk_id"),
                "chapter_id": hit.payload.get("chapter_id"),
                "section_title": hit.payload.get("section_title"),
                "content": hit.payload.get("content"),
                "relevance_score": hit.score,
            }
            for hit in results.points
        ]

    def delete_by_chapter(self, chapter_id: str) -> None:
        """Delete all chunks for a chapter.

        Args:
            chapter_id: Chapter ID to delete
        """
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="chapter_id",
                        match=MatchValue(value=chapter_id),
                    )
                ]
            ),
        )
        logger.info(f"Deleted chunks for chapter: {chapter_id}")

    def get_collection_info(self) -> Dict[str, Any]:
        """Get collection information.

        Returns:
            Collection info including point count
        """
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "points_count": info.points_count,
                "status": "ok",
            }
        except Exception as e:
            return {
                "name": self.collection_name,
                "points_count": 0,
                "status": "error",
                "error": str(e),
            }
