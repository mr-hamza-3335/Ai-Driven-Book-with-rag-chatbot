"""Indexing service for book content."""

import logging
import uuid
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

from app.models.index import IndexStatus, ChapterContent
from app.services.embedding_service import EmbeddingService
from app.services.vector_service import VectorService
from app.services.chunking_service import ChunkingService

logger = logging.getLogger(__name__)


@dataclass
class IndexingJob:
    """Represents an indexing job."""

    job_id: str
    status: IndexStatus
    total_chapters: int
    chapters_processed: int = 0
    chunks_created: int = 0
    error_message: Optional[str] = None
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None


class IndexingService:
    """Service for indexing book content into vector store."""

    # In-memory job storage (for simplicity; could use Redis/DB in production)
    _jobs: Dict[str, IndexingJob] = {}

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_service: VectorService,
        chunking_service: ChunkingService,
    ):
        """Initialize indexing service.

        Args:
            embedding_service: Service for generating embeddings
            vector_service: Service for vector storage
            chunking_service: Service for text chunking
        """
        self.embedding_service = embedding_service
        self.vector_service = vector_service
        self.chunking_service = chunking_service

    def chunk_and_embed(
        self,
        chapter: ChapterContent,
    ) -> tuple[List[Dict[str, Any]], List[List[float]]]:
        """Chunk and embed a single chapter.

        Args:
            chapter: Chapter content to process

        Returns:
            Tuple of (chunks, embeddings)
        """
        # Create chunks
        chunks = self.chunking_service.chunk_chapter(
            chapter_id=chapter.chapter_id,
            title=chapter.title,
            content=chapter.content,
        )

        # Generate embeddings for all chunks
        chunk_texts = [chunk["content"] for chunk in chunks]
        embeddings = self.embedding_service.embed_batch(chunk_texts)

        return chunks, embeddings

    def index_chapters(
        self,
        chapters: List[ChapterContent],
        force_reindex: bool = False,
    ) -> IndexingJob:
        """Index multiple chapters.

        Args:
            chapters: List of chapters to index
            force_reindex: If true, delete existing chunks first

        Returns:
            IndexingJob with progress information
        """
        job_id = str(uuid.uuid4())
        job = IndexingJob(
            job_id=job_id,
            status=IndexStatus.IN_PROGRESS,
            total_chapters=len(chapters),
        )
        self._jobs[job_id] = job

        logger.info(f"Starting indexing job {job_id} for {len(chapters)} chapters")

        try:
            for i, chapter in enumerate(chapters):
                # Delete existing chunks if force_reindex
                if force_reindex:
                    self.vector_service.delete_by_chapter(chapter.chapter_id)

                # Chunk and embed
                chunks, embeddings = self.chunk_and_embed(chapter)

                # Store in vector database
                count = self.vector_service.upsert_chunks(chunks, embeddings)

                job.chunks_created += count
                job.chapters_processed = i + 1

                logger.info(
                    f"Indexed chapter {chapter.chapter_id}: "
                    f"{count} chunks ({job.chapters_processed}/{job.total_chapters})"
                )

            job.status = IndexStatus.COMPLETED
            job.completed_at = datetime.utcnow()

            logger.info(
                f"Indexing job {job_id} completed: "
                f"{job.chapters_processed} chapters, {job.chunks_created} chunks"
            )

        except Exception as e:
            logger.error(f"Indexing job {job_id} failed: {e}", exc_info=True)
            job.status = IndexStatus.FAILED
            job.error_message = str(e)

        return job

    def get_status(self, job_id: str) -> Optional[IndexingJob]:
        """Get status of an indexing job.

        Args:
            job_id: Job identifier

        Returns:
            IndexingJob or None if not found
        """
        return self._jobs.get(job_id)

    def get_progress_percent(self, job: IndexingJob) -> float:
        """Calculate progress percentage.

        Args:
            job: Indexing job

        Returns:
            Progress as percentage (0-100)
        """
        if job.total_chapters == 0:
            return 0.0
        return (job.chapters_processed / job.total_chapters) * 100
