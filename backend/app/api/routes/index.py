"""Index endpoint for book content ingestion."""

import logging

from fastapi import APIRouter, Depends, HTTPException

from app.models.index import (
    IndexRequest,
    IndexResponse,
    IndexStatusResponse,
    IndexStatus,
)
from app.services.indexing_service import IndexingService
from app.services.embedding_service import EmbeddingService
from app.services.vector_service import VectorService
from app.services.chunking_service import ChunkingService
from app.api.dependencies import (
    get_embedding_service,
    get_vector_service,
    get_chunking_service,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["index"])


def get_indexing_service(
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    vector_service: VectorService = Depends(get_vector_service),
    chunking_service: ChunkingService = Depends(get_chunking_service),
) -> IndexingService:
    """Get indexing service with dependencies."""
    return IndexingService(
        embedding_service=embedding_service,
        vector_service=vector_service,
        chunking_service=chunking_service,
    )


@router.post(
    "/index",
    response_model=IndexResponse,
    responses={
        400: {"description": "Validation error"},
        500: {"description": "Server error"},
    },
)
async def index_content(
    request: IndexRequest,
    indexing_service: IndexingService = Depends(get_indexing_service),
) -> IndexResponse:
    """Index book content into the vector store.

    Args:
        request: Index request with chapters to process

    Returns:
        IndexResponse with job status and counts
    """
    logger.info(
        f"Starting indexing: {len(request.chapters)} chapters, "
        f"force_reindex={request.force_reindex}"
    )

    try:
        job = indexing_service.index_chapters(
            chapters=request.chapters,
            force_reindex=request.force_reindex,
        )

        return IndexResponse(
            job_id=job.job_id,
            status=job.status,
            chapters_indexed=job.chapters_processed,
            chunks_created=job.chunks_created,
        )

    except Exception as e:
        logger.error(f"Indexing error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Indexing failed: {str(e)}",
        )


@router.get(
    "/index/{job_id}",
    response_model=IndexStatusResponse,
    responses={
        404: {"description": "Job not found"},
    },
)
async def get_index_status(
    job_id: str,
    indexing_service: IndexingService = Depends(get_indexing_service),
) -> IndexStatusResponse:
    """Get status of an indexing job.

    Args:
        job_id: Job identifier

    Returns:
        IndexStatusResponse with current progress
    """
    job = indexing_service.get_status(job_id)

    if not job:
        raise HTTPException(
            status_code=404,
            detail=f"Job not found: {job_id}",
        )

    return IndexStatusResponse(
        job_id=job.job_id,
        status=job.status,
        progress_percent=indexing_service.get_progress_percent(job),
        chapters_processed=job.chapters_processed,
        total_chapters=job.total_chapters,
        error_message=job.error_message,
    )
