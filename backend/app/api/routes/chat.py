"""Chat endpoint for RAG-based question answering."""

import logging
import time
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException

from app.models.chat import ChatRequest, ChatResponse, ChatError, Source
from app.services.rag_service import RAGService
from app.services.embedding_service import EmbeddingService
from app.services.vector_service import VectorService
from app.services.llm_service import LLMService
from app.api.dependencies import (
    get_embedding_service,
    get_vector_service,
    get_llm_service,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["chat"])


def get_rag_service(
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    vector_service: VectorService = Depends(get_vector_service),
    llm_service: LLMService = Depends(get_llm_service),
) -> RAGService:
    """Get RAG service with dependencies."""
    return RAGService(
        embedding_service=embedding_service,
        vector_service=vector_service,
        llm_service=llm_service,
    )


@router.post(
    "/chat",
    response_model=ChatResponse,
    responses={
        400: {"model": ChatError, "description": "Validation error"},
        429: {"model": ChatError, "description": "Rate limit exceeded"},
        500: {"model": ChatError, "description": "Server error"},
        503: {"model": ChatError, "description": "Service unavailable"},
        504: {"model": ChatError, "description": "Timeout"},
    },
)
async def chat(
    request: ChatRequest,
    rag_service: RAGService = Depends(get_rag_service),
) -> ChatResponse:
    """Process a chat query and return an answer with sources.

    Args:
        request: Chat request with question and optional selected text

    Returns:
        ChatResponse with answer, sources, and timing info
    """
    start_time = time.time()
    conversation_id = request.conversation_id or str(uuid.uuid4())

    logger.info(
        f"Processing chat request: question='{request.question[:50]}...', "
        f"selected_text={'yes' if request.selected_text else 'no'}, "
        f"conversation_id={conversation_id}"
    )

    try:
        # Get answer from RAG service
        answer, chunks = rag_service.answer(
            question=request.question,
            selected_text=request.selected_text,
        )

        # Convert chunks to sources
        sources = [
            Source(
                chapter_id=chunk.chapter_id,
                section_title=chunk.section_title,
                snippet=chunk.content[:500],  # Truncate to max length
                relevance_score=chunk.relevance_score,
            )
            for chunk in chunks
        ]

        # Calculate processing time
        processing_time_ms = int((time.time() - start_time) * 1000)

        logger.info(
            f"Chat response generated in {processing_time_ms}ms, "
            f"sources={len(sources)}"
        )

        return ChatResponse(
            answer=answer,
            sources=sources,
            conversation_id=conversation_id,
            processing_time_ms=processing_time_ms,
        )

    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)

        # Determine error type
        error_message = str(e)
        if "timeout" in error_message.lower():
            raise HTTPException(
                status_code=504,
                detail={
                    "error": "Answer generation timed out, please retry",
                    "code": "TIMEOUT",
                },
            )
        elif "qdrant" in error_message.lower() or "vector" in error_message.lower():
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "Search service temporarily unavailable",
                    "code": "SERVER_ERROR",
                },
            )
        else:
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "An error occurred processing your request",
                    "code": "SERVER_ERROR",
                },
            )
