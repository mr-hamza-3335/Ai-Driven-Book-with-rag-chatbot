"""Pydantic models for chat functionality."""

from typing import Optional, List

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Incoming chat request from frontend."""

    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The question to ask about the book",
    )
    selected_text: Optional[str] = Field(
        None,
        min_length=10,
        max_length=5000,
        description="Optional selected text to answer from (bypasses RAG)",
    )
    conversation_id: Optional[str] = Field(
        None,
        description="Optional conversation ID for context tracking",
    )


class Source(BaseModel):
    """Source reference for an answer."""

    chapter_id: str = Field(..., description="Chapter identifier")
    section_title: str = Field(..., description="Section title")
    snippet: str = Field(
        ...,
        max_length=500,
        description="Relevant text snippet",
    )
    relevance_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Relevance score from 0 to 1",
    )


class ChatResponse(BaseModel):
    """Response to chat request."""

    answer: str = Field(..., description="Generated answer")
    sources: List[Source] = Field(
        default_factory=list,
        description="Source references for the answer",
    )
    conversation_id: str = Field(..., description="Conversation ID")
    processing_time_ms: int = Field(
        ...,
        ge=0,
        description="Processing time in milliseconds",
    )


class ChatError(BaseModel):
    """Error response for chat endpoint."""

    error: str = Field(..., description="Error message")
    code: str = Field(
        ...,
        description="Error code: VALIDATION_ERROR, RATE_LIMIT, SERVER_ERROR, TIMEOUT",
    )
    retry_after: Optional[int] = Field(
        None,
        description="Seconds to wait before retrying (for rate limit errors)",
    )
