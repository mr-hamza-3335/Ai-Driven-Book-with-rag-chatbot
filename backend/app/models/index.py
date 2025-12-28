"""Pydantic models for indexing functionality."""

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class IndexStatus(str, Enum):
    """Status of an indexing job."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class ChapterContent(BaseModel):
    """Single chapter for indexing."""

    chapter_id: str = Field(
        ...,
        pattern=r"^[a-z0-9-]+$",
        description="Unique chapter identifier (alphanumeric with hyphens)",
    )
    title: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Chapter title",
    )
    content: str = Field(
        ...,
        min_length=100,
        max_length=100000,
        description="Markdown content of the chapter",
    )
    sections: Optional[List[str]] = Field(
        None,
        description="Optional list of section titles",
    )


class IndexRequest(BaseModel):
    """Request to index book content."""

    chapters: List[ChapterContent] = Field(
        ...,
        min_length=1,
        max_length=20,
        description="List of chapters to index",
    )
    force_reindex: bool = Field(
        False,
        description="If true, delete existing chunks before indexing",
    )


class IndexResponse(BaseModel):
    """Response to index request."""

    job_id: str = Field(..., description="Unique job identifier")
    status: IndexStatus = Field(..., description="Job status")
    chapters_indexed: int = Field(
        ...,
        ge=0,
        description="Number of chapters indexed",
    )
    chunks_created: int = Field(
        ...,
        ge=0,
        description="Total chunks created",
    )


class IndexStatusResponse(BaseModel):
    """Status of ongoing indexing job."""

    job_id: str = Field(..., description="Job identifier")
    status: IndexStatus = Field(..., description="Current status")
    progress_percent: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Progress percentage",
    )
    chapters_processed: int = Field(
        ...,
        ge=0,
        description="Chapters processed so far",
    )
    total_chapters: int = Field(
        ...,
        ge=0,
        description="Total chapters to process",
    )
    error_message: Optional[str] = Field(
        None,
        description="Error message if status is FAILED",
    )
