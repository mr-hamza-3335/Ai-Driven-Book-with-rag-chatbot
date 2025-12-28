# Data Model: Backend API

**Feature**: 003-backend-api
**Date**: 2025-12-27

## Overview

This feature uses Qdrant Cloud for vector storage and Neon Postgres for rate limiting. The data models are defined using Pydantic for API and SQLAlchemy for database.

## Pydantic Models (API Layer)

### Chat Models (`app/models/chat.py`)

```python
from pydantic import BaseModel, Field
from typing import Optional, List

class ChatRequest(BaseModel):
    """Incoming chat request from frontend."""
    question: str = Field(..., min_length=1, max_length=2000)
    selected_text: Optional[str] = Field(None, min_length=10, max_length=5000)
    conversation_id: Optional[str] = None

class Source(BaseModel):
    """Source reference for an answer."""
    chapter_id: str
    section_title: str
    snippet: str = Field(..., max_length=500)
    relevance_score: float = Field(..., ge=0.0, le=1.0)

class ChatResponse(BaseModel):
    """Response to chat request."""
    answer: str
    sources: List[Source]
    conversation_id: str
    processing_time_ms: int

class ChatError(BaseModel):
    """Error response."""
    error: str
    code: str  # VALIDATION_ERROR, RATE_LIMIT, SERVER_ERROR, TIMEOUT
    retry_after: Optional[int] = None  # Seconds
```

### Index Models (`app/models/index.py`)

```python
from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum

class IndexStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

class ChapterContent(BaseModel):
    """Single chapter for indexing."""
    chapter_id: str
    title: str
    content: str  # Markdown content
    sections: Optional[List[str]] = None

class IndexRequest(BaseModel):
    """Request to index book content."""
    chapters: List[ChapterContent]
    force_reindex: bool = False

class IndexResponse(BaseModel):
    """Response to index request."""
    job_id: str
    status: IndexStatus
    chapters_indexed: int
    chunks_created: int

class IndexStatusResponse(BaseModel):
    """Status of ongoing indexing job."""
    job_id: str
    status: IndexStatus
    progress_percent: float
    chapters_processed: int
    total_chapters: int
    error_message: Optional[str] = None
```

### Health Models

```python
from pydantic import BaseModel
from typing import Dict

class ServiceStatus(BaseModel):
    """Status of a single service."""
    status: str  # "ok" or "error"
    latency_ms: Optional[int] = None
    error: Optional[str] = None

class HealthResponse(BaseModel):
    """Health check response."""
    status: str  # "healthy" or "unhealthy"
    services: Dict[str, ServiceStatus]
    version: str
```

## Qdrant Vector Schema

### Collection: `book_chunks`

```python
from qdrant_client.models import VectorParams, Distance

# Collection configuration
collection_config = {
    "vectors": VectorParams(
        size=1536,  # text-embedding-3-small dimension
        distance=Distance.COSINE
    )
}

# Point payload schema
point_payload = {
    "chunk_id": str,           # Unique chunk identifier
    "chapter_id": str,         # e.g., "chapter-1-intro-to-ai"
    "section_title": str,      # e.g., "What is Machine Learning?"
    "content": str,            # The actual text chunk
    "chunk_index": int,        # Order within chapter
    "word_count": int,         # For reading time estimation
    "created_at": str,         # ISO timestamp
}
```

### Example Qdrant Point

```json
{
  "id": "chunk_001_ch1_sec2",
  "vector": [0.023, -0.041, ...],  // 1536 dimensions
  "payload": {
    "chunk_id": "chunk_001_ch1_sec2",
    "chapter_id": "chapter-1-intro-to-ai",
    "section_title": "Types of Machine Learning",
    "content": "Supervised learning is a type of machine learning where the algorithm learns from labeled training data. The training data consists of input-output pairs...",
    "chunk_index": 2,
    "word_count": 127,
    "created_at": "2025-12-27T10:00:00Z"
  }
}
```

## PostgreSQL Schema (Neon)

### Table: `rate_limits`

```sql
CREATE TABLE rate_limits (
    id SERIAL PRIMARY KEY,
    client_ip VARCHAR(45) NOT NULL,  -- IPv6 compatible
    request_count INT NOT NULL DEFAULT 1,
    window_start TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(client_ip)
);

CREATE INDEX idx_rate_limits_ip ON rate_limits(client_ip);
CREATE INDEX idx_rate_limits_window ON rate_limits(window_start);
```

### SQLAlchemy Model

```python
from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.sql import func
from app.database import Base

class RateLimit(Base):
    __tablename__ = "rate_limits"

    id = Column(Integer, primary_key=True, index=True)
    client_ip = Column(String(45), unique=True, nullable=False, index=True)
    request_count = Column(Integer, default=1, nullable=False)
    window_start = Column(DateTime, server_default=func.now(), nullable=False)
```

## Internal Service Models

### RAG Context

```python
from dataclasses import dataclass
from typing import List

@dataclass
class RetrievedChunk:
    """Single chunk retrieved from vector search."""
    chunk_id: str
    chapter_id: str
    section_title: str
    content: str
    relevance_score: float

@dataclass
class RAGContext:
    """Context for answer generation."""
    question: str
    chunks: List[RetrievedChunk]
    mode: str  # "book_wide" or "selected_text"
    selected_text: Optional[str] = None
```

## Validation Rules

### Chat Request
- question: 1-2000 characters, required
- selected_text: 10-5000 characters if provided
- conversation_id: UUID format if provided

### Index Request
- chapters: 1-20 chapters per request
- content: 100-100000 characters per chapter
- chapter_id: alphanumeric with hyphens

### Rate Limiting
- Window: 60 seconds
- Limit: 60 requests per window per IP
- Reset: Window resets after 60 seconds of inactivity

## Data Flow

```
1. Chat Request (Frontend)
   ↓
2. Rate Limit Check (Postgres)
   ↓
3. Embedding Generation (OpenAI)
   ↓
4. Vector Search (Qdrant)
   ↓
5. Context Assembly (RAGContext)
   ↓
6. Answer Generation (OpenAI)
   ↓
7. Source Extraction
   ↓
8. Chat Response (Frontend)
```

## Notes

- Qdrant points use composite IDs for easy debugging
- Rate limit cleanup job runs hourly (delete windows > 1 hour old)
- Embeddings cached in memory during indexing batch
- Selected-text mode skips steps 3-4 (uses text directly)
