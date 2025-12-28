# Chapter 4: Backend Architecture with FastAPI

## Learning Objectives

By the end of this chapter, you will be able to:
- Design a production-ready RAG backend architecture
- Implement REST APIs with FastAPI for RAG operations
- Structure services for embedding, retrieval, and generation
- Handle errors, validation, and logging properly
- Configure the backend for scalability and maintainability

---

## 4.1 Why FastAPI for RAG

FastAPI has emerged as the preferred framework for building RAG backends due to its unique combination of developer experience and production readiness. Built on modern Python features like type hints and async/await, FastAPI delivers both rapid development and high performance.

The automatic API documentation generated from type hints provides immediate visibility into endpoints, request formats, and response schemas. This is invaluable during development and for frontend integration. OpenAPI schemas are generated automatically, enabling client code generation in any language.

Performance is exceptional. FastAPI runs on Starlette and uses Pydantic for validation, achieving throughput comparable to Node.js and Go frameworks. The async support is native, not bolted on, making it natural to handle concurrent requests efficiently—essential for RAG systems that make multiple external API calls per request.

Dependency injection simplifies service management. Database connections, API clients, and configuration objects can be injected into route handlers, promoting clean architecture and testability.

Type safety catches errors early. Request validation happens automatically based on Pydantic models, returning clear error messages for invalid inputs. Response serialization is similarly automatic and validated.

## 4.2 RAG Backend Architecture

A well-structured RAG backend separates concerns into distinct layers: API routes, service layer, and data access layer.

The API layer handles HTTP concerns: request parsing, response formatting, authentication, and routing. Routes are thin, delegating business logic to services.

The service layer contains business logic. The RAG service orchestrates the retrieval and generation pipeline. The embedding service interfaces with embedding APIs. The vector service manages Qdrant operations. The LLM service handles generation requests.

The data access layer abstracts database and external service interactions. This isolation enables testing with mocks and switching implementations without affecting business logic.

```
┌─────────────────────────────────────────────┐
│                 API Layer                   │
│   /api/chat  /api/index  /health           │
└─────────────────────────────────────────────┘
                      │
┌─────────────────────────────────────────────┐
│              Service Layer                  │
│  RAGService  EmbeddingService  LLMService  │
└─────────────────────────────────────────────┘
                      │
┌─────────────────────────────────────────────┐
│            Data Access Layer                │
│    VectorService  DatabaseService           │
└─────────────────────────────────────────────┘
                      │
┌─────────────────────────────────────────────┐
│           External Services                 │
│     Qdrant    Gemini API    PostgreSQL     │
└─────────────────────────────────────────────┘
```

## 4.3 Project Structure

A clean project structure promotes maintainability as the codebase grows.

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py              # Application entry point
│   ├── config.py            # Configuration management
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── chat.py      # Chat endpoint
│   │   │   ├── index.py     # Indexing endpoint
│   │   │   └── health.py    # Health check
│   │   └── models/
│   │       ├── __init__.py
│   │       ├── chat.py      # Request/response models
│   │       └── index.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── rag_service.py
│   │   ├── embedding_service.py
│   │   ├── vector_service.py
│   │   └── llm_service.py
│   └── database/
│       ├── __init__.py
│       └── connection.py
├── scripts/
│   └── index_book.py
├── requirements.txt
├── .env.example
└── Dockerfile
```

## 4.4 Application Entry Point

The main.py file configures and starts the FastAPI application.

```python
# app/main.py
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.config import settings
from app.api.routes import chat, index, health
from app.database.connection import init_db

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info(f"Starting RAG Backend v{settings.VERSION}")
    await init_db()
    yield
    # Shutdown
    logger.info("Shutting down RAG Backend")

app = FastAPI(
    title="RAG Backend API",
    description="Retrieval-Augmented Generation API for book content",
    version=settings.VERSION,
    lifespan=lifespan
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, tags=["Health"])
app.include_router(chat.router, prefix="/api", tags=["Chat"])
app.include_router(index.router, prefix="/api", tags=["Index"])
```

The lifespan context manager handles startup and shutdown events, initializing database connections and cleaning up resources.

## 4.5 Configuration Management

Configuration should be loaded from environment variables with sensible defaults.

```python
# app/config.py
from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    # Application
    VERSION: str = "1.0.0"
    LOG_LEVEL: str = "INFO"

    # API Keys
    GEMINI_API_KEY: str

    # Qdrant
    QDRANT_URL: str
    QDRANT_API_KEY: str
    QDRANT_COLLECTION: str = "book_chunks"

    # Database
    DATABASE_URL: str

    # CORS
    CORS_ORIGINS: List[str] = ["http://localhost:3000"]

    # RAG Configuration
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 100
    RETRIEVAL_TOP_K: int = 5

    class Config:
        env_file = ".env"

settings = Settings()
```

Pydantic Settings automatically loads values from environment variables and validates types.

## 4.6 Request and Response Models

Pydantic models define clear contracts for API requests and responses.

```python
# app/api/models/chat.py
from pydantic import BaseModel, Field
from typing import Optional, List

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    selected_text: Optional[str] = Field(None, max_length=5000)
    conversation_id: Optional[str] = None

class Source(BaseModel):
    chapter_id: str
    section_title: str
    snippet: str
    relevance_score: float

class ChatResponse(BaseModel):
    answer: str
    sources: List[Source]
    conversation_id: str
    processing_time_ms: int
```

Field validators ensure input quality before processing.

## 4.7 Implementing the Chat Endpoint

The chat endpoint orchestrates the RAG pipeline.

```python
# app/api/routes/chat.py
import time
import uuid
import logging
from fastapi import APIRouter, HTTPException, Depends

from app.api.models.chat import ChatRequest, ChatResponse
from app.services.rag_service import RAGService

router = APIRouter()
logger = logging.getLogger(__name__)

def get_rag_service() -> RAGService:
    return RAGService()

@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    rag_service: RAGService = Depends(get_rag_service)
):
    start_time = time.time()
    conversation_id = request.conversation_id or str(uuid.uuid4())

    logger.info(f"Processing chat request: question='{request.question[:50]}...'")

    try:
        if request.selected_text:
            # Selected text mode - answer based on provided text
            answer, sources = rag_service.answer_from_text(
                question=request.question,
                text=request.selected_text
            )
        else:
            # Book-wide mode - retrieve and generate
            answer, sources = rag_service.answer(
                question=request.question
            )

        processing_time = int((time.time() - start_time) * 1000)

        return ChatResponse(
            answer=answer,
            sources=sources,
            conversation_id=conversation_id,
            processing_time_ms=processing_time
        )

    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": "An error occurred", "code": "SERVER_ERROR"}
        )
```

## 4.8 The RAG Service

The RAG service coordinates retrieval and generation.

```python
# app/services/rag_service.py
import logging
from typing import List, Tuple

from app.services.embedding_service import EmbeddingService
from app.services.vector_service import VectorService
from app.services.llm_service import LLMService
from app.api.models.chat import Source

logger = logging.getLogger(__name__)

class RAGService:
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.vector_service = VectorService()
        self.llm_service = LLMService()

    def answer(self, question: str) -> Tuple[str, List[Source]]:
        # Step 1: Embed the question
        query_embedding = self.embedding_service.embed_query(question)

        # Step 2: Retrieve relevant chunks
        chunks = self.vector_service.search(
            query_vector=query_embedding,
            limit=5
        )

        if not chunks:
            return "I don't have information about that in the book.", []

        # Step 3: Format context
        context = self._format_context(chunks)

        # Step 4: Generate answer
        answer = self.llm_service.generate(
            question=question,
            context=context
        )

        # Step 5: Format sources
        sources = [
            Source(
                chapter_id=chunk.payload.get("chapter_id", ""),
                section_title=chunk.payload.get("section_title", ""),
                snippet=chunk.payload.get("text", "")[:500],
                relevance_score=chunk.score
            )
            for chunk in chunks
        ]

        return answer, sources

    def answer_from_text(
        self, question: str, text: str
    ) -> Tuple[str, List[Source]]:
        # Selected text mode - no retrieval needed
        answer = self.llm_service.generate_from_selection(
            question=question,
            selected_text=text
        )
        return answer, []

    def _format_context(self, chunks) -> str:
        context_parts = []
        for chunk in chunks:
            chapter = chunk.payload.get("chapter_id", "unknown")
            section = chunk.payload.get("section_title", "")
            text = chunk.payload.get("text", "")
            context_parts.append(f"[{chapter} - {section}]\n{text}")
        return "\n\n---\n\n".join(context_parts)
```

## 4.9 Error Handling and Logging

Comprehensive error handling and logging are essential for production systems.

```python
# Custom exception handler
from fastapi import Request
from fastapi.responses import JSONResponse

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "detail": {
                "error": "An unexpected error occurred",
                "code": "INTERNAL_ERROR"
            }
        }
    )
```

Structured logging enables analysis and debugging:

```python
logger.info("Request processed", extra={
    "question_length": len(question),
    "chunks_retrieved": len(chunks),
    "processing_time_ms": processing_time
})
```

## 4.10 Health Checks

Health endpoints verify system readiness.

```python
# app/api/routes/health.py
from fastapi import APIRouter
from app.services.vector_service import VectorService
from app.config import settings

router = APIRouter()

@router.get("/health")
async def health_check():
    vector_service = VectorService()

    checks = {
        "qdrant": vector_service.check_health(),
        "database": await check_database(),
        "gemini": check_gemini_api()
    }

    status = "healthy" if all(checks.values()) else "unhealthy"

    return {
        "status": status,
        "services": checks,
        "version": settings.VERSION
    }
```

---

## Chapter Summary

This chapter presented a production-ready FastAPI backend architecture for RAG systems. We structured the application into API, service, and data access layers for clean separation of concerns. Configuration management uses environment variables with Pydantic validation. Request/response models define clear API contracts. The RAG service orchestrates embedding, retrieval, and generation. Error handling and logging support production operations. Health checks verify system readiness. This architecture provides a solid foundation for building scalable, maintainable RAG applications.

---

## Review Questions

1. Why is FastAPI well-suited for RAG backends?
2. Describe the three-layer architecture and the responsibility of each layer.
3. How does Pydantic Settings simplify configuration management?
4. What is the purpose of the lifespan context manager?
5. How should errors be handled to provide useful feedback without exposing internals?

---

## Hands-On Exercises

**Exercise 4.1**: Create a new FastAPI project with the structure described in this chapter. Implement a basic health endpoint that returns the application version.

**Exercise 4.2**: Implement request validation using Pydantic models. Test with valid and invalid inputs and observe the error responses.

**Exercise 4.3**: Add structured logging to track request processing times and result statistics. View the logs during operation.
