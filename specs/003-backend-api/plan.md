# Implementation Plan: Backend API

**Branch**: `003-backend-api` | **Date**: 2025-12-27 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/003-backend-api/spec.md`

## Summary

Build a FastAPI backend that provides RAG-based question answering for the book chatbot. The API uses Qdrant Cloud for vector similarity search, Neon Postgres for metadata storage, and OpenAI API for embeddings and completions. Implements zero hallucination by grounding all answers in retrieved book content.

## Technical Context

**Language/Version**: Python 3.11+
**Primary Dependencies**: FastAPI 0.109+, Qdrant Client 1.7+, OpenAI SDK 1.10+, SQLAlchemy 2.0+, Pydantic 2.0+
**Storage**: Qdrant Cloud (vectors), Neon Serverless Postgres (metadata, rate limiting)
**Testing**: pytest, pytest-asyncio, httpx for API testing
**Target Platform**: Linux server (Docker), deployable to Railway/Render/Fly.io
**Project Type**: Single Python project (API server)
**Performance Goals**: < 5 second p90 response time, 100 concurrent users
**Constraints**: Rate limit 60 req/min/IP, max 5000 char selected text
**Scale/Scope**: ~1000 book chunks indexed, ~100 concurrent users

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Evidence |
|-----------|--------|----------|
| I. Spec-Driven Development | PASS | Spec complete at spec.md, plan follows spec |
| II. Zero Hallucination | PASS | RAG retrieval is mandatory; answer generation uses only retrieved chunks (FR-003); empty retrieval returns "I don't have information" |
| III. Context-Only Answers | PASS | Selected-text mode bypasses vector search; uses only provided text (FR-011) |
| IV. Clean Architecture | PASS | Layered architecture (API → Services → Repositories); clear separation of concerns |
| V. Secure by Default | PASS | All credentials via environment variables (.env); .env.example provided; no hardcoded secrets (NFR-004) |
| VI. Small Testable Changes | PASS | Each endpoint independently testable; services have unit tests |

**Gate Result**: PASSED - No violations

## Project Structure

### Documentation (this feature)

```text
specs/003-backend-api/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # OpenAPI spec
└── tasks.md             # Phase 2 output (created by /sp.tasks)
```

### Source Code (repository root)

```text
backend/
├── app/
│   ├── __init__.py
│   ├── main.py                    # FastAPI app entry point
│   ├── config.py                  # Environment configuration
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── chat.py            # POST /api/chat
│   │   │   ├── index.py           # POST /api/index
│   │   │   └── health.py          # GET /health
│   │   └── dependencies.py        # Dependency injection
│   ├── models/
│   │   ├── __init__.py
│   │   ├── chat.py                # Pydantic models for chat
│   │   └── index.py               # Pydantic models for indexing
│   ├── services/
│   │   ├── __init__.py
│   │   ├── rag_service.py         # RAG orchestration
│   │   ├── embedding_service.py   # OpenAI embeddings
│   │   ├── llm_service.py         # OpenAI completions
│   │   ├── vector_service.py      # Qdrant operations
│   │   └── chunking_service.py    # Text chunking
│   ├── repositories/
│   │   ├── __init__.py
│   │   └── rate_limit_repo.py     # Rate limiting with Postgres
│   └── middleware/
│       ├── __init__.py
│       ├── cors.py                # CORS configuration
│       └── rate_limit.py          # Rate limiting middleware
├── tests/
│   ├── __init__.py
│   ├── conftest.py                # Pytest fixtures
│   ├── test_chat.py               # Chat endpoint tests
│   ├── test_index.py              # Index endpoint tests
│   ├── test_rag_service.py        # RAG service unit tests
│   └── test_health.py             # Health check tests
├── scripts/
│   └── index_book.py              # CLI script to index book content
├── requirements.txt               # Pinned dependencies
├── requirements-dev.txt           # Dev dependencies
├── Dockerfile                     # Container build
├── docker-compose.yml             # Local dev environment
├── .env.example                   # Environment template
└── README.md                      # Setup instructions
```

**Structure Decision**: Single Python project with layered architecture. API routes call services; services orchestrate business logic; repositories handle data access.

## Complexity Tracking

No violations requiring justification.

## Implementation Approach

### Phase 1: Project Setup
1. Initialize FastAPI project structure
2. Configure environment variables and .env.example
3. Set up Docker and docker-compose for local dev
4. Configure pytest and test infrastructure

### Phase 2: Core Services
1. Implement embedding_service with OpenAI
2. Implement vector_service with Qdrant client
3. Implement chunking_service for markdown
4. Implement llm_service for answer generation

### Phase 3: RAG Pipeline
1. Implement rag_service orchestrating retrieval and generation
2. Add zero-hallucination guardrails (context-only prompts)
3. Implement selected-text mode (bypass retrieval)
4. Add source reference extraction

### Phase 4: API Endpoints
1. Implement POST /api/chat with Pydantic validation
2. Implement POST /api/index for content ingestion
3. Implement GET /health with dependency checks
4. Add CORS middleware for frontend

### Phase 5: Rate Limiting & Production
1. Implement rate_limit_repo with Postgres
2. Add rate_limit middleware (60 req/min/IP)
3. Configure logging and error handling
4. Create deployment configuration
