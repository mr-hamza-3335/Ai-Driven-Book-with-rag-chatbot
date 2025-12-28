# Research: Backend API

**Feature**: 003-backend-api
**Date**: 2025-12-27
**Status**: Complete

## Research Tasks

### 1. Vector Database Selection

**Decision**: Qdrant Cloud (Free Tier)

**Rationale**:
- Constitution mandates Qdrant Cloud
- Free tier provides 1GB storage (sufficient for book)
- Native Python client with async support
- Excellent documentation and community support
- Supports filtering and payload storage

**Alternatives Considered**:
- Pinecone: Higher free tier limits but not mandated
- Weaviate: More complex setup
- Chroma: Good for local but no managed cloud option

### 2. LLM Provider

**Decision**: OpenAI API (gpt-4o-mini for answers, text-embedding-3-small for embeddings)

**Rationale**:
- Constitution allows OpenAI or ChatKit SDK
- OpenAI has better documentation and stability
- gpt-4o-mini balances cost and quality
- text-embedding-3-small is cost-effective for book content

**Alternatives Considered**:
- Anthropic Claude: Excellent but different SDK pattern
- Azure OpenAI: Additional complexity
- ChatKit SDK: Less documented

### 3. Database for Rate Limiting

**Decision**: Neon Serverless Postgres

**Rationale**:
- Constitution mandates Neon
- Serverless scales to zero (cost-effective)
- Standard PostgreSQL with connection pooling
- Simple rate limiting table schema

**Alternatives Considered**:
- Redis: Faster but adds another service
- In-memory: Doesn't persist across restarts
- SQLite: Not suitable for serverless

### 4. Chunking Strategy

**Decision**: Recursive character splitting with 500 token chunks, 50 token overlap

**Rationale**:
- Book content is markdown with headers
- 500 tokens provides good context without exceeding limits
- Overlap preserves context across chunk boundaries
- LangChain's RecursiveCharacterTextSplitter is well-tested

**Alternatives Considered**:
- Sentence splitting: Loses paragraph context
- Semantic chunking: More complex, marginal benefit for structured book
- Fixed character: Ignores markdown structure

### 5. Zero Hallucination Strategy

**Decision**: Strict context-only prompting with no-answer fallback

**Rationale**:
- System prompt explicitly forbids generating information not in context
- If no relevant chunks retrieved, return standard message
- Response validation checks for grounding
- Source references required for every answer

**Prompt Template**:
```
You are a helpful assistant that answers questions about an AI/ML book.
You MUST only use information from the provided context.
If the context does not contain relevant information, respond with exactly:
"I don't have information about that in the book."
Do NOT make up or infer information not explicitly stated in the context.
Always cite the chapter and section for your answer.
```

**Alternatives Considered**:
- RAG + filtering: Still requires prompt engineering
- Confidence thresholds only: LLMs can be confidently wrong

### 6. API Framework

**Decision**: FastAPI with Pydantic v2

**Rationale**:
- Constitution mandates FastAPI
- Pydantic v2 has better performance
- Automatic OpenAPI documentation
- Native async support for I/O-bound operations

**Alternatives Considered**:
- Flask: No async, more boilerplate
- Django: Overkill for API-only project
- Starlette: FastAPI adds useful abstractions

### 7. Deployment Target

**Decision**: Docker container on Railway/Render

**Rationale**:
- Simple Docker deployment
- Free tiers available for hackathon
- Automatic HTTPS and domain
- Environment variable support

**Alternatives Considered**:
- AWS Lambda: Cold start issues for Python
- Fly.io: Good but slightly more complex
- Heroku: Deprecated free tier

## Dependencies Summary

```
fastapi>=0.109.0
uvicorn>=0.27.0
pydantic>=2.5.0
openai>=1.10.0
qdrant-client>=1.7.0
sqlalchemy>=2.0.0
psycopg2-binary>=2.9.0
python-dotenv>=1.0.0
httpx>=0.26.0
langchain-text-splitters>=0.0.1
```

## Environment Variables

```bash
# Required
OPENAI_API_KEY=sk-...
QDRANT_URL=https://xxx.qdrant.cloud
QDRANT_API_KEY=...
DATABASE_URL=postgresql://...

# Optional
CORS_ORIGINS=http://localhost:3000,https://xxx.github.io
RATE_LIMIT_PER_MINUTE=60
LOG_LEVEL=INFO
```

## Resolved Clarifications

All technical decisions have been made. No outstanding NEEDS CLARIFICATION items remain.
