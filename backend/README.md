# Backend API

RAG-based question answering API for the AI/ML Book Chatbot.

## Overview

This FastAPI backend provides:
- **Chat Queries**: Answer questions using RAG (Retrieval-Augmented Generation)
- **Content Indexing**: Index book content into Qdrant vector store
- **Selected-Text Mode**: Answer questions based on user-selected text
- **Health Monitoring**: Check status of all dependencies
- **Rate Limiting**: Protect against abuse (60 req/min/IP)

## Tech Stack

- **Framework**: FastAPI 0.109+ with Pydantic 2.0+
- **Vector Store**: Qdrant Cloud
- **Database**: Neon Serverless Postgres
- **LLM**: OpenAI API (gpt-4o-mini, text-embedding-3-small)
- **Python**: 3.11+

## Quick Start

### 1. Clone and Navigate

```bash
cd hackathonnnnn/backend
```

### 2. Create Environment File

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```bash
OPENAI_API_KEY=sk-your-key-here
QDRANT_URL=https://your-cluster.qdrant.cloud
QDRANT_API_KEY=your-qdrant-key
DATABASE_URL=postgresql://user:pass@host/db?sslmode=require
```

### 3. Start with Docker (Recommended)

```bash
docker-compose up --build
```

API available at `http://localhost:8000`

### 4. Alternative: Local Python

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Start server
uvicorn app.main:app --reload
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/chat` | POST | Process chat question |
| `/api/index` | POST | Index book content |
| `/api/index/{job_id}` | GET | Check indexing status |
| `/health` | GET | Health check |
| `/docs` | GET | OpenAPI documentation |

## Usage Examples

### Chat Query

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is machine learning?"}'
```

Response:
```json
{
  "answer": "Machine learning is...",
  "sources": [
    {
      "chapter_id": "chapter-1",
      "section_title": "Introduction",
      "snippet": "...",
      "relevance_score": 0.95
    }
  ],
  "conversation_id": "...",
  "processing_time_ms": 1234
}
```

### Selected-Text Query

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Explain this concept",
    "selected_text": "Neural networks are computational models..."
  }'
```

### Index Content

```bash
curl -X POST http://localhost:8000/api/index \
  -H "Content-Type: application/json" \
  -d '{
    "chapters": [{
      "chapter_id": "chapter-1",
      "title": "Introduction",
      "content": "# Introduction\n\nThis chapter..."
    }]
  }'
```

### Health Check

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "services": {
    "qdrant": {"status": "ok", "latency_ms": 50},
    "database": {"status": "ok", "latency_ms": 20},
    "openai": {"status": "ok", "latency_ms": 100}
  },
  "version": "1.0.0"
}
```

## Index Book Content (CLI)

```bash
python scripts/index_book.py --book-dir ../book/docs
```

Options:
- `--book-dir`: Path to Docusaurus docs directory
- `--force`: Force reindex (delete existing)
- `--dry-run`: Preview without indexing
- `--api-url`: API URL (default: http://localhost:8000)

## Development

### Run Tests

```bash
pip install -r requirements-dev.txt
pytest
```

### Code Quality

```bash
# Format
black app tests

# Lint
ruff check app tests

# Type check
mypy app
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| OPENAI_API_KEY | Yes | - | OpenAI API key |
| QDRANT_URL | Yes | - | Qdrant Cloud URL |
| QDRANT_API_KEY | Yes | - | Qdrant API key |
| DATABASE_URL | Yes | - | Postgres connection string |
| CORS_ORIGINS | No | * | Allowed CORS origins |
| RATE_LIMIT_PER_MINUTE | No | 60 | Rate limit per IP |
| LOG_LEVEL | No | INFO | Logging level |

## Architecture

```
app/
├── main.py              # FastAPI entry point
├── config.py            # Environment configuration
├── database.py          # SQLAlchemy setup
├── api/
│   ├── routes/          # API endpoints
│   └── dependencies.py  # Dependency injection
├── models/              # Pydantic models
├── services/            # Business logic
├── repositories/        # Data access
└── middleware/          # CORS, rate limiting
```

## Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| VALIDATION_ERROR | 400 | Invalid request |
| RATE_LIMIT | 429 | Too many requests |
| SERVER_ERROR | 500 | Internal error |
| TIMEOUT | 504 | Request timeout |

## License

MIT
