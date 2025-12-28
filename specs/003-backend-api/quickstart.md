# Quickstart: Backend API

**Feature**: 003-backend-api
**Date**: 2025-12-27

## Prerequisites

- Python 3.11+
- Docker and Docker Compose (recommended)
- OpenAI API key
- Qdrant Cloud account (free tier)
- Neon Postgres database (free tier)

## Cloud Service Setup

### 1. Qdrant Cloud

1. Go to https://cloud.qdrant.io/
2. Create a free cluster
3. Copy the cluster URL and API key
4. Save for environment variables

### 2. Neon Postgres

1. Go to https://neon.tech/
2. Create a free project
3. Copy the connection string
4. Save for environment variables

### 3. OpenAI API

1. Go to https://platform.openai.com/
2. Create an API key
3. Save for environment variables

## Setup

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
# Required
OPENAI_API_KEY=sk-your-key-here
QDRANT_URL=https://your-cluster.qdrant.cloud
QDRANT_API_KEY=your-qdrant-key
DATABASE_URL=postgresql://user:pass@host/db?sslmode=require

# Optional
CORS_ORIGINS=http://localhost:3000
RATE_LIMIT_PER_MINUTE=60
LOG_LEVEL=INFO
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

# Run database migrations
python -m app.database

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

## Testing the API

### Health Check

```bash
curl http://localhost:8000/health
```

Expected response:
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

### Chat Query

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is machine learning?"}'
```

### Index Book Content

```bash
curl -X POST http://localhost:8000/api/index \
  -H "Content-Type: application/json" \
  -d '{
    "chapters": [{
      "chapter_id": "chapter-1-intro-to-ai",
      "title": "Introduction to AI",
      "content": "# Introduction\n\nMachine learning is..."
    }]
  }'
```

## Project Commands

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=app

# Format code
black app tests

# Lint code
ruff check app tests

# Type check
mypy app
```

## Index Book Content

Use the CLI script to index the full book:

```bash
python scripts/index_book.py --book-dir ../book/docs
```

Options:
- `--book-dir`: Path to Docusaurus docs directory
- `--force`: Force reindex (delete existing)
- `--dry-run`: Preview without indexing

## Verification Checklist

- [ ] `/health` returns healthy status
- [ ] All services show "ok" status
- [ ] `/api/chat` returns answers with sources
- [ ] Rate limiting works (429 after 60 requests/min)
- [ ] CORS allows book origin
- [ ] OpenAPI docs available at `/docs`

## Troubleshooting

### Connection Errors

```bash
# Check Qdrant connection
curl $QDRANT_URL/collections -H "api-key: $QDRANT_API_KEY"

# Check Postgres connection
psql $DATABASE_URL -c "SELECT 1"
```

### API Errors

- 400: Check request body format
- 429: Rate limit exceeded, wait and retry
- 500: Check logs for stack trace
- 503: External service unavailable

### Docker Issues

```bash
# Rebuild from scratch
docker-compose down -v
docker-compose up --build

# View logs
docker-compose logs -f api
```

## Environment Variables Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| OPENAI_API_KEY | Yes | - | OpenAI API key |
| QDRANT_URL | Yes | - | Qdrant Cloud URL |
| QDRANT_API_KEY | Yes | - | Qdrant API key |
| DATABASE_URL | Yes | - | Postgres connection string |
| CORS_ORIGINS | No | * | Allowed CORS origins |
| RATE_LIMIT_PER_MINUTE | No | 60 | Rate limit per IP |
| LOG_LEVEL | No | INFO | Logging level |
