# AI-Driven Textbook with Embedded RAG Chatbot

An interactive textbook on **Physical AI & Humanoid Robotics** featuring an embedded RAG (Retrieval-Augmented Generation) chatbot powered by Google Gemini and Qdrant vector database.

## Architecture

```
+------------------+     +------------------+     +------------------+
|                  |     |                  |     |                  |
|   Docusaurus     |---->|   FastAPI        |---->|   Qdrant Cloud   |
|   Frontend       |     |   Backend        |     |   Vector DB      |
|                  |     |                  |     |                  |
+------------------+     +------------------+     +------------------+
        |                        |                        |
        v                        v                        v
+------------------+     +------------------+     +------------------+
|  React Chatbot   |     | Gemini 2.5 Flash |     |  768-dim Vectors |
|  Component       |     | LLM Generation   |     |  95 Book Chunks  |
+------------------+     +------------------+     +------------------+
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| Frontend | Docusaurus 3.x, React, TypeScript |
| Backend | FastAPI, Python 3.11+ |
| Vector DB | Qdrant Cloud |
| Embeddings | Gemini text-embedding-004 (768 dims) |
| LLM | Gemini 2.5 Flash |
| Deployment | GitHub Pages (frontend), Local (backend) |

## Features

- **6 Comprehensive Chapters** on Physical AI & Humanoid Robotics
- **Embedded RAG Chatbot** on every page
- **Selected Text Mode** - Highlight text to ask questions about specific content
- **Zero-Hallucination Design** - Answers only from book content
- **Source Citations** - Every answer includes chapter and section references
- **Dark/Light Mode** - Full theme support

## Project Structure

```
hackathonnnnn/
├── books/                    # Source markdown chapters
│   ├── 01-introduction.md
│   ├── 02-ros2-basics.md
│   ├── 03-gazebo-simulation.md
│   ├── 04-nvidia-isaac.md
│   ├── 05-vla.md
│   └── 06-capstone.md
├── backend/                  # FastAPI RAG backend
│   ├── app/
│   │   ├── api/routes/       # API endpoints
│   │   ├── services/         # RAG, embedding, vector services
│   │   └── main.py
│   └── scripts/
│       └── index_books.py    # Book indexing script
├── docs/                     # Docusaurus frontend
│   ├── docs/                 # Book content
│   ├── src/components/       # React components
│   │   └── Chatbot/          # RAG chatbot widget
│   └── docusaurus.config.ts
└── .github/workflows/        # GitHub Actions
    └── deploy-docs.yml       # Auto-deploy to GitHub Pages
```

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 20+
- Gemini API Key
- Qdrant Cloud account

### 1. Clone and Setup

```bash
git clone https://github.com/mr-hamza-3335/Ai-Driven-Book-with-rag-chatbot.git
cd hackathonnnnn
```

### 2. Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys:
# GEMINI_API_KEY=your-key
# QDRANT_URL=your-qdrant-url
# QDRANT_API_KEY=your-qdrant-key

# Run backend
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 3. Index Book Content

```bash
python scripts/index_books.py
```

### 4. Frontend Setup

```bash
cd docs
npm install
npm start
```

Visit http://localhost:3000 to see the book with embedded chatbot.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/chat` | POST | Chat with RAG |
| `/api/index` | POST | Index book chapters |

### Chat Request

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is Physical AI?"}'
```

### Selected Text Mode

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Explain this concept",
    "selected_text": "VLAs operate on continuous action spaces..."
  }'
```

## Book Chapters

1. **Introduction to Physical AI** - What is Physical AI, history, current state
2. **ROS 2 Fundamentals** - Nodes, topics, services, actions, TF2
3. **Gazebo Simulation** - SDF models, sensors, physics, environments
4. **NVIDIA Isaac** - Isaac Sim, Isaac ROS, Omniverse integration
5. **Vision-Language-Action Models** - VLA architecture, training, deployment
6. **Capstone Project** - Build a complete humanoid robot system

## Deployment

### GitHub Pages (Frontend)

Push to `main` branch triggers automatic deployment:
```bash
git push origin main
```

### Backend Deployment

For production, deploy to a cloud provider with:
- Environment variables configured
- HTTPS enabled
- CORS configured for your frontend domain

## Environment Variables

| Variable | Description |
|----------|-------------|
| `GEMINI_API_KEY` | Google Gemini API key |
| `QDRANT_URL` | Qdrant Cloud URL |
| `QDRANT_API_KEY` | Qdrant API key |
| `LOG_LEVEL` | Logging level (INFO/DEBUG) |

## License

MIT

## Acknowledgments

- Google Gemini for embeddings and LLM
- Qdrant for vector database
- Docusaurus for documentation framework
- The robotics and AI community
