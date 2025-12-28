"""Pytest configuration and fixtures."""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def test_client():
    """Create a test client for the FastAPI app."""
    from app.main import app

    with TestClient(app) as client:
        yield client


@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response."""
    return {
        "choices": [
            {
                "message": {
                    "content": "This is a test answer based on the book content."
                }
            }
        ]
    }


@pytest.fixture
def mock_qdrant_results():
    """Mock Qdrant search results."""
    return [
        {
            "id": "chunk_001",
            "score": 0.95,
            "payload": {
                "chunk_id": "chunk_001",
                "chapter_id": "chapter-1",
                "section_title": "Introduction",
                "content": "This is test content from the book.",
                "chunk_index": 0,
                "word_count": 10,
            }
        }
    ]


@pytest.fixture
def sample_chat_request():
    """Sample chat request data."""
    return {
        "question": "What is machine learning?",
        "selected_text": None,
        "conversation_id": None
    }


@pytest.fixture
def sample_index_request():
    """Sample index request data."""
    return {
        "chapters": [
            {
                "chapter_id": "chapter-1",
                "title": "Introduction to AI",
                "content": "# Introduction\n\nThis chapter introduces artificial intelligence...",
                "sections": ["What is AI?", "History of AI"]
            }
        ],
        "force_reindex": False
    }
