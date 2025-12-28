"""FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.api.routes import chat, index, health
from app.middleware.rate_limit import RateLimitMiddleware
from app.middleware.payload_limit import PayloadLimitMiddleware
from app.database import init_db

# Configure structured logging
def setup_logging():
    """Configure logging with settings."""
    settings = get_settings()
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup and shutdown."""
    settings = get_settings()
    logger.info(f"Starting Backend API v{settings.app_version}")
    logger.info(f"Log level: {settings.log_level}")

    # Startup: Initialize connections
    logger.info("Initializing services...")

    # Initialize database tables
    try:
        init_db()
        logger.info("Database tables initialized")
    except Exception as e:
        logger.warning(f"Database initialization skipped: {e}")

    yield

    # Shutdown: Cleanup
    logger.info("Shutting down Backend API...")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="Book Chatbot API",
        description="RAG-based question answering API for the AI/ML book",
        version=settings.app_version,
        lifespan=lifespan,
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add rate limiting middleware
    app.add_middleware(RateLimitMiddleware)

    # Add payload size limiting middleware
    app.add_middleware(PayloadLimitMiddleware, max_size=settings.max_payload_size)

    # Include routers
    app.include_router(chat.router)
    app.include_router(index.router)
    app.include_router(health.router)

    return app


# Create the app instance
app = create_app()
