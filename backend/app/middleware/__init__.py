"""Middleware package for FastAPI application."""

from app.middleware.cors import configure_cors

__all__ = ["configure_cors"]
