"""Payload size limiting middleware for FastAPI."""

import logging

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from app.config import get_settings

logger = logging.getLogger(__name__)

# Default max payload size: 1MB
DEFAULT_MAX_PAYLOAD_SIZE = 1 * 1024 * 1024  # 1MB in bytes


class PayloadLimitMiddleware(BaseHTTPMiddleware):
    """Middleware for limiting request payload size."""

    # Paths to exclude from payload limiting (e.g., health checks)
    EXCLUDED_PATHS = {"/health", "/docs", "/openapi.json", "/redoc"}

    def __init__(self, app, max_size: int = DEFAULT_MAX_PAYLOAD_SIZE):
        """Initialize middleware.

        Args:
            app: FastAPI application
            max_size: Maximum payload size in bytes (default 1MB)
        """
        super().__init__(app)
        self.max_size = max_size

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        """Process request and check payload size.

        Args:
            request: Incoming request
            call_next: Next middleware/endpoint

        Returns:
            Response or 413 error if payload too large
        """
        # Skip for excluded paths
        if request.url.path in self.EXCLUDED_PATHS:
            return await call_next(request)

        # Skip for GET, HEAD, OPTIONS requests (no body)
        if request.method in {"GET", "HEAD", "OPTIONS"}:
            return await call_next(request)

        # Check Content-Length header first (if present)
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                length = int(content_length)
                if length > self.max_size:
                    logger.warning(
                        f"Payload too large: {length} bytes > {self.max_size} bytes limit"
                    )
                    return self._create_413_response(length)
            except ValueError:
                pass  # Invalid content-length, let request proceed

        # For chunked transfers or missing Content-Length, we'll rely on
        # application-level validation after body parsing
        return await call_next(request)

    def _create_413_response(self, actual_size: int) -> JSONResponse:
        """Create 413 Payload Too Large response.

        Args:
            actual_size: Actual payload size in bytes

        Returns:
            JSONResponse with 413 status
        """
        max_size_mb = self.max_size / (1024 * 1024)
        actual_size_mb = actual_size / (1024 * 1024)

        return JSONResponse(
            status_code=413,
            content={
                "error": "Request payload too large",
                "code": "PAYLOAD_TOO_LARGE",
                "max_size_bytes": self.max_size,
                "max_size_mb": round(max_size_mb, 2),
                "actual_size_bytes": actual_size,
                "actual_size_mb": round(actual_size_mb, 2),
                "message": f"Maximum payload size is {max_size_mb:.1f}MB. Your request was {actual_size_mb:.2f}MB.",
            },
        )
