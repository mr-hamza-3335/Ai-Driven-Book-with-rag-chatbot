"""Rate limiting middleware for FastAPI."""

import logging

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from app.database import get_session_local
from app.repositories.rate_limit_repo import RateLimitRepository
from app.config import get_settings

logger = logging.getLogger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware for rate limiting requests per IP."""

    # Paths to exclude from rate limiting
    EXCLUDED_PATHS = {"/health", "/docs", "/openapi.json", "/redoc"}

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        """Process request and apply rate limiting.

        Args:
            request: Incoming request
            call_next: Next middleware/endpoint

        Returns:
            Response
        """
        # Skip rate limiting for excluded paths
        if request.url.path in self.EXCLUDED_PATHS:
            return await call_next(request)

        # Get client IP (handle proxies)
        client_ip = self._get_client_ip(request)

        # Check rate limit
        Session = get_session_local()
        db = Session()
        try:
            settings = get_settings()
            repo = RateLimitRepository(
                db=db,
                limit_per_minute=settings.rate_limit_per_minute,
            )

            allowed, remaining, retry_after = repo.check_and_increment(client_ip)

            if not allowed:
                logger.warning(
                    f"Rate limit exceeded for {client_ip}, retry_after={retry_after}s"
                )
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": "Rate limit exceeded. Please wait before retrying.",
                        "code": "RATE_LIMIT",
                        "retry_after": retry_after,
                    },
                    headers={
                        "Retry-After": str(retry_after),
                        "X-RateLimit-Limit": str(settings.rate_limit_per_minute),
                        "X-RateLimit-Remaining": "0",
                    },
                )

            # Process request
            response = await call_next(request)

            # Add rate limit headers
            response.headers["X-RateLimit-Limit"] = str(settings.rate_limit_per_minute)
            response.headers["X-RateLimit-Remaining"] = str(remaining)

            return response

        except Exception as e:
            logger.error(f"Rate limit error: {e}")
            # On error, allow request to proceed
            return await call_next(request)

        finally:
            db.close()

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request.

        Args:
            request: Incoming request

        Returns:
            Client IP address
        """
        # Check for proxy headers
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            # Take first IP (original client)
            return forwarded.split(",")[0].strip()

        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip.strip()

        # Fall back to direct connection
        if request.client:
            return request.client.host

        return "unknown"
