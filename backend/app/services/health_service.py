"""Health check service for monitoring dependencies."""

import logging
import time
from typing import Dict, Any

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.models.health import ServiceStatus
from app.services.vector_service import VectorService
from app.services.llm_service import LLMService

logger = logging.getLogger(__name__)


class HealthService:
    """Service for checking health of all dependencies."""

    def __init__(
        self,
        vector_service: VectorService,
        llm_service: LLMService,
    ):
        """Initialize health service.

        Args:
            vector_service: Qdrant vector service
            llm_service: OpenAI LLM service
        """
        self.vector_service = vector_service
        self.llm_service = llm_service

    def check_qdrant(self) -> ServiceStatus:
        """Check Qdrant connection health.

        Returns:
            ServiceStatus for Qdrant
        """
        start = time.time()
        try:
            info = self.vector_service.get_collection_info()
            latency = int((time.time() - start) * 1000)

            if info.get("status") == "ok":
                return ServiceStatus(
                    status="ok",
                    latency_ms=latency,
                )
            else:
                return ServiceStatus(
                    status="error",
                    latency_ms=latency,
                    error=info.get("error", "Unknown error"),
                )
        except Exception as e:
            latency = int((time.time() - start) * 1000)
            logger.error(f"Qdrant health check failed: {e}")
            return ServiceStatus(
                status="error",
                latency_ms=latency,
                error=str(e),
            )

    def check_database(self, db: Session) -> ServiceStatus:
        """Check database connection health.

        Args:
            db: Database session

        Returns:
            ServiceStatus for database
        """
        start = time.time()
        try:
            db.execute(text("SELECT 1"))
            latency = int((time.time() - start) * 1000)
            return ServiceStatus(
                status="ok",
                latency_ms=latency,
            )
        except Exception as e:
            latency = int((time.time() - start) * 1000)
            logger.error(f"Database health check failed: {e}")
            return ServiceStatus(
                status="error",
                latency_ms=latency,
                error=str(e),
            )

    def check_gemini(self) -> ServiceStatus:
        """Check Gemini API connection health.

        Returns:
            ServiceStatus for Gemini
        """
        start = time.time()
        try:
            healthy = self.llm_service.check_health()
            latency = int((time.time() - start) * 1000)

            if healthy:
                return ServiceStatus(
                    status="ok",
                    latency_ms=latency,
                )
            else:
                return ServiceStatus(
                    status="error",
                    latency_ms=latency,
                    error="API check failed",
                )
        except Exception as e:
            latency = int((time.time() - start) * 1000)
            logger.error(f"Gemini health check failed: {e}")
            return ServiceStatus(
                status="error",
                latency_ms=latency,
                error=str(e),
            )

    def get_overall_status(self, services: Dict[str, ServiceStatus]) -> str:
        """Determine overall health status.

        Args:
            services: Dictionary of service statuses

        Returns:
            'healthy' if all services are ok, 'unhealthy' otherwise
        """
        for service_status in services.values():
            if service_status.status != "ok":
                return "unhealthy"
        return "healthy"
