"""Health check endpoint for monitoring."""

import logging

from fastapi import APIRouter, Depends, Response
from sqlalchemy.orm import Session

from app.models.health import HealthResponse, ServiceStatus
from app.services.health_service import HealthService
from app.services.vector_service import VectorService
from app.services.llm_service import LLMService
from app.api.dependencies import (
    get_vector_service,
    get_llm_service,
    get_db,
)
from app.config import get_settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


def get_health_service(
    vector_service: VectorService = Depends(get_vector_service),
    llm_service: LLMService = Depends(get_llm_service),
) -> HealthService:
    """Get health service with dependencies."""
    return HealthService(
        vector_service=vector_service,
        llm_service=llm_service,
    )


@router.get(
    "/health",
    response_model=HealthResponse,
    responses={
        200: {"description": "All services healthy"},
        503: {"description": "One or more services unhealthy"},
    },
)
async def health_check(
    response: Response,
    health_service: HealthService = Depends(get_health_service),
    db: Session = Depends(get_db),
) -> HealthResponse:
    """Check health of all backend dependencies.

    Returns:
        HealthResponse with status of each service
    """
    settings = get_settings()

    # Check all services
    services = {
        "qdrant": health_service.check_qdrant(),
        "database": health_service.check_database(db),
        "gemini": health_service.check_gemini(),
    }

    overall_status = health_service.get_overall_status(services)

    # Set response status code
    if overall_status == "unhealthy":
        response.status_code = 503

    logger.info(f"Health check: {overall_status}")

    return HealthResponse(
        status=overall_status,
        services=services,
        version=settings.app_version,
    )
