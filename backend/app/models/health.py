"""Pydantic models for health check functionality."""

from typing import Dict, Optional

from pydantic import BaseModel, Field


class ServiceStatus(BaseModel):
    """Status of a single service."""

    status: str = Field(
        ...,
        description="Status: 'ok' or 'error'",
    )
    latency_ms: Optional[int] = Field(
        None,
        ge=0,
        description="Response latency in milliseconds",
    )
    error: Optional[str] = Field(
        None,
        description="Error message if status is 'error'",
    )


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(
        ...,
        description="Overall status: 'healthy' or 'unhealthy'",
    )
    services: Dict[str, ServiceStatus] = Field(
        ...,
        description="Status of each service",
    )
    version: str = Field(
        ...,
        description="API version",
    )
