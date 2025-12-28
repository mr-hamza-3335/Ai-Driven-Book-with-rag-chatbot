"""Environment configuration using Pydantic Settings."""

from functools import lru_cache
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # OpenAI (optional, fallback)
    openai_api_key: str = ""

    # Gemini API (primary)
    gemini_api_key: str

    # Qdrant
    qdrant_url: str
    qdrant_api_key: str
    qdrant_collection_name: str = "book_chunks"

    # Database
    database_url: str

    # CORS
    cors_origins: str = "*"

    # Rate Limiting
    rate_limit_per_minute: int = 60

    # Payload Size Limit (in bytes, default 1MB)
    max_payload_size: int = 1 * 1024 * 1024

    # Logging
    log_level: str = "INFO"

    # Application
    app_version: str = "1.0.0"

    @property
    def cors_origins_list(self) -> List[str]:
        """Parse CORS origins from comma-separated string."""
        if self.cors_origins == "*":
            return ["*"]
        return [origin.strip() for origin in self.cors_origins.split(",")]


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
