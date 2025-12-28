"""Rate limiting repository with SQLAlchemy."""

import logging
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.sql import func
from sqlalchemy.orm import Session

from app.database import Base

logger = logging.getLogger(__name__)


class RateLimit(Base):
    """SQLAlchemy model for rate limiting."""

    __tablename__ = "rate_limits"

    id = Column(Integer, primary_key=True, index=True)
    client_ip = Column(String(45), unique=True, nullable=False, index=True)
    request_count = Column(Integer, default=1, nullable=False)
    window_start = Column(DateTime, server_default=func.now(), nullable=False)


class RateLimitRepository:
    """Repository for rate limit operations."""

    # Rate limit window in seconds
    WINDOW_SECONDS = 60

    def __init__(self, db: Session, limit_per_minute: int = 60):
        """Initialize rate limit repository.

        Args:
            db: Database session
            limit_per_minute: Maximum requests per minute per IP
        """
        self.db = db
        self.limit_per_minute = limit_per_minute

    def check_and_increment(self, client_ip: str) -> tuple[bool, int, int]:
        """Check rate limit and increment counter.

        Args:
            client_ip: Client IP address

        Returns:
            Tuple of (allowed, remaining, retry_after_seconds)
        """
        now = datetime.utcnow()
        window_start = now - timedelta(seconds=self.WINDOW_SECONDS)

        # Get or create rate limit record
        record = self.db.query(RateLimit).filter(
            RateLimit.client_ip == client_ip
        ).first()

        if record:
            # Check if window has expired
            if record.window_start < window_start:
                # Reset window
                record.request_count = 1
                record.window_start = now
                self.db.commit()
                return True, self.limit_per_minute - 1, 0
            else:
                # Window still active
                if record.request_count >= self.limit_per_minute:
                    # Rate limit exceeded
                    retry_after = int(
                        (record.window_start + timedelta(seconds=self.WINDOW_SECONDS) - now).total_seconds()
                    )
                    return False, 0, max(1, retry_after)
                else:
                    # Increment counter
                    record.request_count += 1
                    self.db.commit()
                    remaining = self.limit_per_minute - record.request_count
                    return True, remaining, 0
        else:
            # Create new record
            record = RateLimit(
                client_ip=client_ip,
                request_count=1,
                window_start=now,
            )
            self.db.add(record)
            self.db.commit()
            return True, self.limit_per_minute - 1, 0

    def get_remaining(self, client_ip: str) -> int:
        """Get remaining requests for a client.

        Args:
            client_ip: Client IP address

        Returns:
            Number of remaining requests
        """
        now = datetime.utcnow()
        window_start = now - timedelta(seconds=self.WINDOW_SECONDS)

        record = self.db.query(RateLimit).filter(
            RateLimit.client_ip == client_ip
        ).first()

        if not record or record.window_start < window_start:
            return self.limit_per_minute

        return max(0, self.limit_per_minute - record.request_count)

    def cleanup_expired(self) -> int:
        """Delete expired rate limit records.

        Returns:
            Number of records deleted
        """
        cutoff = datetime.utcnow() - timedelta(hours=1)
        result = self.db.query(RateLimit).filter(
            RateLimit.window_start < cutoff
        ).delete()
        self.db.commit()

        if result > 0:
            logger.info(f"Cleaned up {result} expired rate limit records")

        return result
