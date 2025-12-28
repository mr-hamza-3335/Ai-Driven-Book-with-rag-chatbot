"""Database configuration with SQLAlchemy."""

from typing import Optional
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.engine import Engine

# Base class for declarative models
Base = declarative_base()

# Lazy initialization
_engine: Optional[Engine] = None
_SessionLocal = None


def get_engine() -> Engine:
    """Get or create SQLAlchemy engine."""
    global _engine
    if _engine is None:
        from app.config import get_settings
        settings = get_settings()

        # Convert postgresql:// to postgresql+psycopg:// for psycopg3
        db_url = settings.database_url
        if db_url.startswith("postgresql://"):
            db_url = db_url.replace("postgresql://", "postgresql+psycopg://", 1)
        elif db_url.startswith("postgres://"):
            db_url = db_url.replace("postgres://", "postgresql+psycopg://", 1)

        _engine = create_engine(
            db_url,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
        )
    return _engine


def get_session_local():
    """Get or create session factory."""
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=get_engine(),
        )
    return _SessionLocal


# For backwards compatibility
@property
def SessionLocal():
    return get_session_local()


def get_db():
    """Dependency for database session."""
    Session = get_session_local()
    db = Session()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Initialize database tables."""
    engine = get_engine()
    Base.metadata.create_all(bind=engine)
