"""Database configuration and session management for async SQLAlchemy."""

from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy import text
from alembic.config import Config
from alembic.script import ScriptDirectory
from pathlib import Path

from core.config import settings

# Create async engine
engine = create_async_engine(
    settings.database_url,
    echo=settings.app_env == "development",
    hide_parameters=True,  # Job descriptions/source text must not appear in SQL logs.
    future=True,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
)

# Create async session factory
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)

# Declarative base for models
Base = declarative_base()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for FastAPI that yields a database session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db() -> None:
    """Check migration head only. Apply schema changes with alembic upgrade head."""
    config = Config(str(Path(__file__).resolve().parents[1] / "alembic.ini"))
    config.set_main_option("script_location", str(Path(__file__).resolve().parents[1] / "alembic"))
    expected = ScriptDirectory.from_config(config).get_current_head()
    async with engine.connect() as connection:
        versions = (await connection.execute(text("SELECT version_num FROM alembic_version"))).scalars().all()
        if versions != [expected]:
            raise RuntimeError("Database schema is not current; run alembic upgrade head")
