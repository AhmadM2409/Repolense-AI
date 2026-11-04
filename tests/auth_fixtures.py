"""Isolated real databases; no mocked authentication or SQL sessions."""

from uuid import uuid4
import os
from contextlib import asynccontextmanager
from sqlalchemy.engine import make_url
from sqlalchemy import text
from sqlalchemy.pool import NullPool

import pytest
import pytest_asyncio
from sqlalchemy import event
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from core.models import Profile, Review, User
from alembic import command
from alembic.config import Config
from core.security import hash_password

PASSWORD = "ValidPassword123!"


@pytest.fixture(scope="session")
def password_hash():
    return hash_password(PASSWORD)


@asynccontextmanager
async def isolated_engine():
    """Use only a dedicated verification DB and a newly created random schema."""
    url = os.environ.get("REPOLENS_TEST_DATABASE_URL")
    admin = None
    schema = None
    if url:
        parsed = make_url(url)
        if parsed.drivername != "postgresql+asyncpg" or not (parsed.database or "").startswith(
            "repolens_verify_"
        ):
            raise RuntimeError("PostgreSQL tests require a dedicated repolens_verify_* database")
        admin = create_async_engine(url, poolclass=NullPool)
        schema = "verify_" + uuid4().hex
        async with admin.begin() as connection:
            await connection.execute(text('CREATE SCHEMA "' + schema + '"'))
        engine = create_async_engine(
            url, poolclass=NullPool, connect_args={"server_settings": {"search_path": schema}}
        )
    else:
        engine = create_async_engine("sqlite+aiosqlite://", poolclass=StaticPool)

        @event.listens_for(engine.sync_engine, "connect")
        def foreign_keys(connection, _):
            connection.execute("PRAGMA foreign_keys=ON")

    try:
        yield engine
    finally:
        await engine.dispose()
        if admin is not None:
            try:
                async with admin.begin() as connection:
                    await connection.execute(text('DROP SCHEMA "' + schema + '" CASCADE'))
            finally:
                await admin.dispose()


@pytest_asyncio.fixture
async def database():
    async with isolated_engine() as engine:
        async with engine.begin() as connection:

            def migrate(conn):
                config = Config("alembic.ini")
                config.attributes["connection"] = conn
                command.upgrade(config, "head")

            await connection.run_sync(migrate)
        yield async_sessionmaker(engine, expire_on_commit=False)


@pytest_asyncio.fixture
async def actors(database, password_hash):
    async with database() as db:
        users = [
            User(id=str(uuid4()), email=email, hashed_password=password_hash, is_active=True)
            for email in ("alice@example.com", "bob@example.com")
        ]
        db.add_all(users)
        await db.flush()
        profiles = [Profile(id=str(uuid4()), user_id=user.id) for user in users]
        db.add_all(profiles)
        await db.flush()
        reviews = [
            Review(id=str(uuid4()), profile_id=profile.id, status="pending") for profile in profiles
        ]
        db.add_all(reviews)
        await db.commit()
    return {"users": users, "profiles": profiles, "reviews": reviews}
