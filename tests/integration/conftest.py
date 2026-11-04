"""Real HTTP routes, authentication and background jobs against an isolated database."""

import httpx
import pytest_asyncio

from api.main import app
from core.database import get_db
from core.services import review_service,repository_analysis


@pytest_asyncio.fixture
async def client(database, monkeypatch):
    async def session_dependency():
        async with database() as session:
            yield session

    # Override only the external database connection; use actual auth and job code.
    previous = app.dependency_overrides.copy()
    app.dependency_overrides[get_db] = session_dependency
    monkeypatch.setattr(review_service, "AsyncSessionLocal", database)
    monkeypatch.setattr(repository_analysis, "AsyncSessionLocal", database)
    try:
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
            yield http
    finally:
        app.dependency_overrides.clear()
        app.dependency_overrides.update(previous)
