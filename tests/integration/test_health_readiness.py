"""Readiness checks use real schema state and never invent optional service health."""

import pytest
from sqlalchemy import text

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


async def test_health_requires_current_database_only(client, database):
    result = await client.get("/health")
    assert result.status_code == 200
    assert result.json()["dependencies"] == {
        "postgres": "healthy",
        "redis": "not_required",
        "vector_db": "not_required",
    }
    assert result.json()["safety_events_last_hour"] is None
    async with database() as db:
        await db.execute(text("UPDATE alembic_version SET version_num='005'"))
        await db.commit()
    result = await client.get("/health")
    assert result.status_code == 503
    assert result.json()["detail"]["dependencies"]["postgres"] == "unhealthy"
    assert "password" not in result.text.lower()
