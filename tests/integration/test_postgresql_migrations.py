"""Live PostgreSQL migration gates. Opt in with a dedicated verification database."""

import os
from datetime import datetime, UTC
from uuid import uuid4
import pytest
from sqlalchemy import MetaData, text, inspect
from sqlalchemy.exc import DBAPIError
from alembic import command
from alembic.config import Config
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from tests.auth_fixtures import isolated_engine
from core.models import Base

pytestmark = [
    pytest.mark.integration,
    pytest.mark.asyncio,
    pytest.mark.skipif(
        not os.environ.get("REPOLENS_TEST_DATABASE_URL"),
        reason="Set a dedicated PostgreSQL verification URL",
    ),
]


def migration_config(connection):
    config = Config("alembic.ini")
    config.attributes["connection"] = connection
    return config


@pytest.mark.parametrize("existing", [False, True])
async def test_live_postgresql_clean_and_existing_upgrade(existing):
    async with isolated_engine() as engine:
        async with engine.begin() as connection:

            def exercise(conn):
                config = migration_config(conn)
                command.upgrade(config, "002" if existing else "head")
                if existing:
                    metadata = MetaData()
                    metadata.reflect(conn)
                    now = datetime.now(UTC)
                    uid, pid, rid, sid = [uuid4() for _ in range(4)]
                    conn.execute(
                        metadata.tables["users"]
                        .insert()
                        .values(
                            id=uid,
                            email="fixture@example.com",
                            hashed_password="fixture-only",
                            created_at=now,
                            updated_at=now,
                            is_active=True,
                        )
                    )
                    conn.execute(
                        metadata.tables["profiles"]
                        .insert()
                        .values(
                            id=pid,
                            user_id=uid,
                            resume_filename="resume.md",
                            resume_text="Real fixture text",
                            created_at=now,
                            updated_at=now,
                        )
                    )
                    conn.execute(
                        metadata.tables["reviews"]
                        .insert()
                        .values(
                            id=rid,
                            profile_id=pid,
                            status="complete",
                            sections=[],
                            overall_score=0.5,
                            created_at=now,
                            updated_at=now,
                        )
                    )
                    conn.execute(
                        metadata.tables["ingested_sources"]
                        .insert()
                        .values(
                            id=sid,
                            profile_id=pid,
                            source_type="resume",
                            filename="resume.md",
                            chunk_count=0,
                            ingested_at=now,
                        )
                    )
                    command.upgrade(config, "head")
                    assert (
                        conn.scalar(text("SELECT resume_text FROM profiles")) == "Real fixture text"
                    )
                    assert conn.scalar(text("SELECT sections FROM reviews")) == []
                    assert (
                        conn.scalar(text("SELECT report_kind FROM reviews")) == "legacy_unverified"
                    )
                    assert conn.scalar(text("SELECT count(*) FROM ingested_sources")) == 1
                    assert conn.scalar(text("SELECT count(*) FROM findings")) == 0
                assert conn.scalar(text("SELECT version_num FROM alembic_version")) == "012"
                diffs = compare_metadata(
                    MigrationContext.configure(
                        conn, opts={"compare_type": True, "compare_server_default": True}
                    ),
                    Base.metadata,
                )
                assert diffs == []
                for table in Base.metadata.sorted_tables:
                    expected = {
                        item.name
                        for item in table.constraints
                        if item.__class__.__name__ == "CheckConstraint"
                    }
                    actual = {
                        item["name"] for item in inspect(conn).get_check_constraints(table.name)
                    }
                    assert expected == actual
                command.check(config)
                command.upgrade(config, "head")
                command.downgrade(config, "002")
                assert "evidence_spans" not in inspect(conn).get_table_names()
                if existing:
                    assert (
                        conn.scalar(text("SELECT resume_text FROM profiles")) == "Real fixture text"
                    )
                    assert conn.scalar(text("SELECT sections FROM reviews")) == []
                command.upgrade(config, "head")
                assert (
                    compare_metadata(
                        MigrationContext.configure(
                            conn, opts={"compare_type": True, "compare_server_default": True}
                        ),
                        Base.metadata,
                    )
                    == []
                )

            await connection.run_sync(exercise)


async def test_live_postgresql_startup_gate(monkeypatch):
    from core import database

    async with isolated_engine() as engine:
        monkeypatch.setattr(database, "engine", engine)
        with pytest.raises(DBAPIError):
            await database.init_db()
        async with engine.begin() as conn:
            assert await conn.run_sync(lambda c: inspect(c).get_table_names()) == []
            await conn.run_sync(lambda c: command.upgrade(migration_config(c), "head"))
        await database.init_db()
        async with engine.begin() as conn:
            await conn.execute(text("UPDATE alembic_version SET version_num='002'"))
        with pytest.raises(RuntimeError, match="alembic upgrade head"):
            await database.init_db()
