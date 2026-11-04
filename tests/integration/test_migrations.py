"""Authoritative migration roundtrips on PostgreSQL, including historical 002 data.

007 is intentionally PostgreSQL/pgvector-only; these formerly SQLite head tests
now exercise the actual production engine without removing their assertions.
"""

import io
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from sqlalchemy import MetaData, inspect, text
from sqlalchemy.exc import DBAPIError

from alembic import command
from core.models import Base
from tests.auth_fixtures import isolated_engine


def config(connection):
    cfg = Config("alembic.ini")
    cfg.attributes["connection"] = connection
    return cfg


@pytest.mark.asyncio
@pytest.mark.parametrize("existing", [False, True])
async def test_upgrade_retains_existing_data_and_matches_models(existing):
    async with isolated_engine() as engine, engine.begin() as connection:

        def exercise(conn):
            cfg = config(conn)
            command.upgrade(cfg, "002" if existing else "head")
            if existing:
                metadata = MetaData()
                metadata.reflect(conn)
                stamp = datetime.now(UTC)
                uid, pid, rid, sid = [uuid4() for _ in range(4)]
                conn.execute(
                    metadata.tables["users"]
                    .insert()
                    .values(
                        id=uid,
                        email="fixture@example.com",
                        hashed_password="not-a-real-hash",
                        created_at=stamp,
                        updated_at=stamp,
                        is_active=True,
                    )
                )
                conn.execute(
                    metadata.tables["profiles"]
                    .insert()
                    .values(
                        id=pid,
                        user_id=uid,
                        github_username="example",
                        resume_filename="resume.md",
                        resume_text="actual uploaded text",
                        created_at=stamp,
                        updated_at=stamp,
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
                        created_at=stamp,
                        updated_at=stamp,
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
                        ingested_at=stamp,
                    )
                )
                command.upgrade(cfg, "head")
                assert (
                    conn.scalar(text("SELECT resume_text FROM profiles")) == "actual uploaded text"
                )
                assert conn.scalar(text("SELECT report_kind FROM reviews")) == "legacy_unverified"
                assert conn.scalar(text("SELECT sections FROM reviews")) == []
                assert conn.scalar(text("SELECT count(*) FROM ingested_sources")) == 1
                assert conn.scalar(text("SELECT count(*) FROM findings")) == 0
            assert conn.scalar(text("SELECT version_num FROM alembic_version")) == "012"
            conn.execute(text("SET CONSTRAINTS ALL IMMEDIATE"))
            assert (
                compare_metadata(
                    MigrationContext.configure(conn, opts={"compare_type": True}), Base.metadata
                )
                == []
            )
            for table in Base.metadata.sorted_tables:
                actual = {c["name"] for c in inspect(conn).get_check_constraints(table.name)}
                expected = {
                    c.name for c in table.constraints if c.__class__.__name__ == "CheckConstraint"
                }
                assert actual == expected
            command.upgrade(cfg, "head")
            command.downgrade(cfg, "002")
            assert "repositories" not in inspect(conn).get_table_names()
            if existing:
                assert (
                    conn.scalar(text("SELECT resume_text FROM profiles")) == "actual uploaded text"
                )
            command.upgrade(cfg, "head")
            assert (
                compare_metadata(
                    MigrationContext.configure(conn, opts={"compare_type": True}), Base.metadata
                )
                == []
            )

        await connection.run_sync(exercise)


def test_postgresql_offline_migration_ddl():
    output = io.StringIO()
    cfg = Config("alembic.ini", output_buffer=output)
    cfg.set_main_option("sqlalchemy.url", "postgresql+asyncpg://unused:unused@localhost/unused")
    command.upgrade(cfg, "head", sql=True)
    sql = output.getvalue()
    assert "CREATE TABLE repositories" in sql
    assert "CREATE TABLE evidence_spans" in sql
    assert "fk_evidence_review_revision" in sql
    assert "CREATE FUNCTION immutable_repository_revisions" in sql
    assert sql.index("ADD CONSTRAINT uq_review_scope") < sql.index("CREATE TABLE findings")
    assert "CREATE EXTENSION IF NOT EXISTS vector" in sql
    assert "CREATE CONSTRAINT TRIGGER retrieval_ready" in sql


@pytest.mark.asyncio
async def test_startup_checks_version_without_creating_schema(monkeypatch):
    from core import database

    async with isolated_engine() as engine:
        monkeypatch.setattr(database, "engine", engine)
        with pytest.raises(DBAPIError):
            await database.init_db()
        async with engine.begin() as conn:
            assert await conn.run_sync(lambda c: inspect(c).get_table_names()) == []
            await conn.run_sync(lambda c: command.upgrade(config(c), "head"))
        await database.init_db()
        async with engine.begin() as conn:
            await conn.execute(text("UPDATE alembic_version SET version_num='002'"))
        with pytest.raises(RuntimeError, match="alembic upgrade head"):
            await database.init_db()
