"""Official SDK stdio subprocess with real isolated PostgreSQL; no inference."""

import os
import site
import sys
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import update

from core.config import settings
from core.models import Profile, SourceChunk, User
from core.security import create_access_token
from tests.embedding_fixtures import ingested
from tests.integration.test_semantic_retrieval import counts

ROOT = Path(__file__).resolve().parents[2]

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


@asynccontextmanager
async def connect(database, token, errfile):
    site.addsitedir(str(ROOT / ".test-deps"))
    from mcp import Client
    from mcp.client.stdio import StdioServerParameters, stdio_client

    # Child must use the same isolated schema, never another database or public data.
    from sqlalchemy.engine import make_url

    base = make_url(os.environ["REPOLENS_TEST_DATABASE_URL"])
    # Existing engine keeps schema in connect_args; read it from an actual connection.
    from sqlalchemy import text

    async with database() as db:
        schema = await db.scalar(text("SELECT current_schema()"))
    env = {
        "APP_ENV": "test",
        "DATABASE_URL": str(base.render_as_string(hide_password=False)),
        "SECRET_KEY": settings.secret_key,
        "REPOLENS_MCP_ACCESS_TOKEN": token,
        "PYTHONPATH": str(ROOT),
        "REPOLENS_MCP_TEST_SCHEMA": schema,
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    # Schema selection is provided by PostgreSQL connection URL server_settings via a
    # test-only bootstrap; production server/session management itself stays unchanged.
    bootstrap = """
import site
site.addsitedir('.test-deps')
import os
import core.database
from sqlalchemy.ext.asyncio import create_async_engine
replacement = create_async_engine(os.environ['DATABASE_URL'],
    connect_args={'server_settings':
        {'search_path':os.environ.pop('REPOLENS_MCP_TEST_SCHEMA')}})
core.database.AsyncSessionLocal.configure(bind=replacement)
core.database.engine = replacement
import runpy
runpy.run_module('repolens_mcp.server',run_name='__main__')
"""
    with errfile.open("w", encoding="utf-8") as err:
        async with Client(
            stdio_client(
                StdioServerParameters(
                    command=sys.executable, args=["-B", "-c", bootstrap], env=env, cwd=str(ROOT)
                ),
                errlog=err,
            ),
            read_timeout_seconds=12,
        ) as client:
            yield client


async def test_real_protocol_tools_security_and_changes(database, actors, tmp_path):
    source = await ingested(database, actors, files={"guard.py": b"def guard():\n    return 1\n"})
    foreign = await ingested(
        database, actors, index=1, files={"other.py": b"def other():\n    return 2\n"}
    )
    token = create_access_token({"sub": str(actors["users"][0].id)})
    before = await counts(database)
    err = tmp_path / "stderr.log"
    async with connect(database, token, err) as client:
        tools = await client.list_tools(cache_mode="bypass")
        assert {t.name for t in tools.tools} == {
            "list_repositories",
            "list_revisions",
            "search_source",
            "read_source_excerpt",
        }
        assert all(t.input_schema["additionalProperties"] is False for t in tools.tools)

        async def call(name, args):
            return await client.call_tool(name, args)

        repos = await call("list_repositories", {"limit": 1})
        assert not repos.is_error
        assert (
            repos.structured_content["items"][0]["id"] == source.ingestion_result["repository_id"]
        )
        revs = await call(
            "list_revisions", {"repository_id": source.ingestion_result["repository_id"]}
        )
        assert not revs.is_error
        assert (
            revs.structured_content["items"][0]["commit_sha"]
            == source.ingestion_result["commit_sha"]
        )
        scope = {
            "repository_id": source.ingestion_result["repository_id"],
            "revision_id": source.revision_id,
        }
        found = await call("search_source", {**scope, "query": "guard"})
        assert not found.is_error
        match = found.structured_content["matches"][0]
        excerpt = await call(
            "read_source_excerpt",
            {**scope, "chunk_id": match["chunk_id"], "line_start": 1, "line_end": 2},
        )
        assert not excerpt.is_error
        assert excerpt.structured_content["excerpt"] == "def guard():\n    return 1"
        for name, args in [
            ("list_repositories", {"owner_id": "SECRET_OWNER"}),
            ("list_revisions", {"repository_id": foreign.ingestion_result["repository_id"]}),
            ("search_source", {**scope, "revision_id": foreign.revision_id, "query": "guard"}),
            ("read_source_excerpt", {**scope, "chunk_id": str(uuid4())}),
            (
                "read_source_excerpt",
                {**scope, "chunk_id": match["chunk_id"], "line_start": 1, "line_end": 99},
            ),
            ("read_source_excerpt", {**scope, "chunk_id": match["chunk_id"], "path": "C:/secret"}),
            ("run_shell", {"command": "echo forbidden"}),
        ]:
            result = await call(name, args)
            assert result.is_error
            assert token not in str(result) and "SECRET_OWNER" not in str(result)
        async with database() as db:
            await db.execute(
                update(Profile)
                .where(Profile.id == actors["profiles"][0].id)
                .values(user_id=actors["users"][1].id)
            )
            await db.commit()
        assert (await call("search_source", {**scope, "query": "guard"})).is_error
        async with database() as db:
            await db.execute(
                update(User).where(User.id == actors["users"][0].id).values(is_active=False)
            )
            await db.commit()
        assert (await call("list_repositories", {})).is_error
        from mcp.shared.exceptions import MCPError

        with pytest.raises(MCPError):
            await client.list_tools(cache_mode="bypass")
    assert token not in err.read_text() and await counts(database) == before


@pytest.mark.parametrize("fault", ["missing", "malformed", "expired", "inactive"])
async def test_startup_denies_invalid_credentials(database, actors, tmp_path, fault):
    token = (
        ""
        if fault == "missing"
        else "NOT_A_TOKEN"
        if fault == "malformed"
        else create_access_token(
            {"sub": str(actors["users"][0].id)},
            expires_delta=timedelta(seconds=-1) if fault == "expired" else None,
        )
    )
    if fault == "inactive":
        async with database() as db:
            await db.execute(
                update(User).where(User.id == actors["users"][0].id).values(is_active=False)
            )
            await db.commit()
    err = tmp_path / "stderr.log"
    with pytest.raises(ExceptionGroup):
        async with connect(database, token, err) as client:
            await client.list_tools(cache_mode="bypass")
    assert err.read_text().strip() == "RepoLens MCP unavailable"
    assert not token or token not in err.read_text()


async def test_source_corruption_denied(database, actors, tmp_path):
    source = await ingested(database, actors, files={"a.py": b"def guard():\n    return 1\n"})
    token = create_access_token({"sub": str(actors["users"][0].id)})
    async with connect(database, token, tmp_path / "stderr.log") as client:
        scope = {
            "repository_id": source.ingestion_result["repository_id"],
            "revision_id": source.revision_id,
        }
        found = await client.call_tool("search_source", {**scope, "query": "guard"})
        match = found.structured_content["matches"][0]
        async with database() as db:
            await db.execute(
                update(SourceChunk)
                .where(SourceChunk.id == match["chunk_id"])
                .values(text="CORRUPT")
            )
            await db.commit()
        result = await client.call_tool(
            "read_source_excerpt", {**scope, "chunk_id": match["chunk_id"]}
        )
        assert result.is_error and result.structured_content["error"] == "source_integrity"


async def test_token_expiry_between_calls(database, actors, tmp_path):
    import asyncio

    token = create_access_token(
        {"sub": str(actors["users"][0].id)}, expires_delta=timedelta(seconds=8)
    )
    async with connect(database, token, tmp_path / "stderr.log") as client:
        assert not (await client.call_tool("list_repositories", {})).is_error
        from mcp.shared.exceptions import MCPError

        await asyncio.sleep(8)
        result = await client.call_tool("list_repositories", {})
        assert result.is_error and result.structured_content["error"] == "unauthorized"
        with pytest.raises(MCPError):
            await client.list_tools(cache_mode="bypass")
