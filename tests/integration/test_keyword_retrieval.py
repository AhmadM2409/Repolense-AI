"""Real PostgreSQL/API isolation and integrity; only GitHub HTTP is simulated."""

import asyncio
import copy
import threading
from uuid import uuid4

import pytest
from fastapi import HTTPException
from sqlalchemy import delete, func, select, update

from core.config import settings
from core.models import Evidence, Finding, IngestedSource, Profile, Review, SourceChunk, User
from core.models.repository import Repository
from core.security import create_access_token
from core.services import keyword_retrieval as service
from tests.analysis_fixtures import verified_ingestion

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]
FILES = {
    "README.md": b"user cache\n",
    "src/cache.py": b"def getUser(user_cache):\n    return user_cache\n",
    "tests/test_cache.py": b"assert cache\n",
}


def headers(actors, index=0):
    return {"Authorization": "Bearer " + create_access_token({"sub": actors["users"][index].id})}


async def ingested(database, actors, index=0, files=None):
    async with database() as db:
        return await verified_ingestion(
            db, actors["profiles"][index].id, actors["users"][index].id, files or FILES
        )


def url(job):
    return (
        "/repositories/"
        + job.ingestion_result["repository_id"]
        + "/revisions/"
        + job.revision_id
        + "/search"
    )


async def call(client, actors, job, **data):
    return await client.post(url(job), headers=headers(actors), json={"query": "cache", **data})


async def counts(database):
    async with database() as db:
        return [
            await db.scalar(select(func.count()).select_from(model))
            for model in (Review, IngestedSource, SourceChunk, Finding, Evidence)
        ]


async def test_exact_canonical_results_determinism_and_no_writes(client, database, actors):
    job = await ingested(database, actors)
    before = await counts(database)
    response = await call(client, actors, job)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result == (await call(client, actors, job)).json()
    assert result["total_candidate_count"] == result["total_match_count"] == 3
    assert result["commit_sha"] == job.ingestion_result["commit_sha"]
    assert result["mode"] == "keyword" and "query" not in result
    assert result["query_metadata"] == {"character_count": 5, "term_count": 1}
    async with database() as db:
        for rank, match in enumerate(result["matches"], 1):
            chunk = await db.get(SourceChunk, match["chunk_id"])
            source = await db.get(IngestedSource, match["source_id"])
            assert chunk.source_id == source.id and source.revision_id == job.revision_id
            assert (
                match["excerpt"]
                == chunk.text
                == "\n".join(source.text.splitlines()[match["line_start"] - 1 : match["line_end"]])
            )
            assert match["path"] == source.path
            assert match["commit_sha"] == result["commit_sha"]
            assert match["rank"] == rank and match["bm25_score"] > 0
    assert (await call(client, actors, job, top_k=1)).json()["total_match_count"] == 3
    assert len((await call(client, actors, job, top_k=1)).json()["matches"]) == 1
    assert await counts(database) == before


@pytest.mark.parametrize("query", ["!!!", "not_present_anywhere", "___"])
async def test_no_meaningful_matches(client, database, actors, query):
    job = await ingested(database, actors)
    response = await call(client, actors, job, query=query)
    assert response.status_code == 200, response.text
    assert response.json()["matches"] == [] and response.json()["total_candidate_count"] == 3


async def test_two_users_two_revisions_and_ties(client, database, actors):
    first = await ingested(database, actors, files={"a.py": b"needle", "b.py": b"needle"})
    foreign = await ingested(database, actors, 1, files={"a.py": b"needle", "b.py": b"needle"})
    other = await ingested(database, actors, files={"a.py": b"new_revision"})
    a = await call(client, actors, first, query="needle")
    assert a.status_code == 200, a.text
    matches = a.json()["matches"]
    assert [m["chunk_id"] for m in matches] == sorted(m["chunk_id"] for m in matches)
    assert len({m["bm25_score"] for m in matches}) == 1
    assert first.revision_id != other.revision_id
    assert (await call(client, actors, other, query="needle")).json()["matches"] == []
    denied = await call(client, actors, foreign, query="needle")
    assert denied.status_code == 404
    async with database() as db:
        with pytest.raises(HTTPException) as error:
            await service.search(
                db,
                actors["users"][0].id,
                foreign.ingestion_result["repository_id"],
                foreign.revision_id,
                {"query": "needle"},
            )
        assert error.value.status_code == 404


async def test_unauthenticated_inactive_missing_and_mismatched_revision(client, database, actors):
    job = await ingested(database, actors)
    assert (await client.post(url(job), json={"query": "cache"})).status_code == 401
    foreign = await ingested(database, actors, 1)
    mixed = url(job).replace(job.revision_id, foreign.revision_id)
    assert (
        await client.post(mixed, headers=headers(actors), json={"query": "cache"})
    ).status_code == 404
    missing = url(job).replace(job.revision_id, str(uuid4()))
    assert (
        await client.post(missing, headers=headers(actors), json={"query": "cache"})
    ).status_code == 404
    async with database() as db:
        await db.execute(
            update(User).where(User.id == actors["users"][0].id).values(is_active=False)
        )
        await db.commit()
    assert (await call(client, actors, job)).status_code == 401
    async with database() as db:
        with pytest.raises(HTTPException) as error:
            await service.search(
                db,
                actors["users"][0].id,
                job.ingestion_result["repository_id"],
                job.revision_id,
                {"query": "cache"},
            )
        assert error.value.status_code == 401


@pytest.mark.parametrize(
    "data",
    [
        {"query": ""},
        {"query": " "},
        {"query": "x" * 1025},
        {"top_k": 0},
        {"top_k": True},
        {"top_k": 21},
        {"mode": "unsupported"},
        {"profile_id": "SECRET_SENSITIVE_VALUE"},
    ],
)
async def test_invalid_inputs_do_not_echo_input(client, database, actors, data):
    job = await ingested(database, actors)
    response = await call(client, actors, job, **data)
    assert response.status_code == 422, response.text
    assert "SECRET_SENSITIVE_VALUE" not in response.text and '"input"' not in response.text


async def test_transport_body_limit(client, database, actors):
    job = await ingested(database, actors)
    response = await client.post(url(job), headers=headers(actors), json={"query": "secret" * 4000})
    assert response.status_code == 413

    async def body():
        yield b'{"query":"'
        yield b"a" * 17000
        yield b'"}'

    response = await client.post(
        url(job), headers={**headers(actors), "Content-Type": "application/json"}, content=body()
    )
    assert response.status_code == 413, response.text


@pytest.mark.parametrize(
    "fault",
    [
        "unverified",
        "failed",
        "missing_manifest",
        "wrong_sha",
        "hash",
        "chunk_text",
        "line",
        "missing_chunk",
        "blob",
        "count",
    ],
)
async def test_corruption_fails_closed(client, database, actors, fault):
    job = await ingested(database, actors)
    async with database() as db:
        if fault in {
            "unverified",
            "failed",
            "missing_manifest",
            "wrong_sha",
            "blob",
            "count",
            "hash",
        }:
            proof = copy.deepcopy(job.ingestion_result)
            if fault == "unverified":
                proof["verified"] = False
            if fault == "missing_manifest":
                proof["files"] = []
            if fault == "wrong_sha":
                proof["commit_sha"] = "a" * 40
            if fault == "blob":
                proof["files"][0]["blob_sha"] = "b" * 40
            if fault == "count":
                proof["chunk_count"] += 1
            if fault == "hash":
                proof["files"][0]["content_hash"] = "0" * 64
            await db.execute(
                update(Review)
                .where(Review.id == job.id)
                .values(
                    ingestion_result=proof, status="failed" if fault == "failed" else "complete"
                )
            )
        else:
            ident = await db.scalar(
                select(SourceChunk.id)
                .join(IngestedSource)
                .where(IngestedSource.revision_id == job.revision_id)
            )
            if fault == "missing_chunk":
                await db.execute(delete(SourceChunk).where(SourceChunk.id == ident))
            else:
                await db.execute(
                    update(SourceChunk)
                    .where(SourceChunk.id == ident)
                    .values(**({"text": "forged"} if fault == "chunk_text" else {"line_end": 999}))
                )
        await db.commit()
    response = await call(client, actors, job)
    assert response.status_code == 409, response.text
    assert "cache" not in response.text and "matches" not in response.json()


@pytest.mark.parametrize(
    "setting,value",
    [
        ("keyword_search_max_sources", 1),
        ("keyword_search_max_chunks", 1),
        ("keyword_search_max_corpus_bytes", 1),
        ("keyword_search_max_metadata_bytes", 1),
        ("keyword_search_max_tokens", 1),
        ("keyword_search_max_excerpt_bytes", 1),
        ("keyword_search_max_response_bytes", 1),
    ],
)
async def test_resource_limits_explicit_no_partial(
    client, database, actors, monkeypatch, setting, value
):
    job = await ingested(database, actors)
    monkeypatch.setattr(settings, setting, value)
    response = await call(client, actors, job)
    assert response.status_code == 413, response.text
    assert "matches" not in response.json()


@pytest.mark.parametrize(
    "change,status", [("inactive", 401), ("ownership", 404), ("delete", 404), ("snapshot", 409)]
)
async def test_recheck_across_real_async_ranking(
    client, database, actors, monkeypatch, change, status
):
    job = await ingested(database, actors)
    entered, release = threading.Event(), threading.Event()
    real_rank = service.rank

    def paused(*args):
        entered.set()
        assert release.wait(10), "test synchronization timed out"
        return real_rank(*args)

    monkeypatch.setattr(service, "rank", paused)
    pending = asyncio.create_task(call(client, actors, job))
    try:
        assert await asyncio.to_thread(entered.wait, 10)
        async with database() as db:
            if change == "inactive":
                await db.execute(
                    update(User).where(User.id == actors["users"][0].id).values(is_active=False)
                )
            elif change == "ownership":
                await db.execute(
                    update(Profile)
                    .where(Profile.id == actors["profiles"][0].id)
                    .values(user_id=actors["users"][1].id)
                )
            elif change == "delete":
                await db.execute(
                    delete(Repository).where(Repository.id == job.ingestion_result["repository_id"])
                )
            else:
                source = await db.scalar(
                    select(IngestedSource)
                    .where(IngestedSource.revision_id == job.revision_id)
                    .limit(1)
                )
                metadata = copy.deepcopy(source.source_metadata)
                metadata["test_marker"] = "changed during ranking"
                await db.execute(
                    update(IngestedSource)
                    .where(IngestedSource.id == source.id)
                    .values(source_metadata=metadata)
                )
            await db.commit()
    finally:
        release.set()
    response = await pending
    assert response.status_code == status, response.text


async def test_parallel_users_stateless(client, database, actors):
    a = await ingested(database, actors, files={"a.py": b"private_alice"})
    b = await ingested(database, actors, 1, files={"b.py": b"private_bob"})
    results = await asyncio.gather(
        call(client, actors, a, query="private_alice"),
        client.post(url(b), headers=headers(actors, 1), json={"query": "private_bob"}),
    )
    assert all(r.status_code == 200 for r in results), [r.text for r in results]
    assert results[0].json()["matches"][0]["path"] == "a.py"
    assert results[1].json()["matches"][0]["path"] == "b.py"


async def test_empty_verified_text(client, database, actors):
    job = await ingested(database, actors, files={"README.md": b""})
    response = await call(client, actors, job)
    assert response.status_code == 200, response.text
    assert response.json()["matches"] == [] and response.json()["total_candidate_count"] == 0


async def test_capacity_timeout_and_recovery(client, database, actors, monkeypatch):
    job = await ingested(database, actors)
    entered, release = threading.Event(), threading.Event()
    real_rank = service.rank

    def paused(*args):
        entered.set()
        assert release.wait(10)
        return real_rank(*args)

    monkeypatch.setattr(service, "rank", paused)
    monkeypatch.setattr(settings, "keyword_search_timeout_seconds", 0.05)
    # Consume the other slot so a timed-out CPU task must keep its own capacity.
    assert service._SLOTS.acquire(blocking=False)
    try:
        pending = asyncio.create_task(call(client, actors, job))
        assert await asyncio.to_thread(entered.wait, 10)
        response = await pending
        assert response.status_code == 503, response.text
        busy = await call(client, actors, job)
        assert busy.status_code == 429 and busy.headers["Retry-After"] == "1"
    finally:
        release.set()
        service._SLOTS.release()
    await asyncio.sleep(0.1)
    monkeypatch.setattr(service, "rank", real_rank)
    monkeypatch.setattr(settings, "keyword_search_timeout_seconds", 5)
    assert (await call(client, actors, job)).status_code == 200


async def test_ownership_denied_before_corpus_queries(database, actors):
    from sqlalchemy import event

    job = await ingested(database, actors, 1)
    statements = []
    async with database() as db:
        engine = db.bind.sync_engine

        def record(connection, cursor, statement, parameters, context, executemany):
            statements.append(statement.lower())

        event.listen(engine, "before_cursor_execute", record)
        try:
            with pytest.raises(HTTPException) as error:
                await service.search(
                    db,
                    actors["users"][0].id,
                    job.ingestion_result["repository_id"],
                    job.revision_id,
                    {"query": "cache"},
                )
            assert error.value.status_code == 404
        finally:
            event.remove(engine, "before_cursor_execute", record)
    assert not any(
        "ingested_sources" in statement or "source_chunks" in statement for statement in statements
    )


async def test_dirty_session_is_not_rolled_back(database, actors):
    job = await ingested(database, actors)
    async with database() as db:
        profile = await db.get(Profile, actors["profiles"][0].id)
        profile.github_username = "pending_update"
        with pytest.raises(RuntimeError, match="clean session"):
            await service.search(
                db,
                actors["users"][0].id,
                job.ingestion_result["repository_id"],
                job.revision_id,
                {"query": "cache"},
            )
        assert db.dirty
        await db.rollback()


async def test_query_and_text_not_logged(client, database, actors, capsys, caplog):
    secret = "SENSITIVE_QUERY_CONTENT"
    job = await ingested(database, actors, files={"a.py": secret.encode()})
    response = await call(client, actors, job, query=secret)
    assert response.status_code == 200, response.text
    assert secret not in capsys.readouterr().out + caplog.text


async def test_sql_lock_timeout_is_sanitized_and_recoverable(client, database, actors):
    job = await ingested(database, actors)
    async with database() as lock:
        if lock.bind.dialect.name != "postgresql":
            pytest.fail("This lock-timeout check requires the configured real PostgreSQL database")
        await lock.execute(
            select(Profile).where(Profile.id == actors["profiles"][0].id).with_for_update()
        )
        response = await call(client, actors, job)
        assert response.status_code == 503, response.text
        assert "SELECT" not in response.text and "cache" not in response.text
        await lock.rollback()
    assert (await call(client, actors, job)).status_code == 200


async def test_caller_cancellation_keeps_capacity_until_thread_exits(database, actors, monkeypatch):
    job = await ingested(database, actors, files={"a.py": b"cache"})
    entered, release = threading.Event(), threading.Event()
    real_rank = service.rank

    def paused(*args):
        entered.set()
        assert release.wait(10)
        return real_rank(*args)

    monkeypatch.setattr(service, "rank", paused)
    assert service._SLOTS.acquire(blocking=False)
    recovered = False

    async def execute():
        async with database() as db:
            return await service.search(
                db,
                actors["users"][0].id,
                job.ingestion_result["repository_id"],
                job.revision_id,
                {"query": "cache"},
            )

    task = asyncio.create_task(execute())
    try:
        assert await asyncio.to_thread(entered.wait, 10)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not service._SLOTS.acquire(blocking=False)
        release.set()
        for _ in range(100):
            if service._SLOTS.acquire(blocking=False):
                recovered = True
                break
            await asyncio.sleep(0.01)
        assert recovered, "cancelled request leaked its CPU capacity slot"
    finally:
        release.set()
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        if recovered:
            service._SLOTS.release()
        service._SLOTS.release()


async def test_existing_immutable_source_guards_remain_enforced(client, database, actors):
    from sqlalchemy.exc import DBAPIError

    job = await ingested(database, actors, files={"a.py": b"cache"})
    async with database() as db:
        for changed in ({"content_hash": "0" * 64}, {"text": "forged"}):
            with pytest.raises(DBAPIError, match="immutable source/revision"):
                await db.execute(
                    update(IngestedSource)
                    .where(IngestedSource.revision_id == job.revision_id)
                    .values(**changed)
                )
            await db.rollback()
    assert (await call(client, actors, job)).status_code == 200


async def test_snapshot_locks_prevent_chunk_insert_phantoms(database, actors):
    from sqlalchemy import text
    from sqlalchemy.exc import DBAPIError

    job = await ingested(database, actors, files={"a.py": b"cache"})
    async with database() as snapshot_db:
        snapshot = await service._snapshot(
            snapshot_db,
            actors["users"][0].id,
            job.ingestion_result["repository_id"],
            job.revision_id,
            service.SearchLimits.configured(),
        )
        chunk = snapshot.chunks[0][0]
        async with database() as writer:
            await writer.execute(text("SET LOCAL lock_timeout = '100ms'"))
            writer.add(
                SourceChunk(
                    id=str(uuid4()),
                    profile_id=str(chunk.profile_id),
                    source_id=str(chunk.source_id),
                    chunk_index=99,
                    text="cache",
                    line_start=1,
                    line_end=1,
                )
            )
            with pytest.raises(DBAPIError, match="lock timeout"):
                await writer.flush()
            await writer.rollback()
        await snapshot_db.rollback()
