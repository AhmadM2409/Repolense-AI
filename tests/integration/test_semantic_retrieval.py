"""Actual PostgreSQL/pgvector semantic API; external embeddings are deterministic doubles."""

import asyncio
import copy

import pytest
from fastapi import HTTPException
from pydantic import SecretStr
from sqlalchemy import delete, func, select, update
from sqlalchemy.exc import DBAPIError

from core.config import settings
from core.models import Evidence, Finding, IngestedSource, Profile, Review, SourceChunk, User
from core.models.retrieval import ChunkEmbedding, RetrievalIndex
from core.security import create_access_token
from core.services import retrieval_indexing
from core.services import semantic_retrieval as service
from ingestion.embeddings.approved import EmbeddingError
from tests.embedding_fixtures import submitted
from workers.run import tick

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]
FILES = {"a.py": b"POS\nline two\n", "b.py": b"ZERO\n", "c.py": b"NEG\n", "d.py": b"POS\n"}


class RankProvider:
    calls = []

    def __init__(self, config):
        self.config = config

    async def embed_documents(self, texts, reserve):
        await reserve()
        return [self.vector(text) for text in texts]

    def vector(self, text):
        vector = [0.0] * self.config.dimensions
        if text.startswith("NEG"):
            vector[0] = -1.0
        elif text.startswith("ZERO"):
            vector[1] = 1.0
        else:
            vector[0] = 1.0
        return vector

    async def embed_query(self, text, reserve):
        await reserve()
        type(self).calls.append(self.config.provider)
        return self.vector("POS")


@pytest.fixture
def provider(monkeypatch):
    RankProvider.calls = []
    monkeypatch.setattr(retrieval_indexing, "make_provider", RankProvider)
    return RankProvider


def auth(actors, index=0):
    return {"Authorization": "Bearer " + create_access_token({"sub": actors["users"][index].id})}


async def ready(database, actors, provider, index=0, files=None, configuration="default"):
    ident, source = await submitted(database, actors, index, files or FILES, configuration)
    assert await tick(database, "semantic-index")
    async with database() as db:
        g = await db.get(RetrievalIndex, ident)
        assert g.state == "ready"
    return ident, source


def endpoint(source):
    return (
        "/repositories/"
        + source.ingestion_result["repository_id"]
        + "/revisions/"
        + source.revision_id
        + "/search"
    )


async def query(client, actors, source, **data):
    return await client.post(
        endpoint(source),
        headers=auth(actors),
        json={"query": "PRIVATE_QUERY", "mode": "semantic", **data},
    )


async def counts(database):
    async with database() as db:
        return [
            await db.scalar(select(func.count()).select_from(table))
            for table in (Review, Finding, Evidence, RetrievalIndex, ChunkEmbedding)
        ]


async def test_exact_ranking_negative_distance_ties_provenance_and_no_writes(
    client, database, actors, provider
):
    ident, source = await ready(database, actors, provider)
    before = await counts(database)
    response = await query(client, actors, source)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["index_id"] == ident and body["dimensions"] == 768
    assert body["total_candidate_count"] == 4 and body["mode"] == "semantic"
    assert [row["cosine_distance"] for row in body["matches"]] == [0, 0, 1, 2]
    assert [row["chunk_id"] for row in body["matches"][:2]] == sorted(
        row["chunk_id"] for row in body["matches"][:2]
    )
    async with database() as db:
        for rank, row in enumerate(body["matches"], 1):
            chunk = await db.get(SourceChunk, row["chunk_id"])
            doc = await db.get(IngestedSource, row["source_id"])
            assert chunk.source_id == doc.id and doc.revision_id == source.revision_id
            assert (
                row["excerpt"]
                == chunk.text
                == "\n".join(doc.text.splitlines()[row["line_start"] - 1 : row["line_end"]])
            )
            assert (
                row["path"] == doc.path
                and row["rank"] == rank
                and row["provenance_verified"] is True
            )
            assert row["commit_sha"] == source.ingestion_result["commit_sha"]
    assert (await query(client, actors, source, index_id=ident)).json() == body
    assert len((await query(client, actors, source, top_k=1)).json()["matches"]) == 1
    assert await counts(database) == before
    assert "PRIVATE_QUERY" not in response.text and '"embedding":' not in response.text


async def test_missing_ready_no_provider(client, database, actors, provider):
    ident, source = await submitted(database, actors, files=FILES)
    result = await query(client, actors, source)
    assert result.status_code == 409, result.text
    assert provider.calls == []


async def test_two_users_two_revisions_and_foreign_selectors(client, database, actors, provider):
    a, source = await ready(database, actors, provider)
    b, foreign = await ready(database, actors, provider, 1)
    c, other = await ready(database, actors, provider, files={"x.py": b"POS another revision"})
    for path, headers in [
        (endpoint(source), auth(actors, 1)),
        (endpoint(foreign), auth(actors)),
        (endpoint(source).replace(source.revision_id, other.revision_id), auth(actors)),
    ]:
        # The other revision remains isolated by the explicit index selector.
        result = await client.post(
            path, headers=headers, json={"query": "private", "mode": "semantic", "index_id": a}
        )
        assert result.status_code == 404, result.text
    assert (await query(client, actors, source, index_id=b)).status_code == 404
    assert provider.calls == []
    one, two = await asyncio.gather(query(client, actors, source), query(client, actors, other))
    assert one.status_code == two.status_code == 200, [one.text, two.text]
    assert one.json()["index_id"] == a and two.json()["index_id"] == c
    assert all(m["path"] == "x.py" for m in two.json()["matches"])
    assert (
        await client.post(endpoint(source), json={"mode": "semantic", "query": "x"})
    ).status_code == 401
    async with database() as db:
        with pytest.raises(HTTPException) as error:
            await service.search(
                db,
                actors["users"][1].id,
                source.ingestion_result["repository_id"],
                source.revision_id,
                {"mode": "semantic", "query": "x"},
            )
        assert error.value.status_code == 404


@pytest.mark.parametrize(
    "fault",
    [
        "inactive",
        "failed",
        "cancelled",
        "building",
        "deleted",
        "version",
        "endpoint",
        "dimension",
        "metric",
        "source",
        "proof",
        "missing_vector",
    ],
)
async def test_unavailable_or_incompatible_before_provider(
    client, database, actors, provider, monkeypatch, fault
):
    ident, source = await ready(database, actors, provider)
    if fault == "version":
        monkeypatch.setattr(settings, "embedding_lmstudio_version", "other-v2")
    elif fault == "endpoint":
        monkeypatch.setattr(
            settings, "embedding_lmstudio_base_url", "http://host.docker.internal:1235/v1"
        )
    elif fault == "dimension":
        monkeypatch.setattr(settings, "embedding_lmstudio_dimensions", 1024)
    else:
        async with database() as db:
            g = await db.get(RetrievalIndex, ident)
            if fault == "inactive":
                await db.execute(
                    update(User).where(User.id == actors["users"][0].id).values(is_active=False)
                )
            elif fault in {"failed", "cancelled", "building"}:
                g.state = fault
                g.published_at = None
                g.published_count = 0
            elif fault == "deleted":
                await db.delete(g)
            elif fault == "metric":
                # Exercise the real immutable-descriptor SQL guard.
                with pytest.raises(DBAPIError):
                    await db.execute(
                        update(RetrievalIndex).where(RetrievalIndex.id == ident).values(metric="l2")
                    )
                await db.rollback()
                return
            elif fault == "source":
                row = await db.scalar(
                    select(SourceChunk)
                    .where(SourceChunk.profile_id == actors["profiles"][0].id)
                    .limit(1)
                )
                row.text = "corrupt"
            elif fault == "proof":
                proof = copy.deepcopy(source.ingestion_result)
                proof["verified"] = False
                await db.execute(
                    update(Review).where(Review.id == source.id).values(ingestion_result=proof)
                )
            else:
                key = await db.scalar(
                    select(ChunkEmbedding.chunk_id).where(ChunkEmbedding.index_id == ident).limit(1)
                )
                await db.execute(
                    delete(ChunkEmbedding).where(
                        ChunkEmbedding.index_id == ident, ChunkEmbedding.chunk_id == key
                    )
                )
            await db.commit()
    result = await query(client, actors, source, index_id=ident)
    assert result.status_code == (
        401 if fault == "inactive" else 404 if fault == "deleted" else 409
    ), result.text
    assert provider.calls == []


@pytest.mark.parametrize(
    "fault",
    [
        "inactive",
        "owner",
        "invalidate",
        "delete_index",
        "delete_source",
        "chunk",
        "metadata",
        "vector",
        "configuration",
    ],
)
async def test_post_embedding_revalidation(client, database, actors, provider, monkeypatch, fault):
    ident, source = await ready(database, actors, provider)
    entered, release = asyncio.Event(), asyncio.Event()

    class Paused(RankProvider):
        async def embed_query(self, value, reserve):
            await reserve()
            entered.set()
            await release.wait()
            return self.vector("POS")

    monkeypatch.setattr(retrieval_indexing, "make_provider", Paused)
    pending = asyncio.create_task(query(client, actors, source))
    try:
        await asyncio.wait_for(entered.wait(), 10)
        async with database() as db:
            if fault == "inactive":
                await db.execute(
                    update(User).where(User.id == actors["users"][0].id).values(is_active=False)
                )
            elif fault == "owner":
                await db.execute(
                    update(Profile)
                    .where(Profile.id == actors["profiles"][0].id)
                    .values(user_id=actors["users"][1].id)
                )
            elif fault == "delete_index":
                await db.execute(delete(RetrievalIndex).where(RetrievalIndex.id == ident))
            elif fault == "invalidate":
                await db.execute(
                    update(RetrievalIndex)
                    .where(RetrievalIndex.id == ident)
                    .values(state="cancelled", published_at=None, published_count=0)
                )
            elif fault == "delete_source":
                await db.execute(
                    delete(IngestedSource).where(IngestedSource.revision_id == source.revision_id)
                )
            elif fault == "chunk":
                row = await db.scalar(
                    select(SourceChunk)
                    .where(SourceChunk.profile_id == actors["profiles"][0].id)
                    .limit(1)
                )
                row.text = "invalid"
            elif fault == "metadata":
                row = await db.scalar(
                    select(IngestedSource)
                    .where(IngestedSource.revision_id == source.revision_id)
                    .limit(1)
                )
                row.source_metadata = {**row.source_metadata, "mutation": True}
            elif fault == "vector":
                row = await db.scalar(
                    select(ChunkEmbedding).where(ChunkEmbedding.index_id == ident).limit(1)
                )
                row.embedding = [-value for value in row.embedding]
            else:
                monkeypatch.setattr(settings, "embedding_lmstudio_version", "revoked")
            await db.commit()
    finally:
        release.set()
    result = await pending
    assert result.status_code in (
        {401} if fault == "inactive" else {404} if fault == "owner" else {404, 409}
    ), result.text
    assert "matches" not in result.json()


async def test_hosted_query_permission_is_separate_and_dimensions_do_not_mix(
    client, database, actors, provider, monkeypatch
):
    monkeypatch.setattr(settings, "embedding_hosted_source_allowed", True)
    monkeypatch.setattr(settings, "embedding_mistral_api_key", SecretStr("test-only-not-real"))
    monkeypatch.setattr(settings, "embedding_mistral_max_requests", 10)
    ident, source = await ready(database, actors, provider, configuration="mistral")
    denied = await query(client, actors, source, configuration="mistral")
    assert denied.status_code == 409 and provider.calls == []
    monkeypatch.setattr(settings, "semantic_search_hosted_query_allowed", True)
    allowed = await query(client, actors, source, configuration="mistral", index_id=ident)
    assert allowed.status_code == 200, allowed.text
    assert allowed.json()["dimensions"] == 1024 and provider.calls == ["mistral"]
    provider.calls = []
    assert (
        await query(client, actors, source, configuration="lmstudio", index_id=ident)
    ).status_code == 409
    assert provider.calls == []


@pytest.mark.parametrize("fault", ["nan", "zero", "dimension", "none", "multiple", "exception"])
async def test_bad_query_vectors_fail_closed_no_fallback(
    client, database, actors, provider, monkeypatch, fault
):
    _, source = await ready(database, actors, provider)

    class Invalid(RankProvider):
        async def embed_query(self, value, reserve):
            await reserve()
            if fault == "exception":
                raise EmbeddingError("provider_unavailable", True)
            if fault == "none":
                return None
            if fault == "multiple":
                return [self.vector("POS"), self.vector("POS")]
            vector = self.vector("POS")
            if fault == "nan":
                vector[0] = float("nan")
            if fault == "zero":
                vector = [0.0] * self.config.dimensions
            if fault == "dimension":
                vector.pop()
            return vector

    monkeypatch.setattr(retrieval_indexing, "make_provider", Invalid)
    result = await query(client, actors, source)
    assert result.status_code == 503, result.text
    assert "PRIVATE_QUERY" not in result.text and "matches" not in result.json()


@pytest.mark.parametrize(
    "limit",
    [
        "keyword_search_max_excerpt_bytes",
        "keyword_search_max_response_bytes",
        "keyword_search_max_chunks",
        "keyword_search_max_corpus_bytes",
    ],
)
async def test_bounds_do_not_return_partial(client, database, actors, provider, monkeypatch, limit):
    _, source = await ready(database, actors, provider)
    monkeypatch.setattr(settings, limit, 1)
    result = await query(client, actors, source)
    assert result.status_code == 413, result.text
    assert "matches" not in result.json()


async def test_capacity_timeout_cancellation_and_recovery(
    client, database, actors, provider, monkeypatch
):
    _, source = await ready(database, actors, provider)
    entered, release = asyncio.Event(), asyncio.Event()

    class Paused(RankProvider):
        async def embed_query(self, value, reserve):
            await reserve()
            entered.set()
            await release.wait()
            return self.vector("POS")

    monkeypatch.setattr(retrieval_indexing, "make_provider", Paused)
    assert service._SLOTS.acquire(blocking=False)
    task = asyncio.create_task(query(client, actors, source))
    try:
        await asyncio.wait_for(entered.wait(), 10)
        busy = await query(client, actors, source)
        assert busy.status_code == 429
    finally:
        release.set()
        service._SLOTS.release()
    assert (await task).status_code == 200

    class Timeout(RankProvider):
        async def embed_query(self, value, reserve):
            await reserve()
            raise TimeoutError("PRIVATE_QUERY_SECRET")

    monkeypatch.setattr(retrieval_indexing, "make_provider", Timeout)
    result = await query(client, actors, source)
    assert result.status_code == 503 and "PRIVATE_QUERY" not in result.text
    monkeypatch.setattr(retrieval_indexing, "make_provider", RankProvider)
    assert (await query(client, actors, source)).status_code == 200


async def test_empty_ready_corpus_and_dirty_session(client, database, actors, provider):
    ident, source = await ready(database, actors, provider, files={"README.md": b""})
    result = await query(client, actors, source, index_id=ident)
    assert result.status_code == 200 and result.json()["matches"] == [] and provider.calls == []
    async with database() as db:
        user = await db.get(User, actors["users"][0].id)
        user.is_active = False
        with pytest.raises(RuntimeError, match="clean session"):
            await service.search(
                db,
                actors["users"][0].id,
                source.ingestion_result["repository_id"],
                source.revision_id,
                {"mode": "semantic", "query": "x"},
            )
        assert db.dirty
        await db.rollback()


async def test_no_query_or_source_log_leak(client, database, actors, provider, capsys, caplog):
    _, source = await ready(database, actors, provider, files={"a.py": b"PRIVATE_SOURCE_CONTENT"})
    result = await query(client, actors, source)
    assert result.status_code == 200, result.text
    logs = capsys.readouterr().out + caplog.text
    assert "PRIVATE_QUERY" not in logs and "PRIVATE_SOURCE_CONTENT" not in logs


async def test_database_rejects_mixed_profile_or_revision_vectors(database, actors, provider):
    ident, source = await ready(database, actors, provider)
    foreign, _ = await ready(database, actors, provider, 1)
    async with database() as db:
        row = await db.scalar(
            select(ChunkEmbedding).where(ChunkEmbedding.index_id == ident).limit(1)
        )
        with pytest.raises(DBAPIError):
            row.profile_id = actors["profiles"][1].id
            await db.commit()
        await db.rollback()
        other = await db.scalar(
            select(ChunkEmbedding).where(ChunkEmbedding.index_id == foreign).limit(1)
        )
        with pytest.raises(DBAPIError):
            other.revision_id = source.revision_id
            await db.commit()
        await db.rollback()


async def test_sql_timeout_sanitized_and_recoverable(client, database, actors, provider):
    _, source = await ready(database, actors, provider)
    async with database() as lock:
        await lock.execute(
            select(Profile).where(Profile.id == actors["profiles"][0].id).with_for_update()
        )
        response = await query(client, actors, source)
        assert response.status_code == 503, response.text
        assert "SELECT" not in response.text
        await lock.rollback()
    assert (await query(client, actors, source)).status_code == 200


async def test_per_query_retry_budget_and_overall_timeout(
    client, database, actors, provider, monkeypatch
):
    _, source = await ready(database, actors, provider)
    attempted = []

    class Retried(RankProvider):
        async def embed_query(self, value, reserve):
            for _ in range(4):
                await reserve()
                attempted.append(1)
            return self.vector("POS")

    monkeypatch.setattr(settings, "semantic_search_max_provider_requests", 2)
    monkeypatch.setattr(retrieval_indexing, "make_provider", Retried)
    result = await query(client, actors, source)
    assert result.status_code == 503 and len(attempted) == 2, result.text

    class Waits(RankProvider):
        async def embed_query(self, value, reserve):
            await reserve()
            await asyncio.sleep(30)
            return self.vector("POS")

    monkeypatch.setattr(retrieval_indexing, "make_provider", Waits)
    monkeypatch.setattr(settings, "semantic_search_timeout_seconds", 1)
    result = await query(client, actors, source)
    assert result.status_code == 503, result.text
    monkeypatch.setattr(settings, "semantic_search_timeout_seconds", 45)
    monkeypatch.setattr(retrieval_indexing, "make_provider", RankProvider)
    assert (await query(client, actors, source)).status_code == 200


async def test_caller_cancellation_releases_capacity(database, actors, provider, monkeypatch):
    _, source = await ready(database, actors, provider)
    entered = asyncio.Event()

    class Waits(RankProvider):
        async def embed_query(self, value, reserve):
            await reserve()
            entered.set()
            await asyncio.sleep(30)
            return self.vector("POS")

    monkeypatch.setattr(retrieval_indexing, "make_provider", Waits)

    async def execute():
        async with database() as db:
            await service.search(
                db,
                actors["users"][0].id,
                source.ingestion_result["repository_id"],
                source.revision_id,
                {"query": "x", "mode": "semantic"},
            )

    assert service._SLOTS.acquire(blocking=False)
    task = asyncio.create_task(execute())
    try:
        await asyncio.wait_for(entered.wait(), 10)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert service._SLOTS.acquire(blocking=False)
        service._SLOTS.release()
    finally:
        service._SLOTS.release()
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
