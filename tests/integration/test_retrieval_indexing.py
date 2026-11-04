"""Real pgvector, authentication and workers; deterministic external provider."""

import asyncio
import copy
from datetime import timedelta

import pytest
from pydantic import SecretStr
from sqlalchemy import delete, func, select, update
from sqlalchemy.exc import DBAPIError

from core.config import settings
from core.models import IngestedSource, Profile, Review, SourceChunk, User
from core.models.retrieval import ChunkEmbedding, RetrievalIndex
from core.security import create_access_token
from core.services import job_queue as queue
from core.services import retrieval_indexing as service
from ingestion.embeddings.approved import EmbeddingError, approved_config
from tests.embedding_fixtures import FakeProvider, ingested, submitted
from workers.run import tick

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


def auth(actors, index=0):
    return {"Authorization": "Bearer " + create_access_token({"sub": actors["users"][index].id})}


async def test_real_index_worker_and_idempotent_api(client, database, actors, monkeypatch):
    monkeypatch.setattr(service, "make_provider", FakeProvider)
    ident, source = await submitted(database, actors)
    assert await tick(database, "index-worker")
    response = await client.get("/retrieval-indexes/" + ident, headers=auth(actors))
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["state"] == "ready" and result["job_status"] == "complete", result
    assert result["vector_count"] == result["expected_chunks"] == result["published_count"] == 2
    assert result["dimensions"] == 768 and result["reserved_requests"] == 1
    repeated = await client.post(
        "/retrieval-indexes", headers=auth(actors), json={"ingestion_id": source.id}
    )
    assert repeated.status_code == 200 and repeated.json()["index_id"] == ident, repeated.text
    assert '"embedding":' not in response.text and "cache_user" not in response.text
    async with database() as db:
        assert await db.scalar(select(func.count()).select_from(ChunkEmbedding)) == 2
        assert await db.scalar(select(func.count()).select_from(RetrievalIndex)) == 1


async def state(database, ident):
    async with database() as db:
        g = await db.get(RetrievalIndex, ident)
        j = await db.get(Review, g.job_id)
        n = await db.scalar(
            select(func.count()).select_from(ChunkEmbedding).where(ChunkEmbedding.index_id == ident)
        )
        return g.state, j.status, n, j.attempt_count, g.reserved_requests


async def test_two_users_two_revisions_and_foreign_ids(client, database, actors, monkeypatch):
    monkeypatch.setattr(service, "make_provider", FakeProvider)
    a, source = await submitted(database, actors)
    b, _ = await submitted(database, actors, 1)
    other = await ingested(database, actors, files={"a.py": b"other revision"})
    async with database() as db:
        c, _ = await service.submit(db, actors["users"][0].id, other.id)
    await asyncio.gather(tick(database, "w1"), tick(database, "w2"))
    assert await tick(database, "w3")
    for ident in (a, b, c):
        assert (await state(database, ident))[:3] == (
            "ready",
            "complete",
            2 if ident in (a, b) else 1,
        )
    assert (await client.get("/retrieval-indexes/" + b, headers=auth(actors))).status_code == 404
    assert (await client.get("/retrieval-indexes/" + a)).status_code == 401
    async with database() as db:
        foreign = await db.scalar(
            select(SourceChunk.id).where(SourceChunk.profile_id == actors["profiles"][1].id)
        )
        for bad_id in (b, foreign):
            with pytest.raises(Exception) as exc:
                await service.submit(db, actors["users"][0].id, bad_id)
            assert exc.value.status_code == 404
    async with database() as db:
        rows = (
            await db.execute(
                select(
                    ChunkEmbedding.index_id, ChunkEmbedding.revision_id, ChunkEmbedding.profile_id
                )
            )
        ).all()
        assert len(rows) == 5
        assert {row[1] for row in rows if row[0] == a} == {source.revision_id}
        assert {row[1] for row in rows if row[0] == c} == {other.revision_id}


async def test_provider_switching_and_failed_new_generation_preserves_ready(
    database, actors, monkeypatch
):
    monkeypatch.setattr(service, "make_provider", FakeProvider)
    first, source = await submitted(database, actors)
    await tick(database, "local")
    monkeypatch.setattr(settings, "embedding_hosted_source_allowed", True)
    monkeypatch.setattr(settings, "embedding_mistral_api_key", SecretStr("test-key-not-real"))
    monkeypatch.setattr(settings, "embedding_mistral_max_requests", 10)
    async with database() as db:
        second, _ = await service.submit(db, actors["users"][0].id, source.id, "mistral")
    await tick(database, "hosted-contract-double")
    async with database() as db:
        a, b = await db.get(RetrievalIndex, first), await db.get(RetrievalIndex, second)
        assert a.config_fingerprint != b.config_fingerprint and (a.dimensions, b.dimensions) == (
            768,
            1024,
        )
        assert (
            await db.scalar(select(ChunkEmbedding).where(ChunkEmbedding.index_id == second))
        ).dimensions == 1024
    monkeypatch.setattr(settings, "embedding_lmstudio_version", "artifact-v2")
    async with database() as db:
        third, _ = await service.submit(db, actors["users"][0].id, source.id, "lmstudio")

    class Fails(FakeProvider):
        async def embed_documents(self, texts, before_request):
            await before_request()
            raise EmbeddingError("provider_rejected")

    monkeypatch.setattr(service, "make_provider", Fails)
    await tick(database, "new-failure")
    assert (await state(database, third))[:3] == ("failed", "failed", 0)
    assert (await state(database, first))[:3] == ("ready", "complete", 2)
    assert (await state(database, second))[:3] == ("ready", "complete", 2)


@pytest.mark.parametrize(
    "mutation",
    ["inactive", "owner", "source_delete", "metadata", "manifest", "chunk", "configuration"],
)
async def test_revocation_and_source_changes_across_network(
    database, actors, monkeypatch, mutation
):
    entered, release = asyncio.Event(), asyncio.Event()

    class Paused(FakeProvider):
        async def embed_documents(self, texts, before_request):
            await before_request()
            entered.set()
            await release.wait()
            return [[1.0] + [0.0] * (self.config.dimensions - 1) for _ in texts]

    monkeypatch.setattr(service, "make_provider", Paused)
    ident, source = await submitted(database, actors)
    task = asyncio.create_task(tick(database, "paused"))
    try:
        await asyncio.wait_for(entered.wait(), 10)
        async with database() as db:
            if mutation == "configuration":
                monkeypatch.setattr(settings, "embedding_lmstudio_version", "revoked-config-v2")
            elif mutation == "inactive":
                await db.execute(
                    update(User).where(User.id == actors["users"][0].id).values(is_active=False)
                )
            elif mutation == "owner":
                await db.execute(
                    update(Profile)
                    .where(Profile.id == actors["profiles"][0].id)
                    .values(user_id=actors["users"][1].id)
                )
            elif mutation == "source_delete":
                await db.execute(
                    delete(IngestedSource).where(IngestedSource.revision_id == source.revision_id)
                )
            elif mutation == "metadata":
                row = await db.scalar(
                    select(IngestedSource)
                    .where(IngestedSource.revision_id == source.revision_id)
                    .limit(1)
                )
                row.source_metadata = {**row.source_metadata, "changed": True}
            elif mutation == "manifest":
                proof = copy.deepcopy(source.ingestion_result)
                proof["verified"] = False
                await db.execute(
                    update(Review).where(Review.id == source.id).values(ingestion_result=proof)
                )
            else:
                row = await db.scalar(
                    select(SourceChunk)
                    .join(IngestedSource)
                    .where(IngestedSource.revision_id == source.revision_id)
                    .limit(1)
                )
                row.text = "changed chunk"
            await db.commit()
    finally:
        release.set()
    await asyncio.wait_for(task, 10)
    assert (await state(database, ident))[:3] == ("failed", "failed", 0)


@pytest.mark.parametrize("when", ["pending", "during", "ready"])
async def test_cancellation_lifecycle(client, database, actors, monkeypatch, when):
    entered, release = asyncio.Event(), asyncio.Event()

    class Paused(FakeProvider):
        async def embed_documents(self, texts, before_request):
            await before_request()
            entered.set()
            await release.wait()
            return [[1.0] + [0.0] * (self.config.dimensions - 1) for _ in texts]

    monkeypatch.setattr(service, "make_provider", Paused if when == "during" else FakeProvider)
    ident, _ = await submitted(database, actors)
    async with database() as db:
        job_id = (await db.get(RetrievalIndex, ident)).job_id
    task = asyncio.create_task(tick(database, "cancel-worker")) if when == "during" else None
    if task:
        await asyncio.wait_for(entered.wait(), 10)
    if when == "ready":
        await tick(database, "done")
    try:
        response = await client.post("/jobs/" + job_id + "/cancel", headers=auth(actors))
        assert response.status_code == (409 if when == "ready" else 200), response.text
    finally:
        release.set()
    if task:
        await asyncio.wait_for(task, 10)
    assert (await state(database, ident))[:3] == (
        ("ready", "complete", 2) if when == "ready" else ("cancelled", "cancelled", 0)
    )
    assert (
        await client.post("/jobs/" + job_id + "/cancel", headers=auth(actors, 1))
    ).status_code == 404


async def test_expired_lease_stale_fence_and_replacement(database, actors, monkeypatch):
    from workers.run import run_attempt

    entered, release = asyncio.Event(), asyncio.Event()

    class Paused(FakeProvider):
        async def embed_documents(self, texts, before_request):
            await before_request()
            entered.set()
            await release.wait()
            return [[1.0] + [0.0] * (self.config.dimensions - 1) for _ in texts]

    monkeypatch.setattr(service, "make_provider", Paused)
    monkeypatch.setattr(settings, "job_retry_base_seconds", 0)
    ident, _ = await submitted(database, actors)
    task = asyncio.create_task(tick(database, "stale"))
    try:
        await asyncio.wait_for(entered.wait(), 10)
        async with database() as db:
            generation = await db.get(RetrievalIndex, ident)
            job_id = generation.job_id
            await db.execute(
                update(Review)
                .where(Review.id == job_id)
                .values(lease_expires_at=queue.now() - timedelta(seconds=1))
            )
            await db.commit()
            assert await queue.recover(db) == 1
            replacement = await queue.claim(db, "replacement")
            assert replacement and replacement.job_id == job_id
    finally:
        release.set()
    await asyncio.wait_for(task, 10)
    assert (await state(database, ident))[:3] == ("building", "processing", 0)
    monkeypatch.setattr(service, "make_provider", FakeProvider)
    await run_attempt(database, replacement)
    assert (await state(database, ident))[:4] == ("ready", "complete", 2, 2)


@pytest.mark.parametrize("retryable", [True, False])
async def test_batch_failure_cleanup_and_retry_exhaustion(database, actors, monkeypatch, retryable):
    monkeypatch.setattr(settings, "embedding_batch_size", 1)
    monkeypatch.setattr(settings, "job_retry_base_seconds", 0)
    monkeypatch.setattr(settings, "job_max_attempts", 2)

    class FailsSecond(FakeProvider):
        async def embed_documents(self, texts, before_request):
            self.number = getattr(self, "number", 0) + 1
            if self.number == 2:
                await before_request()
                raise EmbeddingError("provider_unavailable", retryable)
            return await super().embed_documents(texts, before_request)

    monkeypatch.setattr(service, "make_provider", FailsSecond)
    ident, _ = await submitted(database, actors)
    await tick(database, "first")
    assert (await state(database, ident))[:3] == (
        ("pending", "pending", 0) if retryable else ("failed", "failed", 0)
    )
    if retryable:
        await tick(database, "exhausted")
        assert (await state(database, ident))[:4] == ("failed", "failed", 0, 2)


async def test_durable_request_budget_no_automatic_fallback(database, actors, monkeypatch):
    monkeypatch.setattr(settings, "embedding_lmstudio_max_requests", 1)
    monkeypatch.setattr(settings, "job_retry_base_seconds", 0)
    seen = []

    class Fails(FakeProvider):
        async def embed_documents(self, texts, before_request):
            await before_request()
            seen.append(self.config.provider)
            raise EmbeddingError("provider_unavailable", True)

    monkeypatch.setattr(service, "make_provider", Fails)
    ident, _ = await submitted(database, actors)
    await tick(database, "first")
    await tick(database, "budget")
    assert seen == ["lmstudio"]
    assert (await state(database, ident))[:3] == ("failed", "failed", 0)
    assert (await state(database, ident))[4] == 1


async def test_duplicate_submissions_and_workers(database, actors, monkeypatch):
    monkeypatch.setattr(service, "make_provider", FakeProvider)
    source = await ingested(database, actors)

    async def submit():
        async with database() as db:
            return await service.submit(db, actors["users"][0].id, source.id)

    results = await asyncio.gather(submit(), submit())
    assert results[0][0] == results[1][0] and sum(r[1] for r in results) == 1
    assert sorted(await asyncio.gather(tick(database, "w1"), tick(database, "w2"))) == [False, True]
    assert (await state(database, results[0][0]))[:3] == ("ready", "complete", 2)


@pytest.mark.parametrize("files", [{"README.md": b""}, {"a.py": b"x" * 3000}])
async def test_empty_and_oversized_canonical_chunks(client, database, actors, monkeypatch, files):
    monkeypatch.setattr(service, "make_provider", FakeProvider)
    source = await ingested(database, actors, files=files)
    result = await client.post(
        "/retrieval-indexes", headers=auth(actors), json={"ingestion_id": source.id}
    )
    if "README.md" in files:
        assert result.status_code == 202, result.text
        ident = result.json()["index_id"]
        await tick(database, "empty")
        assert (await state(database, ident))[:3] == ("ready", "complete", 0)
    else:
        assert result.status_code == 413, result.text
        async with database() as db:
            assert await db.scalar(select(func.count()).select_from(RetrievalIndex)) == 0
            assert (
                await db.scalar(
                    select(func.count())
                    .select_from(Review)
                    .where(Review.job_type == "retrieval_index")
                )
                == 0
            )


async def test_partial_publication_rejected_atomically(database, actors):
    ident, _ = await submitted(database, actors)
    async with database() as db:
        attempt = await queue.claim(db, "partial")
        generation = await db.get(RetrievalIndex, ident)
        job = await db.get(Review, generation.job_id)
        generation.state = "ready"
        generation.published_count = generation.expected_chunks
        generation.published_at = queue.now()
        queue.complete(job, attempt)
        with pytest.raises(DBAPIError, match="coverage incomplete"):
            await db.commit()
        await db.rollback()
    assert (await state(database, ident))[:3] == ("building", "processing", 0)


@pytest.mark.parametrize("mutation", ["chunk", "metadata", "delete", "proof"])
async def test_published_generation_invalidated_when_sources_change(
    database, actors, monkeypatch, mutation
):
    monkeypatch.setattr(service, "make_provider", FakeProvider)
    ident, source = await submitted(database, actors)
    await tick(database, "ready")
    async with database() as db:
        if mutation == "proof":
            proof = copy.deepcopy(source.ingestion_result)
            proof["verified"] = False
            await db.execute(
                update(Review).where(Review.id == source.id).values(ingestion_result=proof)
            )
        elif mutation == "metadata":
            row = await db.scalar(
                select(IngestedSource)
                .where(IngestedSource.revision_id == source.revision_id)
                .limit(1)
            )
            row.source_metadata = {**row.source_metadata, "changed": True}
        else:
            chunk = await db.scalar(
                select(SourceChunk)
                .join(IngestedSource)
                .where(IngestedSource.revision_id == source.revision_id)
                .limit(1)
            )
            if mutation == "delete":
                await db.delete(chunk)
            else:
                chunk.text = "mutation"
        await db.commit()
    assert (await state(database, ident))[:2] == ("failed", "complete")


async def test_validation_and_unexpected_errors_do_not_leak(
    client, database, actors, monkeypatch, capsys
):
    ident, source = await submitted(database, actors)
    bad = await client.post(
        "/retrieval-indexes",
        headers=auth(actors),
        json={"ingestion_id": source.id, "api_key": "PRIVATE_SECRET_PAYLOAD"},
    )
    assert bad.status_code == 422 and "PRIVATE_SECRET_PAYLOAD" not in bad.text

    async def broken(*args):
        raise ValueError("PRIVATE_SOURCE_VECTOR_SECRET")

    monkeypatch.setattr(service, "status", broken)
    result = await client.get("/retrieval-indexes/" + ident, headers=auth(actors))
    assert result.status_code == 500 and "PRIVATE_SOURCE_VECTOR_SECRET" not in result.text
    assert "PRIVATE_SOURCE_VECTOR_SECRET" not in capsys.readouterr().out


@pytest.mark.parametrize(
    "fault",
    [
        "profile",
        "source",
        "chunk",
        "revision",
        "dimension",
        "length",
        "configuration",
        "duplicate",
        "digest",
        "fence",
    ],
)
async def test_vector_constraints_and_publication_guard(database, actors, fault):
    from uuid import uuid4

    ident, _ = await submitted(database, actors, files={"a.py": b"cache"})
    other = await ingested(database, actors, 1, files={"b.py": b"private"})
    async with database() as db:
        attempt = await queue.claim(db, "constraints")
        g = await db.get(RetrievalIndex, ident)
        item = g.chunk_manifest[0]
        foreign = await db.scalar(
            select(SourceChunk).where(SourceChunk.profile_id == actors["profiles"][1].id)
        )
        fields = dict(
            index_id=g.id,
            chunk_id=item["chunk_id"],
            source_id=item["source_id"],
            revision_id=g.revision_id,
            profile_id=g.profile_id,
            config_fingerprint=g.config_fingerprint,
            dimensions=g.dimensions,
            attempt_token=attempt.token,
            input_digest=item["input_digest"],
            embedding=[1.0] + [0.0] * 767,
        )
        if fault == "profile":
            fields["profile_id"] = actors["profiles"][1].id
        if fault == "source":
            fields["source_id"] = foreign.source_id
        if fault == "chunk":
            fields["chunk_id"] = foreign.id
            fields["source_id"] = foreign.source_id
        if fault == "revision":
            fields["revision_id"] = other.revision_id
        if fault == "dimension":
            fields["dimensions"] = 1024
            fields["embedding"] = [1.0] + [0.0] * 1023
        if fault == "length":
            fields["embedding"].pop()
        if fault == "configuration":
            fields["config_fingerprint"] = "f" * 64
        if fault == "digest":
            fields["input_digest"] = "0" * 64
        if fault == "fence":
            fields["attempt_token"] = str(uuid4())
        db.add(ChunkEmbedding(**fields))
        if fault == "duplicate":
            db.add(ChunkEmbedding(**fields))
        if fault in {"digest", "fence"}:
            await db.commit()
            generation = await db.get(RetrievalIndex, ident)
            job = await db.get(Review, generation.job_id)
            generation.state = "ready"
            generation.published_count = 1
            generation.published_at = queue.now()
            queue.complete(job, attempt)
        with pytest.raises(DBAPIError):
            await db.commit()
        await db.rollback()
    assert (await state(database, ident))[:2] == ("building", "processing")
    await queue.fail(database, attempt, ValueError("constraint fixture cleanup"))
    assert (await state(database, ident))[:3] == ("failed", "failed", 0)


async def test_pgvector_exact_distance_in_owned_generation(database, actors, monkeypatch):
    from sqlalchemy import cast, literal

    from core.models.retrieval import Vector

    monkeypatch.setattr(service, "make_provider", FakeProvider)
    ident, _ = await submitted(database, actors)
    await tick(database, "distance")
    async with database() as db:
        g, _, _ = await service.owned(db, ident, actors["users"][0].id)
        vector = await db.scalar(
            select(ChunkEmbedding.embedding).where(ChunkEmbedding.index_id == ident).limit(1)
        )
        distance = ChunkEmbedding.embedding.op("OPERATOR(public.<=>)")(
            cast(literal(vector, type_=Vector()), Vector())
        )
        rows = (
            await db.execute(
                select(ChunkEmbedding.chunk_id, distance.label("distance"))
                .where(
                    ChunkEmbedding.index_id == ident,
                    ChunkEmbedding.profile_id == g.profile_id,
                    ChunkEmbedding.revision_id == g.revision_id,
                )
                .order_by(distance, ChunkEmbedding.chunk_id)
            )
        ).all()
        assert len(rows) == 2 and rows[0].distance == 0
        assert rows[0].distance <= rows[1].distance


async def test_downgrade_refuses_index_history(database, actors):
    from alembic.config import Config

    from alembic import command

    ident, _ = await submitted(database, actors)
    with pytest.raises(RuntimeError, match="destroy indexing history"):
        async with database.kw["bind"].begin() as connection:

            def downgrade(conn):
                cfg = Config("alembic.ini")
                cfg.attributes["connection"] = conn
                command.downgrade(cfg, "006")

            await connection.run_sync(downgrade)
    assert (await state(database, ident))[:3] == ("pending", "pending", 0)


@pytest.mark.parametrize("fault", ["expired", "cancelled"])
async def test_database_publication_checks_lease_and_cancellation(database, actors, fault):
    ident, _ = await submitted(database, actors, files={"a.py": b"cache"})
    async with database() as db:
        attempt = await queue.claim(db, "commit-guard")
        g = await db.get(RetrievalIndex, ident)
        job = await db.get(Review, g.job_id)
        item = g.chunk_manifest[0]
        db.add(
            ChunkEmbedding(
                index_id=g.id,
                chunk_id=item["chunk_id"],
                source_id=item["source_id"],
                revision_id=g.revision_id,
                profile_id=g.profile_id,
                config_fingerprint=g.config_fingerprint,
                dimensions=g.dimensions,
                attempt_token=attempt.token,
                input_digest=item["input_digest"],
                embedding=[1.0] + [0.0] * 767,
            )
        )
        await db.flush()
        queue.complete(job, attempt)
        if fault == "expired":
            job.lease_expires_at = queue.now() - timedelta(seconds=1)
        else:
            job.cancel_requested_at = queue.now()
        g.state = "ready"
        g.published_count = 1
        g.published_at = queue.now()
        with pytest.raises(DBAPIError, match="publication job"):
            await db.commit()
        await db.rollback()
    assert (await state(database, ident))[:3] == ("building", "processing", 0)


async def test_database_ready_guard_rejects_duplicate_manifest(database, actors):
    from dataclasses import asdict
    from uuid import uuid4

    from core.services.keyword_retrieval import SearchLimits, _snapshot

    source = await ingested(database, actors)
    cfg = approved_config("lmstudio")
    async with database() as db:
        snap = await _snapshot(
            db,
            actors["users"][0].id,
            source.ingestion_result["repository_id"],
            source.revision_id,
            SearchLimits.configured(),
        )
        entries = service.manifest(snap, cfg)
        job = Review(
            id=str(uuid4()),
            profile_id=actors["profiles"][0].id,
            revision_id=source.revision_id,
            job_type="retrieval_index",
            requested_repository="example/project",
            status="pending",
        )
        db.add(job)
        await db.flush()
        g = RetrievalIndex(
            id=str(uuid4()),
            profile_id=job.profile_id,
            revision_id=source.revision_id,
            ingestion_id=source.id,
            job_id=job.id,
            provider=cfg.provider,
            model=cfg.model,
            dimensions=cfg.dimensions,
            metric=cfg.metric,
            config_fingerprint=cfg.fingerprint,
            config=asdict(cfg),
            manifest_fingerprint=snap.digest,
            chunk_manifest=[entries[0], entries[0]],
            expected_chunks=2,
            state="pending",
        )
        db.add(g)
        await db.commit()
        attempt = await queue.claim(db, "duplicate-manifest")
        await db.refresh(g)
        for item in entries:
            db.add(
                ChunkEmbedding(
                    index_id=g.id,
                    chunk_id=item["chunk_id"],
                    source_id=item["source_id"],
                    revision_id=g.revision_id,
                    profile_id=g.profile_id,
                    config_fingerprint=g.config_fingerprint,
                    dimensions=g.dimensions,
                    attempt_token=attempt.token,
                    input_digest=item["input_digest"],
                    embedding=[1.0] + [0.0] * 767,
                )
            )
        await db.flush()
        job = await db.get(Review, g.job_id)
        queue.complete(job, attempt)
        g.state = "ready"
        g.published_count = 2
        g.published_at = queue.now()
        with pytest.raises(DBAPIError, match="manifest duplicate"):
            await db.commit()
        await db.rollback()


async def test_source_pagination_preserves_case_sensitive_path_order(client, database, actors):
    job = await ingested(
        database,
        actors,
        files={"README.md": b"README", "Z.py": b"last uppercase", "a.py": b"first lowercase"},
    )
    paths = []
    for page in (1, 2, 3):
        response = await client.get(
            "/repository-ingestions/" + job.id + "/sources",
            headers=auth(actors),
            params={"page": page, "page_size": 1},
        )
        assert response.status_code == 200, response.text
        assert response.json()["total"] == 3
        paths.append(response.json()["items"][0]["path"])
    assert paths == ["README.md", "Z.py", "a.py"]
