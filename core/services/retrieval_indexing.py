"""Durable canonical revision indexing; Review remains the only execution queue."""

from dataclasses import asdict
from uuid import uuid4

from fastapi import HTTPException
from sqlalchemy import delete, func, select

from core.config import settings
from core.models import Profile, Review
from core.models.repository import Repository, RepositoryRevision
from core.models.retrieval import ChunkEmbedding, RetrievalIndex
from core.services import job_queue as queue
from core.services.authorization import require_active_user
from core.services.keyword_retrieval import SearchLimits, _snapshot
from ingestion.embeddings.approved import (
    EmbeddingError,
    LMStudioEmbeddingProvider,
    MistralEmbeddingProvider,
    approved_config,
    validate_vectors,
)


def make_provider(config):
    adapter = (
        LMStudioEmbeddingProvider if config.provider == "lmstudio" else MistralEmbeddingProvider
    )
    return adapter(config)


def batches(snapshot, config):
    if len(snapshot.chunks) > settings.embedding_max_chunks:
        raise EmbeddingError("corpus_limit")
    result, current, tokens, total = [], [], 0, 0
    for chunk, _ in snapshot.chunks:
        bound = config.bound(chunk.text)
        total += bound
        if total > settings.embedding_max_index_tokens:
            raise EmbeddingError("corpus_limit")
        if current and (
            len(current) >= config.batch_size or tokens + bound > config.max_batch_tokens
        ):
            result.append(current)
            current, tokens = [], 0
        if bound > config.max_batch_tokens:
            raise EmbeddingError("batch_limit")
        current.append(chunk)
        tokens += bound
    if current:
        result.append(current)
    if len(result) > config.max_requests:
        raise EmbeddingError("request_budget")
    return result


def manifest(snapshot, config):
    return [
        {
            "chunk_id": str(chunk.id),
            "source_id": str(chunk.source_id),
            "input_digest": config.digest(chunk.text),
        }
        for chunk, _ in snapshot.chunks
    ]


async def owned(db, index_id, user_id, lock=False):
    await require_active_user(db, user_id)
    query = (
        select(RetrievalIndex, Repository.id, RepositoryRevision.commit_sha)
        .join(Profile, Profile.id == RetrievalIndex.profile_id)
        .join(
            RepositoryRevision,
            (RepositoryRevision.id == RetrievalIndex.revision_id)
            & (RepositoryRevision.profile_id == Profile.id),
        )
        .join(
            Repository,
            (Repository.id == RepositoryRevision.repository_id)
            & (Repository.profile_id == Profile.id),
        )
        .where(RetrievalIndex.id == str(index_id), Profile.user_id == str(user_id))
    )
    if lock:
        query = query.with_for_update(of=RetrievalIndex).execution_options(populate_existing=True)
    result = (await db.execute(query)).first()
    if not result:
        raise HTTPException(404, "Retrieval index not found")
    return result


async def submit(db, user_id, ingestion_id, configuration="default"):
    try:
        await require_active_user(db, user_id)
        source = (
            await db.execute(
                select(Review, Repository.id, Repository.full_name)
                .join(Profile, Profile.id == Review.profile_id)
                .join(
                    RepositoryRevision,
                    (RepositoryRevision.id == Review.revision_id)
                    & (RepositoryRevision.profile_id == Profile.id),
                )
                .join(
                    Repository,
                    (Repository.id == RepositoryRevision.repository_id)
                    & (Repository.profile_id == Profile.id),
                )
                .where(
                    Review.id == str(ingestion_id),
                    Profile.user_id == str(user_id),
                    Review.job_type == "github_ingestion",
                )
            )
        ).first()
        if not source:
            raise HTTPException(404, "Ingestion not found")
        job, repository_id, name = source
        if (
            job.status != "complete"
            or not isinstance(job.ingestion_result, dict)
            or job.ingestion_result.get("verified") is not True
            or job.ingestion_result.get("policy") != "github-text-v1"
        ):
            raise HTTPException(409, "Completed verified ingestion required")
        try:
            config = approved_config(configuration)
        except EmbeddingError:
            raise HTTPException(
                409, "Embedding configuration unavailable or hosted transmission not permitted"
            ) from None
        snapshot = await _snapshot(
            db, str(user_id), repository_id, job.revision_id, SearchLimits.configured()
        )
        if job.ingestion_result.get("commit_sha") != snapshot.sha:
            raise HTTPException(409, "Ingestion snapshot inconsistent")
        batches(snapshot, config)  # all input/budget limits before creating any job
        existing = await db.scalar(
            select(RetrievalIndex).where(
                RetrievalIndex.profile_id == job.profile_id,
                RetrievalIndex.revision_id == job.revision_id,
                RetrievalIndex.config_fingerprint == config.fingerprint,
                RetrievalIndex.state.in_(["pending", "building", "ready"]),
            )
        )
        if existing:
            if existing.manifest_fingerprint != snapshot.digest:
                raise HTTPException(409, "Existing index snapshot changed")
            ident = existing.id
            await db.rollback()
            return ident, False
        execution = Review(
            id=str(uuid4()),
            profile_id=job.profile_id,
            revision_id=job.revision_id,
            job_type="retrieval_index",
            requested_repository=name,
            status="pending",
            max_attempts=settings.job_max_attempts,
            sections=None,
            overall_score=None,
        )
        db.add(execution)
        await db.flush()
        index = RetrievalIndex(
            id=str(uuid4()),
            profile_id=job.profile_id,
            revision_id=job.revision_id,
            ingestion_id=job.id,
            job_id=execution.id,
            provider=config.provider,
            model=config.model,
            dimensions=config.dimensions,
            metric=config.metric,
            config_fingerprint=config.fingerprint,
            config=asdict(config),
            manifest_fingerprint=snapshot.digest,
            chunk_manifest=manifest(snapshot, config),
            expected_chunks=len(snapshot.chunks),
            state="pending",
        )
        db.add(index)
        await db.flush()
        ident = index.id
        await db.commit()
        queue.event("submitted", execution)
        return ident, True
    except EmbeddingError:
        await db.rollback()
        raise HTTPException(413, "Embedding input or request budget exceeded") from None
    except BaseException:
        await db.rollback()
        raise


async def status(db, user_id, index_id):
    index, repository_id, sha = await owned(db, index_id, user_id)
    from core.services.keyword_retrieval import _scope

    await _scope(db, str(user_id), repository_id, index.revision_id)
    job = await queue.owned(db, index.job_id, user_id)
    count = await db.scalar(
        select(func.count())
        .select_from(ChunkEmbedding)
        .where(
            ChunkEmbedding.index_id == index.id,
            ChunkEmbedding.profile_id == index.profile_id,
            ChunkEmbedding.revision_id == index.revision_id,
        )
    )
    state = index.state
    error = job.error_message
    if state == "failed" and job.status == "complete":
        error = "Published source snapshot changed; rebuild required"
    return dict(
        index_id=index.id,
        job_id=job.id,
        repository_id=repository_id,
        revision_id=index.revision_id,
        commit_sha=sha,
        provider=index.provider,
        model=index.model,
        dimensions=index.dimensions,
        metric=index.metric,
        configuration_fingerprint=index.config_fingerprint,
        state=state,
        job_status=job.status,
        expected_chunks=index.expected_chunks,
        vector_count=count,
        published_count=index.published_count,
        reserved_requests=index.reserved_requests,
        error_message=error,
        created_at=index.created_at,
        completed_at=job.completed_at,
        published_at=index.published_at,
    )


async def sync_job_state(db, job):
    """Called under the existing queue job lock. No second queue or independent retries."""
    if job.job_type != "retrieval_index":
        return
    index = await db.scalar(
        select(RetrievalIndex).where(RetrievalIndex.job_id == job.id).with_for_update()
    )
    if index is None:
        raise ValueError("Index job association missing")
    if index.state == "ready":
        return
    await db.execute(delete(ChunkEmbedding).where(ChunkEmbedding.index_id == index.id))
    index.state = {
        "processing": "building",
        "pending": "pending",
        "failed": "failed",
        "cancelled": "cancelled",
    }[job.status]
    index.attempt_token = job.execution_token if job.status == "processing" else None
    index.published_count, index.published_at = 0, None


async def _locked(db, attempt, index_id):
    job = await queue.owned(db, attempt.job_id, attempt.user_id, lock=True)
    queue.assert_attempt(job, attempt)
    index, _, _ = await owned(db, index_id, attempt.user_id, lock=True)
    if (
        job.job_type != "retrieval_index"
        or index.job_id != job.id
        or index.state != "building"
        or index.attempt_token != attempt.token
    ):
        raise queue.LostAttemptError()
    return job, index


async def process(db, job_id, user_id, *, attempt=None):
    if attempt is None:
        return await queue.direct(
            db, job_id, user_id, lambda claimed: process(db, job_id, user_id, attempt=claimed)
        )
    try:
        await attempt.checkpoint("validating_index", 0)
        index_id = await db.scalar(
            select(RetrievalIndex.id)
            .join(Profile, Profile.id == RetrievalIndex.profile_id)
            .where(RetrievalIndex.job_id == str(job_id), Profile.user_id == str(user_id))
        )
        if index_id is None:
            raise HTTPException(404, "Retrieval index not found")
        index, repository_id, _ = await owned(db, index_id, user_id)
        config = approved_config(index.provider)
        if config.fingerprint != index.config_fingerprint or asdict(config) != index.config:
            raise EmbeddingError("configuration_changed")
        initial = await _snapshot(
            db, str(user_id), repository_id, index.revision_id, SearchLimits.configured()
        )
        if (
            initial.digest != index.manifest_fingerprint
            or manifest(initial, config) != index.chunk_manifest
        ):
            raise ValueError("Index manifest changed")
        plan = batches(initial, config)
        await db.rollback()  # no row locks or transactions throughout any provider call
        provider = make_provider(config)

        async def reserve():
            async with attempt.factory() as reservation:
                _, generation = await _locked(reservation, attempt, index_id)
                if approved_config(config.provider).fingerprint != config.fingerprint:
                    raise EmbeddingError("configuration_changed")
                if generation.reserved_requests >= config.max_requests:
                    raise EmbeddingError("request_budget")
                generation.reserved_requests += 1
                await reservation.commit()

        for number, batch in enumerate(plan):
            await attempt.checkpoint("embedding_chunks", int(number * 90 / max(1, len(plan))))
            vectors = validate_vectors(
                await provider.embed_documents([c.text for c in batch], reserve),
                len(batch),
                config.dimensions,
            )
            await attempt.checkpoint("staging_vectors", int((number + 1) * 90 / max(1, len(plan))))
            _, generation = await _locked(db, attempt, index_id)
            for chunk, vector in zip(batch, vectors, strict=True):
                db.add(
                    ChunkEmbedding(
                        index_id=index_id,
                        chunk_id=str(chunk.id),
                        source_id=str(chunk.source_id),
                        revision_id=initial.revision_id,
                        profile_id=generation.profile_id,
                        config_fingerprint=config.fingerprint,
                        dimensions=config.dimensions,
                        attempt_token=attempt.token,
                        input_digest=config.digest(chunk.text),
                        embedding=vector,
                    )
                )
            await db.commit()
        await attempt.checkpoint("publishing_index", 95)
        if approved_config(config.provider).fingerprint != config.fingerprint:
            raise EmbeddingError("configuration_changed")
        current = await _snapshot(
            db, str(user_id), initial.repository_id, initial.revision_id, SearchLimits.configured()
        )
        job, generation = await _locked(db, attempt, index_id)
        if (
            current.digest != generation.manifest_fingerprint
            or manifest(current, config) != generation.chunk_manifest
        ):
            raise ValueError("Index input changed")
        generation.state, generation.published_count = "ready", len(current.chunks)
        generation.published_at = queue.now()
        queue.complete(job, attempt)
        await (
            db.commit()
        )  # deferred DB guard verifies complete manifest, dimensions and execution token
        queue.event("completed", job)
    except BaseException:
        await db.rollback()
        raise
