"""Read-only exact semantic retrieval over authorized canonical pgvector generations."""

import asyncio
import hashlib
import json
import math
import threading
from dataclasses import asdict, dataclass
from uuid import UUID

from fastapi import HTTPException
from pydantic import ValidationError
from sqlalchemy import Float, Text, and_, cast, func, literal, select
from sqlalchemy.exc import DBAPIError

from api.schemas.search import (
    SemanticMatch,
    SemanticQueryMetadata,
    SemanticSearchRequest,
    SemanticSearchResponse,
)
from core.config import settings
from core.models import IngestedSource, Profile, Review, SourceChunk, User
from core.models.repository import Repository, RepositoryRevision
from core.models.retrieval import ChunkEmbedding, RetrievalIndex, Vector
from core.services import retrieval_indexing
from core.services.authorization import require_active_user
from core.services.keyword_retrieval import SearchLimits, Snapshot, _scope, _size, _snapshot
from ingestion.embeddings.approved import EmbeddingError, approved_config, validate_vectors

VERSION = "pgvector-exact-cosine-v1"
_SLOTS = threading.BoundedSemaphore(settings.semantic_search_concurrency)


@dataclass(frozen=True)
class View:
    snapshot: Snapshot
    index_id: str
    profile_id: str
    signature: str


def conflict():
    return HTTPException(409, "Compatible complete ready semantic index required")


def bound(ok):
    if not ok:
        raise HTTPException(413, "Semantic retrieval resource limit exceeded")


def configuration(name):
    config = approved_config(name)
    if config.provider == "mistral" and not settings.semantic_search_hosted_query_allowed:
        raise EmbeddingError("hosted_query_not_permitted")
    return config


def generations(uid, repo, rev):
    return (
        select(RetrievalIndex)
        .join(Profile, Profile.id == RetrievalIndex.profile_id)
        .join(
            RepositoryRevision,
            and_(
                RepositoryRevision.id == RetrievalIndex.revision_id,
                RepositoryRevision.profile_id == Profile.id,
            ),
        )
        .join(
            Repository,
            and_(
                Repository.id == RepositoryRevision.repository_id,
                Repository.profile_id == Profile.id,
            ),
        )
        .join(User, User.id == Profile.user_id)
        .where(
            User.id == uid,
            User.is_active.is_(True),
            Repository.id == repo,
            RepositoryRevision.id == rev,
        )
    )


def owned_vectors(uid, repo, rev, index_id):
    # The owned GENERATION constrains even inconsistent rows during the integrity audit.
    return (
        select(ChunkEmbedding)
        .join(RetrievalIndex, RetrievalIndex.id == ChunkEmbedding.index_id)
        .join(Profile, Profile.id == RetrievalIndex.profile_id)
        .join(
            RepositoryRevision,
            and_(
                RepositoryRevision.id == RetrievalIndex.revision_id,
                RepositoryRevision.profile_id == Profile.id,
            ),
        )
        .join(
            Repository,
            and_(
                Repository.id == RepositoryRevision.repository_id,
                Repository.profile_id == Profile.id,
            ),
        )
        .join(User, User.id == Profile.user_id)
        .where(
            User.id == uid,
            User.is_active.is_(True),
            Repository.id == repo,
            RepositoryRevision.id == rev,
            RetrievalIndex.id == index_id,
        )
    )


async def view(db, uid, repo, rev, config, limits, index_id=None):
    snapshot = await _snapshot(db, uid, repo, rev, limits)
    query = generations(uid, repo, rev).where(
        RetrievalIndex.state == "ready", RetrievalIndex.config_fingerprint == config.fingerprint
    )
    if index_id is not None:
        query = query.where(RetrievalIndex.id == index_id)
    query = query.join(
        Review,
        and_(
            Review.id == RetrievalIndex.job_id,
            Review.profile_id == RetrievalIndex.profile_id,
            Review.revision_id == RetrievalIndex.revision_id,
        ),
    )
    keys = (
        await db.execute(
            query.with_only_columns(
                RetrievalIndex.id,
                _size(db, cast(RetrievalIndex.config, Text)).label("config_size"),
                _size(db, cast(RetrievalIndex.chunk_manifest, Text)).label("manifest_size"),
            )
            .order_by(RetrievalIndex.id)
            .limit(2)
            .with_for_update(read=True, of=(RetrievalIndex, Review))
        )
    ).all()
    if len(keys) != 1:
        raise conflict()
    bound(
        keys[0].config_size <= limits.metadata_bytes
        and keys[0].manifest_size <= limits.metadata_bytes
    )
    generation, job = (
        await db.execute(
            query.with_only_columns(RetrievalIndex, Review)
            .where(RetrievalIndex.id == keys[0].id)
            .execution_options(populate_existing=True)
        )
    ).one()
    if (
        generation.provider != config.provider
        or generation.model != config.model
        or generation.dimensions != config.dimensions
        or generation.metric != "cosine"
        or generation.config != asdict(config)
        or generation.manifest_fingerprint != snapshot.digest
        or generation.expected_chunks != len(snapshot.chunks)
        or generation.published_count != generation.expected_chunks
        or generation.published_at is None
        or generation.chunk_manifest != retrieval_indexing.manifest(snapshot, config)
        or job.job_type != "retrieval_index"
        or job.status != "complete"
        or job.cancel_requested_at
        or not generation.attempt_token
        or generation.attempt_token != job.execution_token
        or job.completed_at is None
    ):
        raise conflict()
    vectors = owned_vectors(uid, repo, rev, generation.id)
    rows = (
        (
            await db.execute(
                vectors.with_only_columns(
                    ChunkEmbedding.chunk_id,
                    ChunkEmbedding.source_id,
                    ChunkEmbedding.profile_id,
                    ChunkEmbedding.revision_id,
                    ChunkEmbedding.config_fingerprint,
                    ChunkEmbedding.dimensions,
                    ChunkEmbedding.attempt_token,
                    ChunkEmbedding.input_digest,
                    func.public.vector_dims(ChunkEmbedding.embedding).label("actual_dimensions"),
                    func.public.vector_norm(ChunkEmbedding.embedding).label("norm"),
                    func.encode(
                        func.sha256(func.convert_to(cast(ChunkEmbedding.embedding, Text), "UTF8")),
                        "hex",
                    ).label("vector_digest"),
                )
                .order_by(ChunkEmbedding.chunk_id)
                .limit(limits.chunks + 1)
                .with_for_update(read=True, of=ChunkEmbedding)
            )
        )
        .mappings()
        .all()
    )
    bound(len(rows) <= limits.chunks)
    canonical = {str(c.id): c for c, _ in snapshot.chunks}
    if len(rows) != len(canonical) or {r["chunk_id"] for r in rows} != set(canonical):
        raise conflict()
    for row in rows:
        chunk = canonical[row["chunk_id"]]
        if (
            row["source_id"] != str(chunk.source_id)
            or row["profile_id"] != generation.profile_id
            or row["revision_id"] != rev
            or row["config_fingerprint"] != config.fingerprint
            or row["dimensions"] != config.dimensions
            or row["actual_dimensions"] != config.dimensions
            or row["attempt_token"] != generation.attempt_token
            or row["input_digest"] != config.digest(chunk.text)
            or not math.isfinite(row["norm"])
            or row["norm"] <= 0
        ):
            raise conflict()
    signature = hashlib.sha256(
        json.dumps(
            [
                generation.id,
                snapshot.digest,
                config.fingerprint,
                generation.published_at.isoformat(),
                generation.attempt_token,
                job.completed_at.isoformat(),
                [dict(r) for r in rows],
            ],
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return View(snapshot, generation.id, generation.profile_id, signature)


async def distances(db, uid, current, config, vector, top_k):
    repo, rev = current.snapshot.repository_id, current.snapshot.revision_id
    query_vector = cast(literal(vector, type_=Vector()), Vector())
    dimensions, norm = (
        await db.execute(
            select(func.public.vector_dims(query_vector), func.public.vector_norm(query_vector))
        )
    ).one()
    if dimensions != config.dimensions or not math.isfinite(norm) or norm <= 0:
        raise EmbeddingError("invalid_vector")
    distance = ChunkEmbedding.embedding.op("OPERATOR(public.<=>)", return_type=Float())(
        query_vector
    )
    query = (
        owned_vectors(uid, repo, rev, current.index_id)
        .join(
            SourceChunk,
            and_(
                SourceChunk.id == ChunkEmbedding.chunk_id,
                SourceChunk.source_id == ChunkEmbedding.source_id,
                SourceChunk.profile_id == ChunkEmbedding.profile_id,
            ),
        )
        .join(
            IngestedSource,
            and_(
                IngestedSource.id == SourceChunk.source_id,
                IngestedSource.profile_id == SourceChunk.profile_id,
                IngestedSource.revision_id == rev,
            ),
        )
        .where(
            RetrievalIndex.state == "ready",
            RetrievalIndex.config_fingerprint == config.fingerprint,
            ChunkEmbedding.profile_id == current.profile_id,
            ChunkEmbedding.revision_id == rev,
            ChunkEmbedding.config_fingerprint == config.fingerprint,
            ChunkEmbedding.dimensions == config.dimensions,
        )
        .with_only_columns(SourceChunk.id, distance.label("cosine_distance"))
        # The arithmetic sort prevents a future ANN operator index from changing exact semantics.
        .order_by(distance + 0.0, SourceChunk.id)
        .limit(top_k)
    )
    rows = (await db.execute(query)).all()
    if len(rows) != min(top_k, len(current.snapshot.chunks)):
        raise conflict()
    if any(
        not math.isfinite(r.cosine_distance) or not -1e-6 <= r.cosine_distance <= 2.000001
        for r in rows
    ):
        raise HTTPException(503, "Semantic scoring unavailable")
    return rows


async def search(
    db, user_id, repository_id, revision_id, request, *, _prepare=None, _finalize=None
):
    """Own only clean read transactions; never roll back the caller's pending writes."""
    if db.new or db.dirty or db.deleted:
        raise RuntimeError("Semantic search requires a clean session")
    request = SemanticSearchRequest.model_validate(request)
    if (
        len(request.query) > settings.keyword_search_max_query_characters
        or request.top_k > settings.keyword_search_max_top_k
    ):
        raise HTTPException(422, "Query or top_k exceeds configured search limit")
    acquired = False
    try:
        async with asyncio.timeout(settings.semantic_search_timeout_seconds):
            uid = str((await require_active_user(db, user_id)).id)
            try:
                repo, rev = str(UUID(str(repository_id))), str(UUID(str(revision_id)))
            except (ValueError, TypeError, AttributeError):
                raise HTTPException(404, "Repository revision not found") from None
            await _scope(db, uid, repo, rev)
            if request.index_id is not None:
                present = await db.scalar(
                    generations(uid, repo, rev)
                    .with_only_columns(RetrievalIndex.id)
                    .where(RetrievalIndex.id == str(request.index_id))
                )
                if present is None:
                    raise HTTPException(404, "Retrieval index not found")
            if not _SLOTS.acquire(blocking=False):
                raise HTTPException(
                    429, "Semantic search capacity busy; retry", headers={"Retry-After": "1"}
                )
            acquired = True
            config = configuration(request.configuration)
            config.bound(request.query, query=True)
            limits = SearchLimits.configured()
            initial = await view(
                db,
                uid,
                repo,
                rev,
                config,
                limits,
                str(request.index_id) if request.index_id else None,
            )
            if _prepare is not None:
                # Trusted CPU-only composition after complete authorization/integrity checks.
                await _prepare(initial.snapshot, limits)
            await db.rollback()
            used = 0

            async def reserve():
                nonlocal used
                if used >= min(config.max_requests, settings.semantic_search_max_provider_requests):
                    raise EmbeddingError("request_budget")
                if configuration(request.configuration).fingerprint != config.fingerprint:
                    raise conflict()
                confirmed = await view(db, uid, repo, rev, config, limits, initial.index_id)
                if confirmed.signature != initial.signature:
                    raise conflict()
                await db.rollback()  # provider callback returns with NO transaction/row locks
                used += 1

            vector = None
            if initial.snapshot.chunks:
                vector = validate_vectors(
                    [
                        await retrieval_indexing.make_provider(config).embed_query(
                            request.query, reserve
                        )
                    ],
                    1,
                    config.dimensions,
                )[0]
            if configuration(request.configuration).fingerprint != config.fingerprint:
                raise conflict()
            current = await view(db, uid, repo, rev, config, limits, initial.index_id)
            if current.signature != initial.signature:
                raise conflict()
            ordered = (
                await distances(db, uid, current, config, vector, request.top_k)
                if vector is not None
                else []
            )
            canonical = {str(c.id): (c, path) for c, path in current.snapshot.chunks}
            matches = []
            for position, row in enumerate(ordered, 1):
                chunk, path = canonical[row.id]
                bound(len(chunk.text.encode("utf-8")) <= limits.excerpt_bytes)
                matches.append(
                    SemanticMatch(
                        source_id=chunk.source_id,
                        chunk_id=chunk.id,
                        path=path,
                        commit_sha=current.snapshot.sha,
                        line_start=chunk.line_start,
                        line_end=chunk.line_end,
                        excerpt=chunk.text,
                        cosine_distance=row.cosine_distance,
                        rank=position,
                    )
                )
            response = SemanticSearchResponse(
                repository_id=repo,
                revision_id=rev,
                commit_sha=current.snapshot.sha,
                provider=config.provider,
                model=config.model,
                dimensions=config.dimensions,
                index_id=current.index_id,
                configuration_fingerprint=config.fingerprint,
                query_metadata=SemanticQueryMetadata(character_count=len(request.query)),
                total_candidate_count=len(current.snapshot.chunks),
                implementation=VERSION,
                matches=matches,
            )
            if _finalize is not None:
                # Trusted composition runs under the final canonical/generation locks.
                # No external embedding HTTP occurs after this point.
                response = await _finalize(current, config, response, limits)
            bound(len(response.model_dump_json().encode()) <= limits.response_bytes)
            return response
    except EmbeddingError as exc:
        if exc.code in {"input_limit", "empty_input", "batch_limit"}:
            raise HTTPException(422, "Semantic query input limit exceeded") from None
        if exc.code in {
            "hosted_not_permitted",
            "hosted_query_not_permitted",
            "configuration_unavailable",
        }:
            raise conflict() from None
        raise HTTPException(503, "Semantic embedding unavailable") from None
    except (ValidationError, ValueError, KeyError, TypeError):
        raise conflict() from None
    except (TimeoutError, DBAPIError):
        raise HTTPException(503, "Semantic search temporarily unavailable") from None
    finally:
        try:
            await db.rollback()
        finally:
            if acquired:
                _SLOTS.release()
