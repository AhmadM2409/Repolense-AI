"""Bounded, owner-scoped canonical snapshots and keyword search; no persistent writes."""

import asyncio
import hashlib
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from uuid import UUID

from fastapi import HTTPException
from pydantic import ValidationError
from sqlalchemy import LargeBinary, Text, and_, cast, func, select, text
from sqlalchemy.exc import DBAPIError

from api.schemas.search import (
    KeywordMatch,
    KeywordSearchRequest,
    KeywordSearchResponse,
    QueryMetadata,
)
from core.config import settings
from core.contracts import SourceDocument, TextChunk
from core.models import IngestedSource, Profile, Review, SourceChunk, User
from core.models.repository import Repository, RepositoryRevision
from core.services.authorization import require_active_user
from rag.retriever.revision_keyword import VERSION, CorpusLimitError, ScoringError, rank

_POOL = ThreadPoolExecutor(
    max_workers=settings.keyword_search_concurrency, thread_name_prefix="keyword"
)
_SLOTS = threading.BoundedSemaphore(settings.keyword_search_concurrency)
_SHA = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class SearchLimits:
    sources: int = 100
    chunks: int = 2000
    corpus_bytes: int = 4 * 1024 * 1024
    metadata_bytes: int = 512 * 1024
    tokens: int = 100000
    excerpt_bytes: int = 16 * 1024
    response_bytes: int = 128 * 1024

    @classmethod
    def configured(cls):
        return cls(
            sources=settings.keyword_search_max_sources,
            chunks=settings.keyword_search_max_chunks,
            corpus_bytes=settings.keyword_search_max_corpus_bytes,
            metadata_bytes=settings.keyword_search_max_metadata_bytes,
            tokens=settings.keyword_search_max_tokens,
            excerpt_bytes=settings.keyword_search_max_excerpt_bytes,
            response_bytes=settings.keyword_search_max_response_bytes,
        )


@dataclass(frozen=True)
class Snapshot:
    repository_id: str
    revision_id: str
    sha: str
    digest: str
    chunks: tuple[tuple[TextChunk, str], ...]


def _conflict():
    return HTTPException(409, "Verified revision snapshot is missing or inconsistent")


def _bound(ok):
    if not ok:
        raise HTTPException(413, "Keyword retrieval resource limit exceeded")


def _size(db, column):
    if db.bind.dialect.name == "postgresql":
        return func.octet_length(column)
    return func.length(cast(column, LargeBinary))


def _scope_query(user_id, repository_id, revision_id):
    return (
        select(
            Repository.id.label("repository_id"),
            Repository.profile_id,
            Repository.full_name,
            RepositoryRevision.id.label("revision_id"),
            RepositoryRevision.commit_sha,
        )
        .join(Profile, Profile.id == Repository.profile_id)
        .join(User, User.id == Profile.user_id)
        .join(
            RepositoryRevision,
            and_(
                RepositoryRevision.repository_id == Repository.id,
                RepositoryRevision.profile_id == Profile.id,
            ),
        )
        .where(
            User.id == user_id,
            User.is_active.is_(True),
            Repository.id == repository_id,
            RepositoryRevision.id == revision_id,
            Repository.provider == "github",
        )
    )


async def _scope(db, user_id, repository_id, revision_id):
    if db.bind.dialect.name == "postgresql":
        await db.execute(text("SET LOCAL statement_timeout = '2000ms'"))
        await db.execute(text("SET LOCAL lock_timeout = '500ms'"))
    await require_active_user(db, user_id)
    query = _scope_query(user_id, repository_id, revision_id)
    if db.bind.dialect.name == "postgresql":
        # Short parent locks also prevent FK insert phantoms while loading bounded rows.
        query = query.with_for_update(of=(User, Profile, Repository, RepositoryRevision))
    row = (await db.execute(query)).mappings().first()
    if row is None:
        raise HTTPException(404, "Repository revision not found")
    return row


async def _snapshot(db, user_id, repository_id, revision_id, limits):
    if db.bind.dialect.name == "postgresql":
        await db.execute(text("SET LOCAL statement_timeout = '2000ms'"))
        await db.execute(text("SET LOCAL lock_timeout = '500ms'"))
    scope = await _scope(db, user_id, repository_id, revision_id)
    if not _SHA.fullmatch(scope["commit_sha"]):
        raise _conflict()
    # Use scalar projections first; reject oversized proof metadata before fetching JSON/text.
    job_query = (
        select(Review.id, _size(db, cast(Review.ingestion_result, Text)).label("size"))
        .join(Profile, Profile.id == Review.profile_id)
        .where(
            Profile.user_id == user_id,
            Profile.id == scope["profile_id"],
            Review.revision_id == revision_id,
            Review.job_type == "github_ingestion",
            Review.status == "complete",
        )
        .order_by(Review.created_at.desc(), Review.id.desc())
        .limit(1)
    )
    if db.bind.dialect.name == "postgresql":
        job_query = job_query.with_for_update(read=True, of=Review)
    proof_row = (await db.execute(job_query)).first()
    if proof_row is None or proof_row.size is None:
        raise _conflict()
    _bound(proof_row.size <= limits.metadata_bytes)
    proof = await db.scalar(select(Review.ingestion_result).where(Review.id == proof_row.id))
    if not isinstance(proof, dict) or (
        proof.get("verified") is not True
        or proof.get("policy") != "github-text-v1"
        or proof.get("repository_id") != repository_id
        or proof.get("repository") != scope["full_name"]
        or proof.get("commit_sha") != scope["commit_sha"]
    ):
        raise _conflict()
    manifest = proof.get("files")
    if not isinstance(manifest, list) or not manifest:
        raise _conflict()
    _bound(len(manifest) <= limits.sources)
    expected = {}
    for item in manifest:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            raise _conflict()
        if item["id"] in expected:
            raise _conflict()
        expected[item["id"]] = item

    # Every corpus/aggregate query joins the same owner/repository/revision scope.
    sources = (
        select(IngestedSource)
        .join(
            RepositoryRevision,
            and_(
                IngestedSource.revision_id == RepositoryRevision.id,
                IngestedSource.profile_id == RepositoryRevision.profile_id,
            ),
        )
        .join(
            Repository,
            and_(
                Repository.id == RepositoryRevision.repository_id,
                Repository.profile_id == IngestedSource.profile_id,
            ),
        )
        .join(
            Profile,
            and_(Profile.id == IngestedSource.profile_id, Profile.id == Repository.profile_id),
        )
        .where(
            Profile.user_id == user_id,
            Repository.id == repository_id,
            RepositoryRevision.id == revision_id,
        )
    )
    source_keys = (
        sources.with_only_columns(IngestedSource.id)
        .order_by(IngestedSource.id)
        .limit(limits.sources + 1)
    )
    if db.bind.dialect.name == "postgresql":
        source_keys = source_keys.with_for_update(of=IngestedSource)
    ids = list((await db.scalars(source_keys)).all())
    _bound(len(ids) <= limits.sources)
    if set(ids) != set(expected) or proof.get("file_count") != len(ids):
        raise _conflict()
    aggregate = sources.with_only_columns(
        func.coalesce(func.sum(_size(db, IngestedSource.text)), 0),
        func.coalesce(func.sum(_size(db, cast(IngestedSource.source_metadata, Text))), 0),
    )
    body_size, metadata_size = (await db.execute(aggregate)).one()
    _bound(body_size <= limits.corpus_bytes and metadata_size <= limits.metadata_bytes)
    chunks_query = (
        select(SourceChunk)
        .join(
            IngestedSource,
            and_(
                SourceChunk.source_id == IngestedSource.id,
                SourceChunk.profile_id == IngestedSource.profile_id,
            ),
        )
        .join(
            RepositoryRevision,
            and_(
                RepositoryRevision.id == IngestedSource.revision_id,
                RepositoryRevision.profile_id == IngestedSource.profile_id,
            ),
        )
        .join(
            Repository,
            and_(
                Repository.id == RepositoryRevision.repository_id,
                Repository.profile_id == IngestedSource.profile_id,
            ),
        )
        .join(
            Profile,
            and_(Profile.id == IngestedSource.profile_id, Profile.id == Repository.profile_id),
        )
        .where(
            Profile.user_id == user_id,
            Repository.id == repository_id,
            RepositoryRevision.id == revision_id,
        )
    )
    keys_query = (
        chunks_query.with_only_columns(SourceChunk.id)
        .order_by(SourceChunk.id)
        .limit(limits.chunks + 1)
    )
    if db.bind.dialect.name == "postgresql":
        keys_query = keys_query.with_for_update(read=True, of=SourceChunk)
    chunk_ids = list((await db.scalars(keys_query)).all())
    _bound(len(chunk_ids) <= limits.chunks)
    chunk_bytes = await db.scalar(
        chunks_query.with_only_columns(func.coalesce(func.sum(_size(db, SourceChunk.text)), 0))
    )
    _bound(chunk_bytes <= limits.corpus_bytes)
    rows = (
        (
            await db.execute(
                sources.with_only_columns(
                    IngestedSource.id,
                    IngestedSource.profile_id,
                    IngestedSource.source_type,
                    IngestedSource.revision_id,
                    IngestedSource.path,
                    IngestedSource.text,
                    IngestedSource.content_hash,
                    IngestedSource.source_metadata,
                    IngestedSource.chunk_count,
                )
                .where(IngestedSource.id.in_(ids))
                .order_by(IngestedSource.id)
                .limit(limits.sources + 1)
            )
        )
        .mappings()
        .all()
    )
    chunks = list(
        (
            await db.execute(
                chunks_query.with_only_columns(
                    SourceChunk.id,
                    SourceChunk.profile_id,
                    SourceChunk.source_id,
                    SourceChunk.chunk_index,
                    SourceChunk.text,
                    SourceChunk.line_start,
                    SourceChunk.line_end,
                )
                .where(SourceChunk.id.in_(chunk_ids))
                .order_by(SourceChunk.id)
                .limit(limits.chunks + 1)
            )
        )
        .mappings()
        .all()
    )
    by_source = {}
    for row in chunks:
        chunk = TextChunk.model_validate(row)
        by_source.setdefault(str(chunk.source_id), []).append(chunk)
    output = []
    documents = []
    for row in rows:
        source = SourceDocument.model_validate(
            {key: value for key, value in row.items() if key != "chunk_count"}
        )
        ident = str(source.id)
        item = expected[ident]
        metadata = source.source_metadata
        actual = sorted(by_source.get(ident, []), key=lambda chunk: chunk.chunk_index)
        if (
            source.source_type != "repo"
            or str(source.profile_id) != scope["profile_id"]
            or str(source.revision_id) != revision_id
            or item.get("path") != source.path
            or item.get("content_hash") != source.content_hash
            or metadata.get("verified") is not True
            or metadata.get("commit_sha") != scope["commit_sha"]
            or not isinstance(item.get("blob_sha"), str)
            or not _SHA.fullmatch(item["blob_sha"])
            or metadata.get("blob_sha") != item["blob_sha"]
            or item.get("chunk_count") != len(actual)
            or row["chunk_count"] != len(actual)
            or [chunk.chunk_index for chunk in actual] != list(range(len(actual)))
        ):
            raise _conflict()
        lines = source.text.splitlines()
        for chunk in actual:
            if (
                str(chunk.profile_id) != scope["profile_id"]
                or chunk.line_end > len(lines)
                or chunk.text != "\n".join(lines[chunk.line_start - 1 : chunk.line_end])
            ):
                raise _conflict()
            output.append((chunk, source.path))
        documents.append(source.model_dump(mode="json"))
    if proof.get("chunk_count") != len(chunks) or len(output) != len(chunk_ids):
        raise _conflict()
    fingerprint = json.dumps(
        [
            repository_id,
            revision_id,
            scope["commit_sha"],
            documents,
            [chunk.model_dump(mode="json") for chunk, _ in output],
        ],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return Snapshot(
        repository_id,
        revision_id,
        scope["commit_sha"],
        hashlib.sha256(fingerprint.encode()).hexdigest(),
        tuple(sorted(output, key=lambda item: str(item[0].id))),
    )


async def search(db, user_id, repository_id, revision_id, request):
    """Own read transactions; callers must not mix pending writes into this session."""
    if db.new or db.dirty or db.deleted:
        raise RuntimeError("Keyword search requires a clean session")
    request = KeywordSearchRequest.model_validate(request)
    if (
        len(request.query) > settings.keyword_search_max_query_characters
        or request.top_k > settings.keyword_search_max_top_k
    ):
        raise HTTPException(422, "Query or top_k exceeds configured keyword limit")
    stop = threading.Event()
    acquired = False
    future = None
    try:
        if db.bind.dialect.name == "postgresql":
            await db.execute(text("SET LOCAL statement_timeout = '2000ms'"))
            await db.execute(text("SET LOCAL lock_timeout = '500ms'"))
        uid = str((await require_active_user(db, user_id)).id)
        try:
            repository_id, revision_id = str(UUID(str(repository_id))), str(UUID(str(revision_id)))
        except (ValueError, TypeError, AttributeError):
            raise HTTPException(404, "Repository revision not found") from None
        await _scope(db, uid, repository_id, revision_id)
        # Capacity is process-local and loop-independent; it stores no user/corpus state.
        if not _SLOTS.acquire(blocking=False):
            raise HTTPException(
                429, "Keyword search capacity busy; retry", headers={"Retry-After": "1"}
            )
        acquired = True
        limits = SearchLimits.configured()
        initial = await _snapshot(db, uid, repository_id, revision_id, limits)
        await db.rollback()  # No row locks/transaction across CPU ranking.
        future = asyncio.get_running_loop().run_in_executor(
            _POOL,
            partial(
                rank,
                [(str(chunk.id), chunk.text) for chunk, _ in initial.chunks],
                request.query,
                limits.tokens,
                stop.is_set,
            ),
        )
        ordered, term_count = await asyncio.wait_for(
            asyncio.shield(future), settings.keyword_search_timeout_seconds
        )
        current = await _snapshot(db, uid, repository_id, revision_id, limits)
        if current.digest != initial.digest:
            raise _conflict()
        by_id = {str(chunk.id): (chunk, path) for chunk, path in current.chunks}
        matches = []
        for position, (ident, score) in enumerate(ordered[: request.top_k], 1):
            chunk, path = by_id[ident]
            _bound(len(chunk.text.encode()) <= limits.excerpt_bytes)
            matches.append(
                KeywordMatch(
                    source_id=chunk.source_id,
                    chunk_id=chunk.id,
                    path=path,
                    commit_sha=current.sha,
                    line_start=chunk.line_start,
                    line_end=chunk.line_end,
                    excerpt=chunk.text,
                    bm25_score=score,
                    rank=position,
                )
            )
        response = KeywordSearchResponse(
            repository_id=repository_id,
            revision_id=revision_id,
            commit_sha=current.sha,
            query_metadata=QueryMetadata(character_count=len(request.query), term_count=term_count),
            total_candidate_count=len(current.chunks),
            total_match_count=len(ordered),
            implementation=VERSION,
            matches=matches,
        )
        _bound(len(response.model_dump_json().encode()) <= limits.response_bytes)
        return response
    except CorpusLimitError:
        raise HTTPException(413, "Keyword retrieval token limit exceeded") from None
    except (ValidationError, KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, ScoringError):
            raise HTTPException(503, "Keyword scoring unavailable") from None
        raise _conflict() from None
    except (TimeoutError, DBAPIError):
        raise HTTPException(503, "Keyword search temporarily unavailable") from None
    finally:
        stop.set()
        if acquired:
            if future is not None and not future.done():
                # A timed-out/cancelled coroutine must not free capacity while its CPU task runs.
                future.add_done_callback(
                    lambda finished: (
                        _SLOTS.release(),
                        None if finished.cancelled() else finished.exception(),
                    )
                )
            else:
                _SLOTS.release()
        await db.rollback()


async def rank_snapshot(snapshot, query, limits):
    """Internal bounded ranking of an already authorized, locked canonical snapshot."""
    if not _SLOTS.acquire(blocking=False):
        raise HTTPException(
            429, "Keyword search capacity busy; retry", headers={"Retry-After": "1"}
        )
    stop = threading.Event()
    future = None
    try:
        future = asyncio.get_running_loop().run_in_executor(
            _POOL,
            partial(
                rank,
                [(str(c.id), c.text) for c, _ in snapshot.chunks],
                query,
                limits.tokens,
                stop.is_set,
            ),
        )
        return await asyncio.wait_for(
            asyncio.shield(future), settings.keyword_search_timeout_seconds
        )
    except CorpusLimitError:
        raise HTTPException(413, "Keyword retrieval token limit exceeded") from None
    except (TimeoutError, ScoringError):
        raise HTTPException(503, "Keyword scoring unavailable") from None
    finally:
        stop.set()
        if future is not None and not future.done():
            future.add_done_callback(
                lambda done: (_SLOTS.release(), None if done.cancelled() else done.exception())
            )
        else:
            _SLOTS.release()
