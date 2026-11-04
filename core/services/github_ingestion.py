"""Existing owner-scoped Review jobs drive verified GitHub ingestion; no AI/vector calls."""

import asyncio
import hashlib
from datetime import datetime, UTC
from uuid import uuid4, uuid5, NAMESPACE_URL
from fastapi import HTTPException
from sqlalchemy import select, func
from core.config import settings
from core.database import AsyncSessionLocal
from core.models import Review, Profile, IngestedSource, User
from core.models.repository import Repository, RepositoryRevision
from core.models.evidence import SourceChunk
from core.contracts import SourceDocument, TextChunk
from core.services.authorization import require_active_user
from core.services.source_service import persist_document
from ingestion.chunking.base import source_line_chunks
from ingestion.github import GitHubClient, GitHubError, repository_name


def make_github_client():
    return GitHubClient(settings.github_token)


async def _profile(db, profile_id, user_id, *, lock=False):
    user = await require_active_user(db, user_id)
    if lock:
        # Lock activity before profile ownership for consistent commit-time authorization.
        await db.execute(select(User).where(User.id == user.id).with_for_update())
        user = await require_active_user(db, user_id)
    query = select(Profile).where(Profile.id == str(profile_id), Profile.user_id == user.id)
    if lock:
        query = query.with_for_update()
    profile = await db.scalar(query)
    if profile is None:
        raise HTTPException(404, "Profile not found")
    return profile


async def get_ingestion(db, job_id, user_id, *, lock=False):
    user = await require_active_user(db, user_id)
    query = (
        select(Review)
        .join(Profile)
        .where(
            Review.id == str(job_id),
            Review.job_type == "github_ingestion",
            Profile.user_id == user.id,
        )
    )
    if lock:
        query = query.with_for_update(of=Review).execution_options(populate_existing=True)
    job = await db.scalar(query)
    if job is None:
        raise HTTPException(404, "Ingestion not found")
    return job


async def submit(db, profile_id, user_id, name):
    name = repository_name(name)
    await _profile(db, profile_id, user_id, lock=True)
    try:
        active = await db.scalar(
            select(Review).where(
                Review.profile_id == str(profile_id),
                Review.job_type == "github_ingestion",
                Review.status.in_(["pending", "processing"]),
            )
        )
        if active:
            if active.requested_repository == name:
                await db.commit()
                return active, False
            raise HTTPException(409, "Another repository ingestion is active for this profile")
        count = await db.scalar(
            select(func.count())
            .select_from(Repository)
            .where(Repository.profile_id == str(profile_id))
        )
        existing = await db.scalar(
            select(Repository).where(
                Repository.profile_id == str(profile_id), Repository.full_name == name
            )
        )
        if not existing and count >= settings.max_repos_per_profile:
            raise HTTPException(422, "Profile repository limit reached")
        job = Review(
            id=str(uuid4()),
            profile_id=str(profile_id),
            job_type="github_ingestion",
            requested_repository=name,
            status="pending",
            max_attempts=settings.job_max_attempts,
            sections=None,
            overall_score=None,
        )
        db.add(job)
        await db.commit()
        await db.refresh(job)
        from core.services.job_queue import event
        event("submitted", job)
        return job, True
    except Exception:
        await db.rollback()
        raise


async def _reused(db, job, revision):
    repository = await db.scalar(
        select(Repository).where(
            Repository.profile_id == job.profile_id, Repository.full_name == revision.full_name
        )
    )
    if not repository:
        return None
    stored = await db.scalar(
        select(RepositoryRevision).where(
            RepositoryRevision.repository_id == repository.id,
            RepositoryRevision.commit_sha == revision.sha,
        )
    )
    if not stored:
        return None
    previous = (
        await db.scalars(
            select(Review)
            .where(
                Review.profile_id == job.profile_id,
                Review.job_type == "github_ingestion",
                Review.status == "complete",
                Review.revision_id == stored.id,
            )
            .order_by(Review.created_at.desc())
        )
    ).first()
    if (
        not previous
        or not previous.ingestion_result
        or previous.ingestion_result.get("policy") != "github-text-v1"
    ):
        return None
    result = previous.ingestion_result
    sources = (
        await db.scalars(select(IngestedSource).where(IngestedSource.revision_id == stored.id))
    ).all()
    expected = {item["id"]: item for item in result["files"]}
    if len(sources) != len(expected):
        raise GitHubError(
            "snapshot_conflict", "Stored revision is incomplete; no success was recorded"
        )
    for source in sources:
        item = expected.get(source.id)
        chunks = (
            await db.scalars(select(SourceChunk).where(SourceChunk.source_id == source.id))
        ).all()
        lines = (source.text or "").splitlines()
        metadata = source.source_metadata or {}
        valid_chunks = all(
            chunk.line_end <= len(lines)
            and chunk.text == "\n".join(lines[chunk.line_start - 1 : chunk.line_end])
            for chunk in chunks
        )
        if (
            not item
            or source.content_hash != item["content_hash"]
            or source.path != item["path"]
            or len(chunks) != item["chunk_count"]
            or source.chunk_count != len(chunks)
            or not valid_chunks
            or hashlib.sha256((source.text or "").encode()).hexdigest() != source.content_hash
            or metadata.get("verified") is not True
            or metadata.get("blob_sha") != item["blob_sha"]
            or metadata.get("commit_sha") != revision.sha
        ):
            raise GitHubError("snapshot_conflict", "Stored revision is inconsistent")
    return stored, {**result, "deduplicated": True}


async def _persist(db, job, snapshot, user_id):
    profile = await _profile(db, job.profile_id, user_id, lock=True)
    repository = await db.scalar(
        select(Repository).where(
            Repository.profile_id == profile.id, Repository.full_name == snapshot.revision.full_name
        )
    )
    if repository is None:
        count = await db.scalar(
            select(func.count()).select_from(Repository).where(Repository.profile_id == profile.id)
        )
        if count >= settings.max_repos_per_profile:
            raise GitHubError("repository_limit", "Profile repository limit reached")
        repository = Repository(
            profile_id=profile.id, full_name=snapshot.revision.full_name, provider="github"
        )
        db.add(repository)
        await db.flush()
    revision = await db.scalar(
        select(RepositoryRevision).where(
            RepositoryRevision.repository_id == repository.id,
            RepositoryRevision.commit_sha == snapshot.revision.sha,
        )
    )
    if revision is None:
        revision = RepositoryRevision(
            profile_id=profile.id, repository_id=repository.id, commit_sha=snapshot.revision.sha
        )
        db.add(revision)
        await db.flush()
    # Existing unverified/conflicting sources must never be silently relabelled or overwritten.
    if await db.scalar(
        select(IngestedSource.id).where(IngestedSource.revision_id == revision.id).limit(1)
    ):
        raise GitHubError(
            "snapshot_conflict", "Revision already has an unverified or conflicting source snapshot"
        )
    manifest = []
    total_chunks = 0
    for file in snapshot.files:
        source_id = uuid5(NAMESPACE_URL, revision.id + ":" + file.path)
        raw_chunks = source_line_chunks(
            file.text, {"source_id": str(source_id), "profile_id": profile.id}
        )
        total_chunks += len(raw_chunks)
        if total_chunks > 2000:
            raise GitHubError("chunk_limit", "Repository exceeds the chunk limit")
        document = SourceDocument(
            id=source_id,
            profile_id=profile.id,
            source_type="repo",
            revision_id=revision.id,
            path=file.path,
            text=file.text,
            content_hash=file.content_hash,
            source_metadata={
                "provider": "github",
                "verified": True,
                "repository": snapshot.revision.full_name,
                "commit_sha": snapshot.revision.sha,
                "tree_sha": snapshot.revision.tree_sha,
                "blob_sha": file.blob_sha,
                "raw_bytes": file.raw_bytes,
                "normalization": "utf8-bom-crlf-v1",
            },
        )
        chunks = [
            TextChunk(
                id=uuid5(NAMESPACE_URL, str(source_id) + ":" + str(index)),
                profile_id=profile.id,
                source_id=source_id,
                chunk_index=chunk.chunk_index,
                text=chunk.text,
                line_start=chunk.metadata["line_start"],
                line_end=chunk.metadata["line_end"],
            )
            for index, chunk in enumerate(raw_chunks)
        ]
        await persist_document(db, user_id, document, chunks, commit=False)
        manifest.append(
            {
                "id": str(source_id),
                "path": file.path,
                "content_hash": file.content_hash,
                "blob_sha": file.blob_sha,
                "chunk_count": len(chunks),
            }
        )
    return revision, {
        "policy": "github-text-v1",
        "repository_id": repository.id,
        "repository": snapshot.revision.full_name,
        "commit_sha": snapshot.revision.sha,
        "default_branch": snapshot.revision.branch,
        "tree_sha": snapshot.revision.tree_sha,
        "verified": True,
        "deduplicated": False,
        "files": manifest,
        "file_count": len(manifest),
        "chunk_count": total_chunks,
        "raw_bytes": sum(file.raw_bytes for file in snapshot.files),
        "visited_entries": snapshot.entries,
        "skipped": snapshot.skipped,
    }


async def process(db, job_id, user_id, client, *, attempt=None):
    from core.services import job_queue as queue
    if attempt is None:
        return await queue.direct(db, job_id, user_id,
            lambda owned: process(db, job_id, user_id, client, attempt=owned))
    try:
        await attempt.checkpoint("resolving_revision")
        job = await get_ingestion(db, job_id, user_id)
        queue.assert_attempt(job, attempt)
        requested_repository = job.requested_repository
        await db.commit()
        client.checkpoint = attempt.checkpoint
        async with asyncio.timeout(client.limits.seconds):
            revision = await client.resolve(requested_repository)
            reused = await _reused(db, job, revision)
            await db.commit()
            await attempt.checkpoint("fetching_tree")
            snapshot = None if reused else await client.download(revision)
        await attempt.checkpoint("validating_sources")
        await attempt.checkpoint("persisting")
        await _profile(db, job.profile_id, user_id, lock=True)
        job = await get_ingestion(db, job_id, user_id, lock=True)
        queue.assert_attempt(job, attempt)
        if reused:
            stored, result = await _reused(db, job, revision)
        else:
            stored, result = await _persist(db, job, snapshot, user_id)
        job.revision_id = stored.id
        job.ingestion_result = result
        queue.complete(job, attempt)
        await db.commit()
        queue.event("completed", job)
    except BaseException:
        await db.rollback()
        raise


async def record_failure(db, job_id, user_id, code, message):
    try:
        job = await get_ingestion(db, job_id, user_id, lock=True)
        if job.status == "pending" and job.execution_token is None:
            job.status = "failed"
            job.error_message = message
            job.ingestion_result = {"error_code": code}
            job.updated_at = datetime.now(UTC)
            await db.commit()
    except Exception:
        await db.rollback()  # Revoked/deleted principals cannot mutate even failure state.


async def run_job(job_id, user_id):
    async with AsyncSessionLocal() as db:
        try:
            async with make_github_client() as client:
                await process(db, job_id, user_id, client)
        except HTTPException:
            await db.rollback()
        except Exception as exc:
            await db.rollback()
            if isinstance(exc, GitHubError):
                await record_failure(db, job_id, user_id, exc.code, exc.message)
            else:
                await record_failure(
                    db, job_id, user_id, "ingestion_failed", "GitHub ingestion could not start"
                )


async def sources(db, job_id, user_id, page=1, page_size=10):
    job = await get_ingestion(db, job_id, user_id)
    if job.status != "complete":
        raise HTTPException(409, "Ingestion is not complete")
    query = select(IngestedSource).where(
        IngestedSource.revision_id == job.revision_id, IngestedSource.profile_id == job.profile_id
    )
    total = await db.scalar(select(func.count()).select_from(query.subquery()))
    rows = (
        await db.scalars(
            query.order_by(
                IngestedSource.path.collate("C" if db.bind.dialect.name == "postgresql" else "BINARY"),
                IngestedSource.id,
            ).offset((page - 1) * page_size).limit(page_size)
        )
    ).all()
    return {
        "items": [
            {
                "id": row.id,
                "path": row.path,
                "content_hash": row.content_hash,
                "chunk_count": row.chunk_count,
            }
            for row in rows
        ],
        "total": total,
        "page": page,
        "page_size": page_size,
    }


async def source_detail(db, job_id, source_id, user_id):
    job = await get_ingestion(db, job_id, user_id)
    if job.status != "complete":
        raise HTTPException(409, "Ingestion is not complete")
    source = await db.scalar(
        select(IngestedSource).where(
            IngestedSource.id == str(source_id),
            IngestedSource.revision_id == job.revision_id,
            IngestedSource.profile_id == job.profile_id,
        )
    )
    if source is None:
        raise HTTPException(404, "Source not found")
    chunks = (
        await db.scalars(
            select(SourceChunk)
            .where(SourceChunk.source_id == source.id)
            .order_by(SourceChunk.chunk_index)
        )
    ).all()
    return {
        "source": SourceDocument.model_validate(source),
        "chunks": [TextChunk.model_validate(c) for c in chunks],
    }
