"""Owner-scoped leased deterministic analysis using the canonical evidence validators."""

import asyncio
from uuid import uuid4

from fastapi import HTTPException
from sqlalchemy import select

from analysis.engine import analyze, stable_hash
from analysis.rules import VERSION
from core.contracts import SourceDocument
from core.database import AsyncSessionLocal
from core.models import IngestedSource, Profile, Review
from core.models.repository import Repository, RepositoryRevision
from core.services import github_ingestion as ingestion
from core.services.authorization import require_active_user



def metadata(job):
    return job.analysis_result or {}


async def _owned_job(db, job_id, user_id, lock=False):
    user = await require_active_user(db, user_id)
    query = (
        select(Review)
        .join(Profile)
        .where(
            Review.id == str(job_id),
            Profile.user_id == user.id,
            Review.job_type == "review",
            Review.report_kind == "evidence",
        )
    )
    if lock:
        query = query.with_for_update(of=Review).execution_options(populate_existing=True)
    job = await db.scalar(query)
    if job is None or not metadata(job).get("engine_version"):
        raise HTTPException(404, "Analysis not found")
    return job


async def get_job(db, job_id, user_id):
    # Recovery belongs to the independent worker, never to reads.
    return await _owned_job(db, job_id, user_id)


async def verified_input(db, ingestion_id, user_id):
    source_job = await ingestion.get_ingestion(db, ingestion_id, user_id)
    if source_job.status != "complete" or not source_job.revision_id:
        raise HTTPException(409, "A completed verified repository ingestion is required")
    revision = await db.get(RepositoryRevision, source_job.revision_id)
    repository = await db.get(Repository, revision.repository_id) if revision else None
    result = source_job.ingestion_result or {}
    if (
        not revision
        or not repository
        or repository.profile_id != source_job.profile_id
        or result.get("commit_sha") != revision.commit_sha
        or result.get("repository_id") != repository.id
        or result.get("verified") is not True
    ):
        raise HTTPException(409, "Ingestion provenance is invalid")
    from ingestion.github import Revision

    reused = await ingestion._reused(
        db,
        source_job,
        Revision(
            repository.full_name,
            result.get("default_branch", ""),
            revision.commit_sha,
            result.get("tree_sha", ""),
        ),
    )
    if not reused or reused[0].id != revision.id:
        raise HTTPException(409, "Verified source snapshot is incomplete")
    rows = (
        await db.scalars(
            select(IngestedSource)
            .where(
                IngestedSource.profile_id == source_job.profile_id,
                IngestedSource.revision_id == revision.id,
            )
            .order_by(IngestedSource.path)
        )
    ).all()
    documents = [SourceDocument.model_validate(row) for row in rows]
    input_hash = stable_hash(
        [VERSION, revision.id, [(str(doc.id), doc.path, doc.content_hash) for doc in documents]]
    )
    return source_job, revision, repository, documents, input_hash


async def submit(db, ingestion_id, user_id):
    source_job = await ingestion.get_ingestion(db, ingestion_id, user_id)
    await ingestion._profile(db, source_job.profile_id, user_id, lock=True)
    try:
        source_job, revision, repository, documents, input_hash = await verified_input(
            db, ingestion_id, user_id
        )
        active = (
            await db.scalars(
                select(Review)
                .where(
                    Review.profile_id == source_job.profile_id,
                    Review.job_type == "review",
                    Review.report_kind == "evidence",
                    Review.status.in_(["pending", "processing"]),
                )
                .with_for_update()
            )
        ).all()
        for job in active:
            if metadata(job).get("input_hash") == input_hash:
                await db.commit()
                return job, False
            raise HTTPException(409, "Another analysis is active for this profile")
        completed = (
            await db.scalars(
                select(Review)
                .where(
                    Review.profile_id == source_job.profile_id,
                    Review.revision_id == revision.id,
                    Review.report_kind == "evidence",
                    Review.status == "complete",
                )
                .order_by(Review.created_at.desc())
            )
        ).all()
        for job in completed:
            if (
                metadata(job).get("engine_version") == VERSION
                and metadata(job).get("input_hash") == input_hash
            ):
                await report(db, job.id, user_id)
                await db.commit()
                return job, False
        job = Review(
            id=str(uuid4()),
            profile_id=source_job.profile_id,
            revision_id=revision.id,
            job_type="review",
            report_kind="evidence",
            requested_repository=repository.full_name,
            status="pending",
            max_attempts=ingestion.settings.job_max_attempts,
            sections=None,
            overall_score=None,
            analysis_result={
                "engine_version": VERSION,
                "producer": "deterministic",
                "ingestion_id": source_job.id,
                "input_hash": input_hash,
            },
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


async def process(db, job_id, user_id, *, attempt=None):
    from core.services import job_queue as queue
    if attempt is None:
        return await queue.direct(db, job_id, user_id,
            lambda owned: process(db, job_id, user_id, attempt=owned))
    try:
        await attempt.checkpoint("validating_snapshot")
        job = await _owned_job(db, job_id, user_id)
        queue.assert_attempt(job, attempt)
        data = metadata(job).copy()
        source_job, revision, repository, documents, input_hash = await verified_input(
            db, data["ingestion_id"], user_id)
        if input_hash != data["input_hash"]:
            raise ValueError("Input changed")
        repository_id, sha, profile_id = repository.id, revision.commit_sha, job.profile_id
        await db.commit()
        await attempt.checkpoint("analyzing_sources")
        async with asyncio.timeout(30):
            summary, findings = await asyncio.to_thread(
                analyze, documents, repository_id, sha, job_id, profile_id,
                checkpoint=attempt.check_sync)
        await attempt.checkpoint("validating_evidence")
        await attempt.checkpoint("persisting_results")
        await ingestion._profile(db, profile_id, user_id, lock=True)
        job = await _owned_job(db, job_id, user_id, lock=True)
        queue.assert_attempt(job, attempt)
        *_, current_hash = await verified_input(db, data["ingestion_id"], user_id)
        if current_hash != input_hash:
            raise ValueError("Input changed")
        from core.services.source_service import persist_finding
        for finding in findings:
            await persist_finding(db, user_id, finding, commit=False)
        serialized = sorted(
            [finding.model_dump(mode="json") for finding in findings], key=lambda item: item["id"])
        job.analysis_result = {
            **data, "summary": summary, "result_hash": stable_hash([summary, serialized])}
        job.sections = None
        job.overall_score = None
        queue.complete(job, attempt)
        await db.commit()
        queue.event("completed", job)
    except BaseException:
        await db.rollback()
        raise


async def run_job(job_id, user_id):
    async with AsyncSessionLocal() as db:
        try:
            await process(db, job_id, user_id)
        except HTTPException:
            await db.rollback()


async def report(db, job_id, user_id):
    job = await get_job(db, job_id, user_id)
    if job.status != "complete":
        raise HTTPException(409, "Analysis is not complete")
    result = metadata(job)
    if result.get("engine_version") != VERSION or not result.get("summary"):
        raise HTTPException(410, "Unsupported historical analysis")
    try:
        _, revision, repository, documents, input_hash = await verified_input(
            db, result["ingestion_id"], user_id
        )
        if input_hash != result["input_hash"]:
            raise ValueError("Input changed")
        from core.services.source_service import get_findings

        findings = await get_findings(db, user_id, job.id)
        by_source = {str(doc.id): doc for doc in documents}
        for finding in findings:
            for span in finding.evidence:
                source = by_source.get(str(span.source_id))
                if (
                    not source
                    or span.commit_sha != revision.commit_sha
                    or str(span.repository_id) != repository.id
                    or span.path != source.path
                    or span.line_end > len(source.text.splitlines())
                    or span.quote
                    != "\n".join(source.text.splitlines()[span.line_start - 1 : span.line_end])
                ):
                    raise ValueError("Evidence changed")
        serialized = sorted(
            [finding.model_dump(mode="json") for finding in findings], key=lambda item: item["id"]
        )
        if stable_hash([result["summary"], serialized]) != result["result_hash"]:
            raise ValueError("Results changed")
        return {
            "review_id": job.id,
            "profile_id": job.profile_id,
            "revision_id": revision.id,
            "commit_sha": revision.commit_sha,
            "repository": repository.full_name,
            "engine_version": VERSION,
            "producer": "deterministic",
            "summary": result["summary"],
            "findings": findings,
        }
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(409, "Analysis evidence or source integrity check failed") from None
