"""Owner-scoped persistence of supplied documents/evidence; performs no fetching or analysis.

Functions own their writes (commit/rollback). Do not mix unrelated pending writes in the session.
"""

from fastapi import HTTPException
from sqlalchemy import select
from core.contracts import (
    SourceDocument,
    TextChunk,
    FindingContract,
    RevisionContract,
    RepositoryContract,
)
from core.models import IngestedSource
from core.models.repository import Repository, RepositoryRevision
from core.models.evidence import SourceChunk, Finding, Evidence
from core.services.profile_service import get_profile
from core.services.review_service import get_review


async def _owned(db, profile_id, user_id):
    if await get_profile(db, profile_id, user_id) is None:
        raise HTTPException(404, "Profile not found")


async def register_revision(
    db, user_id, repository: RepositoryContract, revision: RevisionContract
):
    await _owned(db, repository.profile_id, user_id)
    if revision.repository_id != repository.id or revision.profile_id != repository.profile_id:
        raise ValueError("Revision does not match repository/profile")
    try:
        stored = await db.get(Repository, str(repository.id))
        if stored and (
            stored.profile_id != str(repository.profile_id)
            or stored.full_name != repository.full_name
        ):
            raise ValueError("Repository identity cannot be reassigned")
        if not stored:
            db.add(Repository(**repository.model_dump(mode="json")))
            await db.flush()
        db.add(RepositoryRevision(**revision.model_dump(mode="json")))
        await db.commit()
    except Exception:
        await db.rollback()
        raise


async def persist_document(
    db, user_id, document: SourceDocument, chunks: list[TextChunk], *, commit: bool = True
):
    await _owned(db, document.profile_id, user_id)
    try:
        if document.revision_id:
            revision = await db.get(RepositoryRevision, str(document.revision_id))
            if not revision or revision.profile_id != str(document.profile_id):
                raise ValueError("Revision not found in profile")
        lines = document.text.splitlines()
        positions = set()
        for chunk in chunks:
            if chunk.profile_id != document.profile_id or chunk.source_id != document.id:
                raise ValueError("Chunk does not belong to document")
            if chunk.chunk_index in positions:
                raise ValueError("Duplicate chunk index")
            positions.add(chunk.chunk_index)
            if chunk.line_end > len(lines) or chunk.text != "\n".join(
                lines[chunk.line_start - 1 : chunk.line_end]
            ):
                raise ValueError("Chunk text/range does not match source")
        row = IngestedSource(**document.model_dump(mode="json"), chunk_count=len(chunks))
        db.add(row)
        await db.flush()
        db.add_all(SourceChunk(**chunk.model_dump(mode="json")) for chunk in chunks)
        await db.flush()
        if commit:
            await db.commit()
        return row
    except Exception:
        await db.rollback()
        raise


async def persist_finding(db, user_id, finding: FindingContract, *, commit: bool = True):
    review = await get_review(db, finding.review_id, user_id)
    if not review or review.profile_id != str(finding.profile_id):
        raise HTTPException(404, "Review not found")
    try:
        if not review.revision_id:
            raise ValueError("Evidence requires a revision-bound review")
        for span in finding.evidence:
            source = await db.get(IngestedSource, str(span.source_id))
            if (
                not source
                or source.profile_id != review.profile_id
                or source.revision_id != review.revision_id
            ):
                raise ValueError("Evidence source/revision is outside review")
            revision = await db.get(RepositoryRevision, source.revision_id)
            if (
                revision.repository_id != str(span.repository_id)
                or revision.commit_sha != span.commit_sha
                or source.path != span.path
            ):
                raise ValueError("Evidence identity does not match immutable source")
            lines = (source.text or "").splitlines()
            if span.line_end > len(lines) or span.quote != "\n".join(
                lines[span.line_start - 1 : span.line_end]
            ):
                raise ValueError("Evidence quote/range does not match source")
        row = Finding(
            id=str(finding.id),
            profile_id=review.profile_id,
            review_id=review.id,
            title=finding.title,
            detail=finding.detail,
            rule_id=finding.rule_id,
            severity=finding.severity,
            category=finding.category,
            certainty=finding.certainty,
            fingerprint=finding.fingerprint,
        )
        db.add(row)
        await db.flush()
        db.add_all(
            Evidence(
                profile_id=review.profile_id,
                finding_id=row.id,
                review_id=review.id,
                revision_id=review.revision_id,
                source_id=str(span.source_id),
                line_start=span.line_start,
                line_end=span.line_end,
                quote=span.quote,
            )
            for span in finding.evidence
        )
        await db.flush()
        if commit:
            await db.commit()
        return row
    except Exception:
        await db.rollback()
        raise


async def get_findings(db, user_id, review_id):
    """No unscoped report access; future routes must use this boundary."""
    review = await get_review(db, review_id, user_id)
    if not review:
        raise HTTPException(404, "Review not found")
    findings = (await db.scalars(select(Finding).where(Finding.review_id == review.id))).all()
    result = []
    from core.contracts import EvidenceSpan

    for finding in findings:
        spans = (await db.scalars(select(Evidence).where(Evidence.finding_id == finding.id))).all()
        evidence = []
        for span in spans:
            source = await db.get(IngestedSource, span.source_id)
            revision = await db.get(RepositoryRevision, source.revision_id)
            evidence.append(
                EvidenceSpan(
                    source_id=source.id,
                    repository_id=revision.repository_id,
                    commit_sha=revision.commit_sha,
                    path=source.path,
                    line_start=span.line_start,
                    line_end=span.line_end,
                    quote=span.quote,
                )
            )
        result.append(
            FindingContract(
                id=finding.id,
                profile_id=finding.profile_id,
                review_id=finding.review_id,
                title=finding.title,
                detail=finding.detail,
                evidence=evidence,
                rule_id=finding.rule_id,
                severity=finding.severity,
                category=finding.category,
                certainty=finding.certainty,
                fingerprint=finding.fingerprint,
            )
        )
    return result
