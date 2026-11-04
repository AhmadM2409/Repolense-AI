"""Real migrated databases and source/evidence invariants; no mocked SQL/auth."""

import hashlib
from uuid import uuid4
import pytest
from fastapi import HTTPException
from pydantic import ValidationError
from sqlalchemy import select, func, update
from sqlalchemy.exc import IntegrityError, DBAPIError
from core.contracts import (
    RepositoryContract,
    RevisionContract,
    SourceDocument,
    TextChunk,
    EvidenceSpan,
    FindingContract,
    normalize_github_metadata,
)
from core.models import IngestedSource, Review, Profile
from core.models.repository import Repository, RepositoryRevision
from core.models.evidence import Evidence, Finding, SourceChunk
from core.services import source_service as service
from ingestion.chunking.base import source_line_chunks
from ingestion.parsers.repo_analyzer import RepoAnalyzer

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


async def setup_document(db, actors, index=0):
    uid, pid = actors["users"][index].id, actors["profiles"][index].id
    repository = RepositoryContract(id=uuid4(), profile_id=pid, full_name="example/project")
    revision = RevisionContract(
        id=uuid4(), profile_id=pid, repository_id=repository.id, commit_sha="a" * 40
    )
    await service.register_revision(db, uid, repository, revision)
    body = "first line\nsecond line\nthird line\n"
    document = SourceDocument(
        id=uuid4(),
        profile_id=pid,
        source_type="repo",
        revision_id=revision.id,
        path="src/example.py",
        text=body,
        content_hash=hashlib.sha256(body.encode()).hexdigest(),
        source_metadata={"language": "Python"},
    )
    chunks = [
        TextChunk(
            id=uuid4(),
            profile_id=pid,
            source_id=document.id,
            chunk_index=0,
            text="first line\nsecond line",
            line_start=1,
            line_end=2,
        )
    ]
    await service.persist_document(db, uid, document, chunks)
    review = await db.get(Review, actors["reviews"][index].id)
    review.revision_id = str(revision.id)
    await db.commit()
    span = EvidenceSpan(
        source_id=document.id,
        repository_id=repository.id,
        commit_sha=revision.commit_sha,
        path=document.path,
        line_start=2,
        line_end=3,
        quote="second line\nthird line",
    )
    finding = FindingContract(
        id=uuid4(),
        profile_id=pid,
        review_id=review.id,
        title="Synthetic fixture finding",
        detail="Caller-supplied test data",
        evidence=[span],
    )
    return uid, repository, revision, document, chunks, finding


async def test_document_chunks_finding_roundtrip(database, actors):
    async with database() as db:
        uid, repo, rev, doc, chunks, finding = await setup_document(db, actors)
        row = await service.persist_finding(db, uid, finding)
        result = await service.get_findings(db, uid, finding.review_id)
        assert result[0].model_dump(mode="json") == finding.model_dump(mode="json")
        assert (await db.get(SourceChunk, str(chunks[0].id))).text == chunks[0].text
        assert (await db.get(IngestedSource, str(doc.id))).chunk_count == 1
        assert await db.scalar(select(func.count()).select_from(Evidence)) == 1
        assert row.title == finding.title


@pytest.mark.parametrize(
    "mutation", ["sha", "path", "quote", "bounds", "repository", "source", "other_revision"]
)
async def test_invalid_evidence_rolls_back(database, actors, mutation):
    async with database() as db:
        uid, repo, rev, doc, chunks, finding = await setup_document(db, actors)
        data = finding.model_dump(mode="json")
        span = data["evidence"][0]
        if mutation == "sha":
            span["commit_sha"] = "b" * 40
        if mutation == "path":
            span["path"] = "other.py"
        if mutation == "quote":
            span["quote"] = "fabricated"
        if mutation == "bounds":
            span["line_end"] = 99
        if mutation == "repository":
            span["repository_id"] = str(uuid4())
        if mutation == "source":
            span["source_id"] = str(uuid4())
        if mutation == "other_revision":
            other = RepositoryRevision(
                profile_id=str(doc.profile_id), repository_id=str(repo.id), commit_sha="b" * 40
            )
            db.add(other)
            await db.commit()
            review = await db.get(Review, str(finding.review_id))
            review.revision_id = other.id
            await db.commit()
        with pytest.raises(ValueError):
            await service.persist_finding(db, uid, FindingContract.model_validate(data))
        assert await db.scalar(select(func.count()).select_from(Finding)) == 0
        assert await db.scalar(select(func.count()).select_from(Evidence)) == 0
        assert await db.get(IngestedSource, str(doc.id))


async def test_cross_user_new_service_boundaries(database, actors):
    async with database() as db:
        uid, repo, rev, doc, chunks, finding = await setup_document(db, actors)
        other = actors["users"][1].id
        for call in [
            lambda: service.persist_document(db, other, doc, chunks),
            lambda: service.persist_finding(db, other, finding),
            lambda: service.get_findings(db, other, finding.review_id),
            lambda: service.register_revision(db, other, repo, rev),
        ]:
            with pytest.raises(HTTPException) as denied:
                await call()
            assert denied.value.status_code == 404
        assert await db.scalar(select(func.count()).select_from(Finding)) == 0


@pytest.mark.parametrize("field", ["text", "path", "content_hash", "revision_id"])
async def test_revision_source_immutable_in_database(database, actors, field):
    async with database() as db:
        uid, repo, rev, doc, chunks, finding = await setup_document(db, actors)
        value = {
            "text": "changed",
            "path": "elsewhere.py",
            "content_hash": "b" * 64,
            "revision_id": None,
        }[field]
        with pytest.raises(DBAPIError):
            await db.execute(
                update(IngestedSource)
                .where(IngestedSource.id == str(doc.id))
                .values(**{field: value})
            )
            await db.commit()
        await db.rollback()
        assert (await db.get(IngestedSource, str(doc.id))).text == doc.text
        with pytest.raises(DBAPIError):
            await db.execute(
                update(RepositoryRevision)
                .where(RepositoryRevision.id == str(rev.id))
                .values(commit_sha="b" * 40)
            )
        await db.rollback()


async def test_database_scope_and_revision_constraints(database, actors):
    async with database() as db:
        uid, repo, rev, doc, chunks, finding = await setup_document(db, actors)
        # Direct ORM bypass still cannot join a foreign profile to an owned source.
        db.add(
            SourceChunk(
                profile_id=actors["profiles"][1].id,
                source_id=str(doc.id),
                chunk_index=3,
                text="first line",
                line_start=1,
                line_end=1,
            )
        )
        with pytest.raises(IntegrityError):
            await db.commit()
        await db.rollback()
        await service.persist_finding(db, uid, finding)
        other = RepositoryRevision(
            profile_id=str(doc.profile_id), repository_id=str(repo.id), commit_sha="b" * 40
        )
        db.add(other)
        await db.commit()
        db.add(
            Evidence(
                profile_id=str(doc.profile_id),
                review_id=str(finding.review_id),
                revision_id=other.id,
                finding_id=str(finding.id),
                source_id=str(doc.id),
                line_start=1,
                line_end=1,
                quote="first line",
            )
        )
        with pytest.raises(IntegrityError):
            await db.commit()
        await db.rollback()
        assert await db.scalar(select(func.count()).select_from(Evidence)) == 1


async def test_duplicate_chunk_transaction_leaves_no_partial_source(database, actors):
    async with database() as db:
        uid, repo, rev, doc, chunks, finding = await setup_document(db, actors)
        new = doc.model_copy(update={"id": uuid4(), "path": "src/another.py"})
        chunk = chunks[0].model_copy(update={"id": uuid4(), "source_id": new.id})
        with pytest.raises(ValueError):
            await service.persist_document(db, uid, new, [chunk, chunk])
        assert await db.get(IngestedSource, str(new.id)) is None
        # A real uniqueness failure after source insertion also rolls back everything.
        bad = chunk.model_copy(update={"id": chunks[0].id})
        with pytest.raises(IntegrityError):
            await service.persist_document(db, uid, new, [bad])
        assert await db.get(IngestedSource, str(new.id)) is None


async def test_delete_profile_cascades_new_entities(database, actors):
    from core.services.profile_service import delete_profile

    async with database() as db:
        uid, repo, rev, doc, chunks, finding = await setup_document(db, actors)
        await service.persist_finding(db, uid, finding)
        await delete_profile(db, doc.profile_id, uid)
        for model in (
            Repository,
            RepositoryRevision,
            IngestedSource,
            SourceChunk,
            Finding,
            Evidence,
        ):
            assert await db.scalar(select(func.count()).select_from(model)) == 0


@pytest.mark.parametrize(
    "changes",
    [
        {"commit_sha": "main"},
        {"commit_sha": "g" * 40},
        {"commit_sha": "A" * 40},
    ],
)
async def test_invalid_sha_contract(changes):
    with pytest.raises(ValidationError):
        RevisionContract(id=uuid4(), profile_id=uuid4(), repository_id=uuid4(), **changes)


@pytest.mark.parametrize(
    "path", ["../secret", "/etc/passwd", "a/../secret", "C:/secret", "a\\b", ""]
)
async def test_invalid_document_paths(path):
    with pytest.raises(ValidationError):
        SourceDocument(
            id=uuid4(),
            profile_id=uuid4(),
            source_type="repo",
            revision_id=uuid4(),
            path=path,
            text="x",
            content_hash=hashlib.sha256(b"x").hexdigest(),
        )


async def test_hash_and_line_contracts():
    with pytest.raises(ValidationError):
        SourceDocument(
            id=uuid4(), profile_id=uuid4(), source_type="resume", text="x", content_hash="0" * 64
        )
    with pytest.raises(ValidationError):
        TextChunk(
            id=uuid4(),
            profile_id=uuid4(),
            source_id=uuid4(),
            chunk_index=0,
            text="x",
            line_start=3,
            line_end=2,
        )
    chunks = source_line_chunks("a\nb\nc", {"source_id": "source", "profile_id": "profile"}, 2)
    assert [(c.text, c.metadata["line_start"], c.metadata["line_end"]) for c in chunks] == [
        ("a\nb", 1, 2),
        ("c", 3, 3),
    ]
    assert chunks[0].id != chunks[1].id


async def test_github_metadata_raw_and_tool_aliases():
    raw = {
        "name": "project",
        "language": "Python",
        "stargazers_count": 9,
        "forks_count": 3,
        "pushed_at": "2026-01-01",
        "html_url": "https://github.com/example/project",
    }
    tool = {
        "name": "project",
        "primary_language": "Python",
        "star_count": 9,
        "fork_count": 3,
        "last_commit_date": "2026-01-01",
        "repo_url": "https://github.com/example/project",
    }
    assert RepoAnalyzer().parse(raw).text == RepoAnalyzer().parse(tool).text
    assert "commit_sha" not in normalize_github_metadata(tool)


async def test_new_services_reject_inactive_principal(database, actors):
    from core.models import User

    async with database() as db:
        uid, repo, rev, doc, chunks, finding = await setup_document(db, actors)
        await db.execute(update(User).where(User.id == uid).values(is_active=False))
        await db.commit()
        for call in [
            lambda: service.register_revision(db, uid, repo, rev),
            lambda: service.persist_document(db, uid, doc, chunks),
            lambda: service.persist_finding(db, uid, finding),
            lambda: service.get_findings(db, uid, finding.review_id),
        ]:
            with pytest.raises(HTTPException) as denied:
                await call()
            assert denied.value.status_code == 401


async def test_revision_bound_review_creation_is_owner_scoped(database, actors):
    from core.services.review_service import create_review

    async with database() as db:
        uid, repo, rev, doc, chunks, finding = await setup_document(db, actors)
        from tests.analysis_fixtures import verified_ingestion
        source=await verified_ingestion(db,doc.profile_id,uid)
        review = await create_review(db, doc.profile_id, uid, source.revision_id)
        assert review.revision_id == source.revision_id
        with pytest.raises(HTTPException) as denied:
            await create_review(db, actors["profiles"][1].id, actors["users"][1].id, rev.id)
        assert denied.value.status_code == 404


async def test_job_upload_source_is_real_json_and_remote_placeholder_is_not_persisted(
    database, actors
):
    from core.services.review_service import process_review

    async with database() as db:
        profile = await db.get(Profile, actors["profiles"][0].id)
        profile.resume_text = "Caller supplied resume"
        profile.resume_filename = "resume.md"
        profile.github_username = "example"
        profile.portfolio_url = "https://example.invalid"
        await db.commit()
        await process_review(db, actors["reviews"][0].id, profile.id, actors["users"][0].id)
        sources = (await db.scalars(select(IngestedSource))).all()
        assert sources == []  # Unsupported legacy work cannot manufacture new source records.
        result = await db.get(Review, actors["reviews"][0].id)
        assert result.status == "failed" and result.sections is None and result.overall_score is None
        assert await db.scalar(select(func.count()).select_from(Evidence)) == 0

async def test_same_revision_path_cannot_hold_conflicting_sources(database,actors):
    async with database() as db:
        uid,repo,rev,doc,chunks,finding=await setup_document(db,actors)
        duplicate=doc.model_copy(update={"id":uuid4()})
        with pytest.raises(IntegrityError):
            await service.persist_document(db,uid,duplicate,[])
        assert await db.scalar(select(func.count()).select_from(IngestedSource))==1
        assert await db.get(IngestedSource,str(duplicate.id)) is None

async def test_request_session_exception_rolls_back_real_writes(database,actors,monkeypatch):
    from core import database as module
    monkeypatch.setattr(module,"AsyncSessionLocal",database)
    generator=module.get_db()
    session=await anext(generator)
    ident=str(uuid4())
    session.add(IngestedSource(id=ident,profile_id=actors["profiles"][0].id,source_type="resume"))
    await session.flush()
    with pytest.raises(RuntimeError,match="fixture"):
        await generator.athrow(RuntimeError("fixture"))
    async with database() as db:
        assert await db.get(IngestedSource,ident) is None
