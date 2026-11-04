"""Real PostgreSQL/SQLite API+service ingestion; only GitHub HTTP is mocked."""

import asyncio
from uuid import uuid4
import httpx
import pytest
from fastapi import HTTPException
from sqlalchemy import select, func, update
from core.models import Review, User, IngestedSource
from core.models.repository import Repository, RepositoryRevision
from core.models.evidence import SourceChunk, Evidence
from core.security import create_access_token
from core.services import github_ingestion as service
from core.services import review_service
from ingestion.github import GitHubClient, Limits
from tests.github_responses import GitHubResponses

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


@pytest.fixture
def github(monkeypatch, database):
    fixture = GitHubResponses()
    monkeypatch.setattr(service, "AsyncSessionLocal", database)
    monkeypatch.setattr(
        service,
        "make_github_client",
        lambda: GitHubClient(
            "server-only-fixture-token",
            transport=httpx.MockTransport(fixture),
            limits=getattr(fixture, "limits", None),
        ),
    )
    return fixture


def bearer(actors, index=0):
    return {"Authorization": "Bearer " + create_access_token({"sub": actors["users"][index].id})}


async def submit(client, actors, index=0, name="example/project"):
    response = await client.post(
        "/repository-ingestions",
        headers=bearer(actors, index),
        json={"profile_id": actors["profiles"][index].id, "repository": name},
    )
    assert response.status_code == 202, response.text
    return response.json()


async def status(client, actors, job, index=0):
    # Submission is durable; this test helper explicitly runs the separate worker before inspecting results.
    from workers.run import tick
    await tick(service.AsyncSessionLocal, "ingestion-regression-worker")

    response = await client.get(
        "/repository-ingestions/" + job["id"], headers=bearer(actors, index)
    )
    assert response.status_code == 200, response.text
    return response.json()


async def test_end_to_end_verified_content_and_results(client, database, actors, github):
    job = await submit(client, actors, name="https://github.com/Example/Project")
    assert job["status"] == "pending"
    result = await status(client, actors, job)
    assert result["status"] == "complete"
    manifest = result["ingestion_result"]
    assert manifest["verified"] is True and manifest["commit_sha"] == github.sha
    assert manifest["file_count"] == 3 and manifest["chunk_count"] == 3
    response = await client.get(
        "/repository-ingestions/" + job["id"] + "/sources?page=1&page_size=1",
        headers=bearer(actors),
    )
    assert response.status_code == 200 and response.json()["total"] == 3
    first = response.json()["items"][0]
    second = await client.get(
        "/repository-ingestions/" + job["id"] + "/sources?page=2&page_size=1",
        headers=bearer(actors),
    )
    assert second.json()["items"][0]["id"] != first["id"]
    detail = await client.get(
        "/repository-ingestions/" + job["id"] + "/sources/" + first["id"], headers=bearer(actors)
    )
    assert detail.status_code == 200, detail.text
    source = detail.json()["source"]
    assert source["path"] == "README.md" and source["text"] == "# Real fixture\nContent\n"
    assert source["source_metadata"]["blob_sha"] in github.blobs
    assert (
        detail.json()["chunks"][0]["line_start"] == 1
        and detail.json()["chunks"][0]["line_end"] == 2
    )
    assert "server-only-fixture-token" not in str(result) + detail.text
    async with database() as db:
        assert await db.scalar(select(func.count()).select_from(Repository)) == 1
        assert await db.scalar(select(func.count()).select_from(RepositoryRevision)) == 1
        assert await db.scalar(select(func.count()).select_from(Evidence)) == 0
    # Legacy review history and analysis execution cannot treat an ingestion as an AI report.
    listing = await client.get("/reviews", headers=bearer(actors))
    assert listing.json()["total"] == 0  # Historical pending rows are not real analysis reports.
    assert (await client.get("/reviews/" + job["id"], headers=bearer(actors))).status_code == 404
    async with database() as db:
        with pytest.raises(HTTPException):
            await review_service.process_review(
                db, job["id"], actors["profiles"][0].id, actors["users"][0].id
            )


async def test_repeated_revision_is_idempotent_and_does_not_refetch_blobs(
    client, database, actors, github
):
    first = await submit(client, actors)
    assert (await status(client,actors,first))["status"] == "complete"
    before = len([r for r in github.requests if "/git/blobs/" in r.url.path])
    second = await submit(client, actors)
    result = await status(client, actors, second)
    assert result["ingestion_result"]["deduplicated"] is True
    assert len([r for r in github.requests if "/git/blobs/" in r.url.path]) == before
    assert (await status(client, actors, first))["revision_id"] == result["revision_id"]
    async with database() as db:
        for model, count in [
            (Repository, 1),
            (RepositoryRevision, 1),
            (IngestedSource, 3),
            (SourceChunk, 3),
        ]:
            assert await db.scalar(select(func.count()).select_from(model)) == count


@pytest.mark.parametrize("operation", ["submit", "status", "list", "detail"])
async def test_foreign_user_cannot_access_ingestion(client, database, actors, github, operation):
    job = await submit(client, actors)
    result = await status(client, actors, job)
    path = "/repository-ingestions/" + job["id"]
    if operation == "submit":
        response = await client.post(
            "/repository-ingestions",
            headers=bearer(actors, 1),
            json={"profile_id": actors["profiles"][0].id, "repository": "example/project"},
        )
    else:
        if operation == "list":
            path += "/sources"
        if operation == "detail":
            path += "/sources/" + result["ingestion_result"]["files"][0]["id"]
        response = await client.get(path, headers=bearer(actors, 1))
    assert response.status_code == 404


@pytest.mark.parametrize("operation", ["submit", "status", "list", "detail"])
async def test_missing_auth_is_rejected(client, actors, github, operation):
    job = await submit(client, actors)
    path = "/repository-ingestions/" + job["id"]
    if operation == "submit":
        response = await client.post(
            "/repository-ingestions",
            json={"profile_id": actors["profiles"][0].id, "repository": "example/project"},
        )
    else:
        if operation == "list":
            path += "/sources"
        if operation == "detail":
            path += "/sources/" + str(uuid4())
        response = await client.get(path)
    assert response.status_code == 401


async def test_direct_service_and_inactive_guards(database, actors, github):
    uid, pid = actors["users"][0].id, actors["profiles"][0].id
    async with database() as db:
        with pytest.raises(HTTPException) as denied:
            await service.submit(db, pid, actors["users"][1].id, "example/project")
        assert denied.value.status_code == 404
        job, _ = await service.submit(db, pid, uid, "example/project")
        job_id = job.id
        with pytest.raises(HTTPException):
            await service.get_ingestion(db, job.id, actors["users"][1].id)
        await db.rollback()
        await db.execute(update(User).where(User.id == uid).values(is_active=False))
        await db.commit()
        with pytest.raises(HTTPException) as denied:
            await service.submit(db, pid, uid, "example/project")
        assert denied.value.status_code == 401
        await service.run_job(job_id, uid)
    assert not github.requests


@pytest.mark.parametrize(
    "failure", ["rate_limit", "missing", "private", "network", "integrity", "timeout"]
)
async def test_remote_failure_is_recorded_without_snapshot(
    client, database, actors, github, failure
):
    if failure == "rate_limit":
        github.overrides[""] = lambda r: httpx.Response(429, json={})
    if failure == "missing":
        github.overrides[""] = lambda r: httpx.Response(404, json={})
    if failure == "private":
        github.overrides[""] = {"full_name": github.name, "private": True, "default_branch": "main"}
    if failure == "network":

        def fail(r):
            raise httpx.ConnectError("external fixture", request=r)

        github.overrides[""] = fail
    if failure == "integrity":
        github.trees[github.tree_sha]["tree"][0]["sha"] = "a" * 40
    if failure == "timeout":
        github.limits = Limits(seconds=0.01)

        async def slow(r):
            await asyncio.sleep(0.05)
            return httpx.Response(200, json={})

        # MockTransport supports async external handlers.
        github.overrides[""] = lambda r: httpx.Response(503, json={})
    job = await submit(client, actors)
    result = await status(client, actors, job)
    if failure in {"rate_limit", "network", "timeout"}:
        assert result["status"] == "pending" and result["retry_at"]
        assert result["progress_stage"] == "retry_waiting"
        for _ in range(2):
            async with database() as db:
                current = await db.get(Review,job["id"])
                current.retry_at = None
                await db.commit()
            result = await status(client,actors,job)
    assert result["status"] == "failed" and result["error_message"]
    if failure == "timeout":
        assert result["ingestion_result"]["error_code"] == "timeout"
    assert result["revision_id"] is None
    async with database() as db:
        for model in (Repository, RepositoryRevision, IngestedSource, SourceChunk):
            assert await db.scalar(select(func.count()).select_from(model)) == 0


async def test_sql_failure_rolls_back_all_content_and_marks_failed(
    client, database, actors, github
):
    async with database() as db:
        from sqlalchemy import text

        if db.bind.dialect.name == "postgresql":
            await db.execute(
                text(
                    "CREATE FUNCTION reject_second_source() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN IF NEW.path='package.json' THEN RAISE EXCEPTION 'fixture failure' USING ERRCODE='23514'; END IF; RETURN NEW; END; $$"
                )
            )
            await db.execute(
                text(
                    "CREATE TRIGGER reject_second_source BEFORE INSERT ON ingested_sources FOR EACH ROW EXECUTE FUNCTION reject_second_source()"
                )
            )
        else:
            await db.execute(
                text(
                    "CREATE TRIGGER reject_second_source BEFORE INSERT ON ingested_sources WHEN NEW.path='package.json' BEGIN SELECT RAISE(ABORT,'fixture failure'); END"
                )
            )
        await db.commit()
    job = await submit(client, actors)
    result = await status(client, actors, job)
    assert (
        result["status"] == "failed"
        and result["ingestion_result"]["error_code"] == "ingestion_failed"
    )
    async with database() as db:
        for model in (Repository, RepositoryRevision, IngestedSource, SourceChunk):
            assert await db.scalar(select(func.count()).select_from(model)) == 0


async def test_active_jobs_reuse_same_input_and_reject_other_repository(database, actors, github):
    async with database() as db:
        uid, pid = actors["users"][0].id, actors["profiles"][0].id
        first, created = await service.submit(db, pid, uid, "example/project")
        same, again = await service.submit(db, pid, uid, "example/project")
        assert same.id == first.id and created and not again
        with pytest.raises(HTTPException) as busy:
            await service.submit(db, pid, uid, "example/other")
        assert busy.value.status_code == 409


async def test_input_validation_and_pagination_bounds(client, actors, github):
    for name in ["https://localhost/private", "https://github.com/a/b?token=x", "a/.."]:
        response = await client.post(
            "/repository-ingestions",
            headers=bearer(actors),
            json={"profile_id": actors["profiles"][0].id, "repository": name},
        )
        assert response.status_code == 422
    assert not github.requests
    job = await submit(client, actors)
    response = await client.get(
        "/repository-ingestions/" + job["id"] + "/sources?page_size=100", headers=bearer(actors)
    )
    assert response.status_code == 422


async def test_database_partial_active_uniqueness(database, actors, github):
    from sqlalchemy.exc import IntegrityError

    async with database() as db:
        uid, pid = actors["users"][0].id, actors["profiles"][0].id
        await service.submit(db, pid, uid, "example/project")
        db.add(
            Review(
                profile_id=pid,
                job_type="github_ingestion",
                requested_repository="example/other",
                status="pending",
            )
        )
        with pytest.raises(IntegrityError):
            await db.commit()
        await db.rollback()


@pytest.mark.parametrize("operation", ["submit", "status", "list", "detail"])
async def test_inactive_tokens_rejected_on_new_endpoints(
    client, database, actors, github, operation
):
    job = await submit(client, actors)
    result = await status(client, actors, job)
    headers = bearer(actors)
    async with database() as db:
        await db.execute(
            update(User).where(User.id == actors["users"][0].id).values(is_active=False)
        )
        await db.commit()
    path = "/repository-ingestions/" + job["id"]
    if operation == "submit":
        response = await client.post(
            "/repository-ingestions",
            headers=headers,
            json={"profile_id": actors["profiles"][0].id, "repository": "example/project"},
        )
    else:
        if operation == "list":
            path += "/sources"
        if operation == "detail":
            path += "/sources/" + result["ingestion_result"]["files"][0]["id"]
        response = await client.get(path, headers=headers)
    assert response.status_code == 401


async def test_repository_limit_is_enforced_before_network(client, database, actors, github):
    from core.config import settings

    async with database() as db:
        db.add_all(
            Repository(
                profile_id=actors["profiles"][0].id,
                full_name="other/repo" + str(i),
                provider="github",
            )
            for i in range(settings.max_repos_per_profile)
        )
        await db.commit()
    response = await client.post(
        "/repository-ingestions",
        headers=bearer(actors),
        json={"profile_id": actors["profiles"][0].id, "repository": "example/project"},
    )
    assert response.status_code == 422 and not github.requests


@pytest.mark.parametrize("corruption", ["delete", "edit"])
async def test_corrupted_snapshot_is_not_reported_as_dedup_success(
    client, database, actors, github, corruption
):
    first = await submit(client, actors)
    assert (await status(client,actors,first))["status"] == "complete"
    async with database() as db:
        chunk = await db.scalar(select(SourceChunk).limit(1))
        if corruption == "delete":
            await db.delete(chunk)
        else:
            chunk.text = "corrupt"
        await db.commit()
    second = await submit(client, actors)
    result = await status(client, actors, second)
    assert (
        result["status"] == "failed"
        and result["ingestion_result"]["error_code"] == "snapshot_conflict"
    )


async def test_revocation_during_fetch_cannot_commit_content(client, database, actors, github):
    oid = next(iter(github.blobs))

    async def revoked(request):
        async with database() as db:
            await db.execute(
                update(User).where(User.id == actors["users"][0].id).values(is_active=False)
            )
            await db.commit()
        return httpx.Response(200, json=github.blobs[oid])

    github.overrides["git/blobs/" + oid] = revoked
    job = await submit(client, actors)
    async with database() as db:
        stored = await db.get(Review, job["id"])
        assert stored.status != "complete" and stored.revision_id is None
        assert await db.scalar(select(func.count()).select_from(IngestedSource)) == 0


async def test_concurrent_submission_reuses_one_active_job(database, actors, github):
    async with database() as db:
        if db.bind.dialect.name != "postgresql":
            pytest.skip("PostgreSQL row-lock verification")

    async def create():
        async with database() as db:
            job, _ = await service.submit(
                db, actors["profiles"][0].id, actors["users"][0].id, "example/project"
            )
            return job.id

    ids = await asyncio.gather(create(), create())
    assert ids[0] == ids[1]
    async with database() as db:
        assert (
            await db.scalar(
                select(func.count())
                .select_from(Review)
                .where(Review.job_type == "github_ingestion")
            )
            == 1
        )


async def test_invalid_client_configuration_is_recorded(
    client, database, actors, github, monkeypatch
):
    monkeypatch.setattr(service, "make_github_client", lambda: GitHubClient("bad\nkey"))
    job = await submit(client, actors)
    result = await status(client, actors, job)
    assert (
        result["status"] == "failed" and result["ingestion_result"]["error_code"] == "credentials"
    )
    assert not github.requests
