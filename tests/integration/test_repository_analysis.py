"""Real database/API analysis. Only upstream GitHub HTTP uses deterministic fixtures."""

import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from fastapi import HTTPException
from sqlalchemy import func, select, update

from core.models import Evidence, Finding, Review, User
from core.security import create_access_token
from core.services import repository_analysis as service
from tests.analysis_fixtures import verified_ingestion
from tests.database_probes import reject_insert, revoke_processing_job

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


def bearer(actors, index=0):
    return {"Authorization": "Bearer " + create_access_token({"sub": actors["users"][index].id})}


async def ingested(database, actors, index=0, files=None):
    async with database() as db:
        return await verified_ingestion(
            db, actors["profiles"][index].id, actors["users"][index].id, files
        )


async def start(client, actors, ingestion_id):
    response = await client.post(
        "/analyses", headers=bearer(actors), json={"ingestion_id": ingestion_id}
    )
    assert response.status_code == 202, response.text
    result = response.json()
    assert result["status"] == "pending"
    from workers.run import tick
    await tick(service.AsyncSessionLocal, "analysis-regression-worker")
    return result


async def test_real_multifile_analysis_api_and_report(client, database, actors):
    source = await ingested(database, actors)
    job = await start(client, actors, source.id)
    response = await client.get("/analyses/" + job["id"] + "/report", headers=bearer(actors))
    assert response.status_code == 200, response.text
    report = response.json()
    assert (
        report["producer"] == "deterministic"
        and report["commit_sha"] == source.ingestion_result["commit_sha"]
    )
    assert report["summary"]["languages"]["Python"] == 2
    assert len(report["findings"]) == 4
    for finding in report["findings"]:
        assert finding["fingerprint"] and finding["rule_id"].startswith("PY-")
        for span in finding["evidence"]:
            detail = await client.get(
                "/repository-ingestions/" + source.id + "/sources/" + span["source_id"],
                headers=bearer(actors),
            )
            stored = detail.json()["source"]
            assert span["quote"] == "\n".join(
                stored["text"].splitlines()[span["line_start"] - 1 : span["line_end"]]
            )
            assert span["path"] == stored["path"]
    for suffix in ("", "/summary", "/findings"):
        result = await client.get("/analyses/" + job["id"] + suffix, headers=bearer(actors))
        assert result.status_code == 200, result.text
    async with database() as db:
        row = await db.get(Review, job["id"])
        assert row.status == "complete" and row.sections is None and row.overall_score is None


async def test_repeat_and_concurrent_submissions_are_idempotent(database, actors):
    source = await ingested(database, actors)

    async def submit():
        async with database() as db:
            job, _ = await service.submit(db, source.id, actors["users"][0].id)
            return job.id

    async with database() as db:
        postgres = db.bind.dialect.name == "postgresql"
    ids = await asyncio.gather(submit(), submit()) if postgres else [await submit(), await submit()]
    assert ids[0] == ids[1]
    async with database() as db:
        await service.process(db, ids[0], actors["users"][0].id)
        job, created = await service.submit(db, source.id, actors["users"][0].id)
        assert not created and job.id == ids[0]
        assert await db.scalar(select(func.count()).select_from(Finding)) == 4
        assert await db.scalar(select(func.count()).select_from(Evidence)) == 4


@pytest.mark.parametrize("suffix", ["", "/summary", "/findings", "/report"])
async def test_foreign_and_unauthenticated_reads(client, database, actors, suffix):
    source = await ingested(database, actors)
    job = await start(client, actors, source.id)
    path = "/analyses/" + job["id"] + suffix
    assert (await client.get(path, headers=bearer(actors, 1))).status_code == 404
    assert (await client.get(path)).status_code == 401
    async with database() as db:
        with pytest.raises(HTTPException) as denied:
            await service.report(db, job["id"], actors["users"][1].id)
        assert denied.value.status_code == 404


async def test_foreign_missing_incomplete_and_inactive_submission(client, database, actors):
    source = await ingested(database, actors, 1)
    response = await client.post(
        "/analyses", headers=bearer(actors), json={"ingestion_id": source.id}
    )
    assert response.status_code == 404
    async with database() as db:
        from core.services.github_ingestion import submit

        pending, _ = await submit(
            db, actors["profiles"][0].id, actors["users"][0].id, "example/project"
        )
    response = await client.post(
        "/analyses", headers=bearer(actors), json={"ingestion_id": pending.id}
    )
    assert response.status_code == 409
    assert (await client.post("/analyses", json={"ingestion_id": source.id})).status_code == 401
    async with database() as db:
        await db.execute(
            update(User).where(User.id == actors["users"][0].id).values(is_active=False)
        )
        await db.commit()
        with pytest.raises(HTTPException) as denied:
            await service.submit(db, pending.id, actors["users"][0].id)
        assert denied.value.status_code == 401


async def test_transaction_rolls_back_findings_and_evidence(client, database, actors):
    source = await ingested(database, actors)
    async with database() as db:
        await reject_insert(db, "reject_evidence", "evidence_spans", "fixture integrity failure")
        await db.commit()
    job = await start(client, actors, source.id)
    result = await client.get("/analyses/" + job["id"], headers=bearer(actors))
    assert result.json()["status"] == "failed"
    assert (
        await client.get("/analyses/" + job["id"] + "/report", headers=bearer(actors))
    ).status_code == 409
    async with database() as db:
        assert await db.scalar(select(func.count()).select_from(Finding)) == 0
        assert await db.scalar(select(func.count()).select_from(Evidence)) == 0


async def test_expired_job_is_failed_and_retryable(database, actors):
    from core.services import job_queue
    source = await ingested(database, actors)
    uid = actors["users"][0].id
    async with database() as db:
        first, _ = await service.submit(db, source.id, uid)
        ident = first.id
        attempt = await job_queue.claim(db, "expired-worker", ident)
        await db.execute(update(Review).where(Review.id == ident).values(
            lease_expires_at=datetime.now(UTC)-timedelta(seconds=1)))
        await db.commit()
        # Reads no longer perform recovery. The independent queue recovers automatically.
        assert (await service.get_job(db, ident, uid)).status == "processing"
        await db.rollback()
        assert await job_queue.recover(db) == 1
        retried = await db.get(Review,ident)
        assert retried.status == "pending" and retried.retry_at is not None
        retry, created = await service.submit(db, source.id, uid)
        assert not created and retry.id == ident
        with pytest.raises(job_queue.LostAttemptError):
            await attempt.checkpoint()

@pytest.mark.parametrize("revocation", ["inactive", "ownership_transfer"])
async def test_activity_and_ownership_rechecked_before_results(database, actors, revocation):
    source = await ingested(database, actors)
    uid, pid = actors["users"][0].id, actors["profiles"][0].id
    async with database() as db:
        job, _ = await service.submit(db, source.id, uid)
        ident = job.id
        if revocation == "inactive":
            action = "UPDATE users SET is_active=FALSE WHERE id='" + uid.replace("-", "") + "';"
        else:
            action = (
                "UPDATE profiles SET user_id='"
                + actors["users"][1].id.replace("-", "")
                + "' WHERE id='"
                + pid.replace("-", "")
                + "';"
            )
        await revoke_processing_job(db, action)
        await db.commit()
        await service.process(db, ident, uid)
    async with database() as db:
        assert (await db.get(Review, ident)).status != "complete"
        assert await db.scalar(select(func.count()).select_from(Finding)) == 0


async def test_zero_issues_is_truthful_and_legacy_outputs_are_disabled(client, database, actors):
    source = await ingested(
        database,
        actors,
        files={"README.md": b"# Small project\n", "app.py": b"def add(a,b):\n    return a+b\n"},
    )
    job = await start(client, actors, source.id)
    result = (await client.get("/analyses/" + job["id"] + "/report", headers=bearer(actors))).json()
    assert result["findings"] == [] and "score" not in result
    response = await client.post(
        "/reviews", headers=bearer(actors), json={"profile_id": actors["profiles"][0].id}
    )
    assert response.status_code == 422
    async with database() as db:
        legacy = await db.get(Review, actors["reviews"][0].id)
        legacy.status = "complete"
        legacy.report_kind = "demo"
        legacy.sections = [{"fabricated": "historical"}]
        legacy.overall_score = 0.99
        await db.commit()
    response = await client.get("/reviews/" + legacy.id, headers=bearer(actors))
    assert response.status_code == 410 and "historical" not in response.text.lower().replace(
        "historical placeholder review", ""
    )
    listed = (await client.get("/reviews", headers=bearer(actors))).json()
    assert all(item["id"] != legacy.id for item in listed["items"])
    async with database() as db:
        assert (await db.get(Review, legacy.id)).sections == [{"fabricated": "historical"}]


async def test_modified_evidence_cannot_be_served(client, database, actors):
    source = await ingested(database, actors)
    job = await start(client, actors, source.id)
    async with database() as db:
        span = await db.scalar(select(Evidence).limit(1))
        span.quote = "invented excerpt"
        await db.commit()
    response = await client.get("/analyses/" + job["id"] + "/report", headers=bearer(actors))
    assert response.status_code == 409


async def test_expired_ingestion_does_not_indefinitely_block_retry(database, actors):
    from core.services import github_ingestion, job_queue
    async with database() as db:
        first, _ = await github_ingestion.submit(
            db,actors["profiles"][0].id,actors["users"][0].id,"example/project")
        ident=first.id
        await job_queue.claim(db,"crashed-ingestion",ident)
        await db.execute(update(Review).where(Review.id==ident).values(
            lease_expires_at=datetime.now(UTC)-timedelta(seconds=1)))
        await db.commit()
        assert await job_queue.recover(db)==1
        previous=await db.get(Review,ident)
        assert previous.status=="pending" and previous.retry_at is not None
        retry,created=await github_ingestion.submit(
            db,actors["profiles"][0].id,actors["users"][0].id,"example/project")
        assert not created and retry.id==ident


async def test_inactive_user_cannot_read_any_new_analysis_resource(client, database, actors):
    source = await ingested(database, actors)
    job = await start(client, actors, source.id)
    async with database() as db:
        await db.execute(
            update(User).where(User.id == actors["users"][0].id).values(is_active=False)
        )
        await db.commit()
    for suffix in ("", "/summary", "/findings", "/report"):
        assert (
            await client.get("/analyses/" + job["id"] + suffix, headers=bearer(actors))
        ).status_code == 401
