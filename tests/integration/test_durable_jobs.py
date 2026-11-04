"""Real PostgreSQL lifecycle, fencing, cancellation and process-crash checks."""

import asyncio
import os
import sys
from datetime import timedelta

import httpx
import pytest
from sqlalchemy import func, select, update

from core.models import Evidence, Finding, IngestedSource, Review, User
from core.security import create_access_token
from core.services import github_ingestion, repository_analysis
from core.services import job_queue as queue
from ingestion.github import GitHubClient, GitHubError
from tests.analysis_fixtures import verified_ingestion
from tests.github_responses import GitHubResponses
from workers.run import run_attempt, tick

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


def bearer(actors, index=0):
    return {"Authorization": "Bearer " + create_access_token({"sub": actors["users"][index].id})}


async def ingestion(database, actors):
    async with database() as db:
        job, _ = await github_ingestion.submit(
            db, actors["profiles"][0].id, actors["users"][0].id, "example/project"
        )
        return job.id


async def analysis(database, actors):
    async with database() as db:
        source = await verified_ingestion(db, actors["profiles"][0].id, actors["users"][0].id)
        job, _ = await repository_analysis.submit(db, source.id, actors["users"][0].id)
        return job.id


async def claim(database, ident, worker="worker"):
    async with database() as db:
        return await queue.claim(db, worker, ident)


async def state(database, ident):
    async with database() as db:
        return await db.get(Review, ident)


async def expire(database, ident):
    async with database() as db:
        await db.execute(
            update(Review)
            .where(Review.id == ident)
            .values(lease_expires_at=queue.now() - timedelta(seconds=1))
        )
        await db.commit()


@pytest.fixture
def github(monkeypatch):
    response = GitHubResponses()
    monkeypatch.setattr(
        github_ingestion,
        "make_github_client",
        lambda: GitHubClient(transport=httpx.MockTransport(response)),
    )
    return response


async def test_offline_submission_and_independent_discovery(client, database, actors, github):
    response = await client.post(
        "/repository-ingestions",
        headers=bearer(actors),
        json={"profile_id": actors["profiles"][0].id, "repository": "example/project"},
    )
    assert response.status_code == 202
    job = response.json()
    assert job["status"] == "pending" and not github.requests
    assert (await state(database, job["id"])).attempt_count == 0
    # New session/worker discovers the committed row with no request object or task.
    assert await tick(database, "independent-worker")
    stored = await state(database, job["id"])
    assert stored.status == "complete" and stored.progress_pct == 100
    assert stored.attempt_count == 1 and stored.ingestion_result["verified"]
    assert stored.created_at <= stored.started_at <= stored.completed_at
    assert not await tick(database, "second-worker")


async def test_two_workers_claim_exactly_once(database, actors):
    ident = await ingestion(database, actors)
    a, b = await asyncio.gather(claim(database, ident, "one"), claim(database, ident, "two"))
    assert (a is None) != (b is None)
    stored = await state(database, ident)
    assert stored.attempt_count == 1 and stored.execution_token == (a or b).token


async def test_heartbeat_renews_and_stale_progress_is_fenced(database, actors):
    ident = await ingestion(database, actors)
    old = await claim(database, ident)
    stamp = (await state(database, ident)).lease_expires_at
    await old.checkpoint("fetching_tree")
    assert (await state(database, ident)).lease_expires_at > stamp
    await expire(database, ident)
    async with database() as db:
        assert await queue.recover(db) == 1
        row = await db.get(Review, ident)
        row.retry_at = queue.now() - timedelta(seconds=1)
        await db.commit()
    newer = await claim(database, ident, "new-worker")
    assert newer.number == 2 and newer.token != old.token
    with pytest.raises(queue.LostAttemptError):
        await old.checkpoint("complete", 100)
    assert (await state(database, ident)).progress_stage == "claimed"


async def test_two_recoverers_and_retry_exhaustion(database, actors, monkeypatch):
    monkeypatch.setattr(queue.settings, "job_retry_base_seconds", 0)
    ident = await ingestion(database, actors)
    for number in range(1, 4):
        attempt = await claim(database, ident, "worker-" + str(number))
        assert attempt.number == number
        await expire(database, ident)

        async def recover():
            async with database() as db:
                return await queue.recover(db)

        counts = await asyncio.gather(recover(), recover())
        assert sum(counts) == 1
    row = await state(database, ident)
    assert row.status == "failed" and row.attempt_count == 3 and row.retry_at is None
    assert await claim(database, ident) is None


@pytest.mark.parametrize(
    "code,retry",
    [
        ("rate_limited", True),
        ("network", True),
        ("upstream_unavailable", True),
        ("not_public", False),
        ("unsafe_path", False),
        ("integrity", False),
        ("credentials", False),
    ],
)
async def test_explicit_error_classification(database, actors, code, retry):
    ident = await ingestion(database, actors)
    attempt = await claim(database, ident)
    await queue.fail(database, attempt, GitHubError(code, "Safe upstream error"))
    row = await state(database, ident)
    assert row.status == ("pending" if retry else "failed")
    assert row.retry_at is not None if retry else row.retry_at is None
    if retry:
        assert row.progress_stage == "retry_waiting"
        assert await claim(database, ident) is None


@pytest.mark.parametrize("running", [False, True])
async def test_owner_cancellation_is_persisted_and_foreign_ids_hidden(
    client, database, actors, running
):
    ident = await ingestion(database, actors)
    attempt = await claim(database, ident) if running else None
    assert (
        await client.post("/jobs/" + ident + "/cancel", headers=bearer(actors, 1))
    ).status_code == 404
    assert (await client.get("/jobs/" + ident, headers=bearer(actors, 1))).status_code == 404
    assert (await client.post("/jobs/" + ident + "/cancel")).status_code == 401
    result = await client.post("/jobs/" + ident + "/cancel", headers=bearer(actors))
    assert result.status_code == 200 and result.json()["cancel_requested_at"]
    assert "execution_token" not in result.json() and "worker_id" not in result.json()
    if running:
        with pytest.raises(queue.CancellationRequestedError):
            await attempt.checkpoint()
        await queue.fail(database, attempt, queue.CancellationRequestedError())
    row = await state(database, ident)
    assert row.status == "cancelled" and row.retry_at is None
    assert await claim(database, ident) is None


async def test_cancellation_during_download_cannot_publish(database, actors):
    ident = await ingestion(database, actors)
    attempt = await claim(database, ident)
    response = GitHubResponses()
    calls = 0

    async def transport(request):
        nonlocal calls
        calls += 1
        if calls == 5:
            async with database() as db:
                await queue.cancel(db, ident, actors["users"][0].id)
        return response(request)

    async def execute(factory, owned):
        async with factory() as db:  # noqa: SIM117 - retain explicit fault-injection scope
            async with GitHubClient(transport=httpx.MockTransport(transport)) as client:
                await github_ingestion.process(db, ident, owned.user_id, client, attempt=owned)

    await run_attempt(database, attempt, execute)
    row = await state(database, ident)
    assert row.status == "cancelled" and row.revision_id is None
    async with database() as db:
        assert await db.scalar(select(func.count()).select_from(IngestedSource)) == 0


async def test_analysis_retry_is_atomic_and_preserves_evidence(database, actors):
    ident = await analysis(database, actors)
    old = await claim(database, ident, "crashed")
    await expire(database, ident)
    async with database() as db:
        await queue.recover(db)
        row = await db.get(Review, ident)
        row.retry_at = None
        await db.commit()
    assert await tick(database, "replacement")
    async with database() as db:
        result = await repository_analysis.report(db, ident, actors["users"][0].id)
        assert len(result["findings"]) == 4
        assert await db.scalar(select(func.count()).select_from(Finding)) == 4
        assert await db.scalar(select(func.count()).select_from(Evidence)) == 4
    # An old process trying final persistence cannot reuse its previous generation.
    async with database() as db:
        row = await db.get(Review, ident)
        with pytest.raises(queue.LostAttemptError):
            queue.complete(row, old)
    assert not await tick(database, "after-ack-loss")


async def test_inactive_user_worker_and_api_do_not_bypass_auth(client, database, actors, github):
    ident = await ingestion(database, actors)
    async with database() as db:
        await db.execute(
            update(User).where(User.id == actors["users"][0].id).values(is_active=False)
        )
        await db.commit()
    assert (
        await client.post("/jobs/" + ident + "/cancel", headers=bearer(actors))
    ).status_code == 401
    await tick(database, "worker")
    row = await state(database, ident)
    assert row.status == "failed" and row.error_category == "authorization"
    assert row.revision_id is None and not github.requests


async def test_graceful_task_stop_persists_retry(database, actors):
    ident = await ingestion(database, actors)
    attempt = await claim(database, ident)
    entered = asyncio.Event()

    async def blocked(factory, owned):
        entered.set()
        await asyncio.Event().wait()

    task = asyncio.create_task(run_attempt(database, attempt, blocked))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    row = await state(database, ident)
    assert row.status == "pending" and row.error_category == "worker_shutdown"
    assert row.execution_token is None


# Real child-process crashes. No database mocks or API-owned execution.
async def child(database, mode, stop_stage="", after_commit=False):
    engine = database.kw["bind"]
    if engine.dialect.name != "postgresql":
        pytest.skip("Real process recovery requires PostgreSQL")
    async with engine.connect() as conn:
        schema = await conn.scalar(select(func.current_schema()))
    code = """
import asyncio,os,httpx
from sqlalchemy.ext.asyncio import create_async_engine,async_sessionmaker
from core.services import github_ingestion
from ingestion.github import GitHubClient
from tests.github_responses import GitHubResponses
from workers.run import tick,execute
from core.services import job_queue as queue
async def main():
    engine=create_async_engine(os.environ['REPOLENS_TEST_DATABASE_URL'],connect_args={'server_settings':{'search_path':os.environ['VERIFY_SCHEMA']}})
    factory=async_sessionmaker(engine,expire_on_commit=False)
    github_ingestion.make_github_client=lambda:GitHubClient(transport=httpx.MockTransport(GitHubResponses()))
    async def controlled(factory,attempt):
        original=attempt.checkpoint
        async def checkpoint(stage=None,percent=None):
            await original(stage,percent)
            if stage and stage==os.environ.get('STOP_STAGE'):
                await asyncio.Event().wait()
        attempt.checkpoint=checkpoint
        await execute(factory,attempt)
        if os.environ.get('AFTER_COMMIT')=='1':
            await asyncio.Event().wait()
    if os.environ['CHILD_MODE']=='claim':
        async with factory() as db:  # noqa: SIM117 - retain explicit fault-injection scope
            await queue.claim(db,'child-'+str(os.getpid()))
    else:
        await tick(factory,'real-child',controlled)
    await engine.dispose()
asyncio.run(main())
"""
    env = os.environ.copy()
    env.update(
        VERIFY_SCHEMA=schema,
        CHILD_MODE=mode,
        STOP_STAGE=stop_stage,
        AFTER_COMMIT="1" if after_commit else "0",
    )
    return await asyncio.create_subprocess_exec(
        sys.executable,
        "-B",
        "-c",
        code,
        env=env,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )


async def wait_stage(database, ident, stage):
    for _ in range(200):
        row = await state(database, ident)
        if row.progress_stage == stage:
            return row
        await asyncio.sleep(0.05)
    raise AssertionError("Child did not reach persisted stage " + stage)


@pytest.mark.parametrize(
    "kind,stage", [("ingestion", "validating_sources"), ("analysis", "validating_evidence")]
)
async def test_real_worker_process_kill_and_restart(database, actors, kind, stage):
    ident = await (
        ingestion(database, actors) if kind == "ingestion" else analysis(database, actors)
    )
    proc = await child(database, "execute", stage)
    try:
        await wait_stage(database, ident, stage)
        proc.kill()
        await proc.communicate()
        row = await state(database, ident)
        assert row.status == "processing"
        if kind == "analysis":
            async with database() as db:
                assert await db.scalar(select(func.count()).select_from(Finding)) == 0
        else:
            assert row.revision_id is None
        await expire(database, ident)
        async with database() as db:
            await queue.recover(db)
            row = await db.get(Review, ident)
            row.retry_at = None
            await db.commit()
        resumed = await child(database, "execute")
        _, err = await asyncio.wait_for(resumed.communicate(), 30)
        assert resumed.returncode == 0, err.decode()
        row = await state(database, ident)
        assert row.status == "complete" and row.attempt_count == 2
        if kind == "analysis":
            async with database() as db:
                assert (
                    len(
                        (await repository_analysis.report(db, ident, actors["users"][0].id))[
                            "findings"
                        ]
                    )
                    == 4
                )
    finally:
        if proc.returncode is None:
            proc.kill()
            await proc.communicate()


async def test_real_process_crash_before_processing_and_after_commit(database, actors):
    ident = await analysis(database, actors)
    first = await child(database, "claim")
    _, err = await asyncio.wait_for(first.communicate(), 30)
    assert first.returncode == 0, err.decode()
    assert (await state(database, ident)).status == "processing"
    await expire(database, ident)
    async with database() as db:
        await queue.recover(db)
        row = await db.get(Review, ident)
        row.retry_at = None
        await db.commit()
    proc = await child(database, "execute", after_commit=True)
    try:
        await wait_stage(database, ident, "complete")
        proc.kill()
        await proc.communicate()
        assert not await tick(database, "reconcile")
        async with database() as db:
            assert await db.scalar(select(func.count()).select_from(Finding)) == 4
            assert (await db.get(Review, ident)).status == "complete"
    finally:
        if proc.returncode is None:
            proc.kill()
            await proc.communicate()


async def test_cancellation_during_analysis_prevents_partial_report(database, actors):
    ident = await analysis(database, actors)
    attempt = await claim(database, ident)
    original = attempt.checkpoint

    async def checkpoint(stage=None, percent=None):
        await original(stage, percent)
        if stage == "analyzing_sources":
            async with database() as db:
                await queue.cancel(db, ident, actors["users"][0].id)

    attempt.checkpoint = checkpoint
    await run_attempt(database, attempt)
    row = await state(database, ident)
    assert row.status == "cancelled" and row.analysis_result.get("summary") is None
    async with database() as db:
        assert await db.scalar(select(func.count()).select_from(Finding)) == 0
        assert await db.scalar(select(func.count()).select_from(Evidence)) == 0


async def test_automatic_worker_heartbeat_and_sanitized_failure_logs(database, actors, monkeypatch):
    from structlog.testing import capture_logs

    monkeypatch.setattr(queue.settings, "job_heartbeat_seconds", 0.1)
    ident = await ingestion(database, actors)
    attempt = await claim(database, ident)
    initial = (await state(database, ident)).heartbeat_at

    async def work(factory, owned):
        await asyncio.sleep(0.35)
        assert (await state(factory, ident)).heartbeat_at > initial
        raise RuntimeError("SENSITIVE-JWT-PASSWORD-GITHUB-TOKEN")

    with capture_logs() as logs:
        await run_attempt(database, attempt, work)
    assert "SENSITIVE-JWT-PASSWORD-GITHUB-TOKEN" not in str(logs)
    row = await state(database, ident)
    assert row.status == "failed" and row.error_category == "internal"
    assert "SENSITIVE" not in row.error_message


async def test_old_worker_cannot_publish_after_recovery_during_download(database, actors):
    ident = await ingestion(database, actors)
    attempt = await claim(database, ident, "old")
    responses = GitHubResponses()
    replaced = False

    async def transport(request):
        nonlocal replaced
        result = responses(request)
        if "/git/blobs/" in str(request.url) and not replaced:
            replaced = True
            await expire(database, ident)
            async with database() as db:
                await queue.recover(db)
                row = await db.get(Review, ident)
                row.retry_at = None
                await db.commit()
            newer = await claim(database, ident, "replacement")
            assert newer.token != attempt.token
        return result

    async with database() as db:  # noqa: SIM117 - explicit isolated DB and HTTP lifetimes
        async with GitHubClient(transport=httpx.MockTransport(transport)) as client:
            with pytest.raises(queue.LostAttemptError):
                await github_ingestion.process(db, ident, attempt.user_id, client, attempt=attempt)
    row = await state(database, ident)
    assert row.worker_id == "replacement" and row.status == "processing"
    async with database() as db:
        assert await db.scalar(select(func.count()).select_from(IngestedSource)) == 0


async def test_terminal_cancellation_preserves_committed_historical_results(
    client, database, actors
):
    ident = await analysis(database, actors)
    assert await tick(database, "complete-worker")
    response = await client.post("/jobs/" + ident + "/cancel", headers=bearer(actors))
    assert response.status_code == 409
    status = await client.get("/jobs/" + ident, headers=bearer(actors))
    assert status.status_code == 200 and status.json()["progress_pct"] == 100
    assert "execution_token" not in status.json()
    async with database() as db:
        report = await repository_analysis.report(db, ident, actors["users"][0].id)
        assert len(report["findings"]) == 4
        assert await db.scalar(select(func.count()).select_from(Evidence)) == 4


async def test_downgrade_refuses_unrepresentable_cancelled_history(database, actors):
    from alembic.config import Config
    from sqlalchemy import text

    from alembic import command

    ident = await ingestion(database, actors)
    async with database() as db:
        await queue.cancel(db, ident, actors["users"][0].id)
    with pytest.raises(RuntimeError, match="no cancelled jobs"):
        async with database.kw["bind"].begin() as connection:

            def downgrade(conn):
                cfg = Config("alembic.ini")
                cfg.attributes["connection"] = conn
                command.downgrade(cfg, "005")

            await connection.run_sync(downgrade)
    async with database() as db:
        assert (await db.get(Review, ident)).status == "cancelled"
        assert await db.scalar(text("SELECT version_num FROM alembic_version")) == "012"


async def test_two_real_worker_processes_compete_for_single_attempt(database,actors):
    ident=await ingestion(database,actors)
    first,second=await asyncio.gather(child(database,"claim"),child(database,"claim"))
    for proc in (first,second):
        _,error=await asyncio.wait_for(proc.communicate(),30)
        assert proc.returncode==0,error.decode()
    row=await state(database,ident)
    assert row.status=="processing" and row.attempt_count==1
    assert row.worker_id.startswith("child-") and row.execution_token
