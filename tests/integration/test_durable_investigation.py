"""Real PostgreSQL queue/journal; deterministic planner/provider only."""

import asyncio
from datetime import timedelta

import pytest
from fastapi import HTTPException

from core.models import Investigation, Review
from core.services import durable_investigation as durable
from core.services import job_queue as queue
from core.services.investigation import Decision
from tests.embedding_fixtures import ingested

pytest_plugins = ["tests.integration.test_gemini_production"]

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


async def accepted(database, actors, monkeypatch):
    monkeypatch.setattr(durable.settings, "durable_investigation_enabled", True)
    source = await ingested(
        database,
        actors,
        files={"guard.py": b"def guard():\n    return 1\n", "config.py": b"GATE = 83\n"},
    )
    uid, repo, rev = (
        actors["users"][0].id,
        source.ingestion_result["repository_id"],
        source.revision_id,
    )
    async with database() as db:
        identity = await durable.submit(
            db,
            uid,
            repo,
            rev,
            {
                "objective": "Describe guard.",
                "hosted_processing_consent": "gemini-source-processing-v1",
            },
        )
    return uid, repo, rev, identity


async def claim(database, identity, worker="test-worker"):
    async with database() as db:
        return await queue.claim(db, worker, identity)


async def test_persist_claim_cancel_and_owner_isolation(database, actors, monkeypatch):
    uid, repo, rev, identity = await accepted(database, actors, monkeypatch)
    async with database() as db:
        row = await durable.status(db, uid, repo, rev, identity)
        assert row["status"] == "pending" and row["inference_spent"] == 0
        with pytest.raises(HTTPException):
            await durable.cancel(db, actors["users"][1].id, repo, rev, identity)
        await db.rollback()
        result = await durable.cancel(db, uid, repo, rev, identity)
        assert result["status"] == "cancelled"
    assert await claim(database, identity) is None


async def test_atomic_claim_fencing_and_recovery(database, actors, monkeypatch):
    uid, repo, rev, identity = await accepted(database, actors, monkeypatch)
    attempts = await asyncio.gather(
        claim(database, identity, "one"), claim(database, identity, "two")
    )
    old = next(a for a in attempts if a)
    assert sum(a is not None for a in attempts) == 1
    journal = durable.Journal(old)
    await journal.save("plan/0", {"operation": "search", "query": "guard", "reason": ""})
    async with database() as db:
        job = await db.get(Review, identity)
        job.lease_expires_at = queue.now() - timedelta(seconds=1)
        await db.commit()
        assert await queue.recover(db) == 1
        job = await db.get(Review, identity)
        job.retry_at = queue.now() - timedelta(seconds=1)
        await db.commit()
    newer = await claim(database, identity)
    with pytest.raises(queue.LostAttemptError):
        await journal.save("plan/1", {})
    assert (await durable.Journal(newer).cached("plan/0"))["query"] == "guard"


async def test_reservation_committed_before_provider_and_unknown_recovery(
    database, actors, monkeypatch
):
    uid, repo, rev, identity = await accepted(database, actors, monkeypatch)
    attempt = await claim(database, identity)
    journal = durable.Journal(attempt)
    journal.phase = "plan/0"

    class Interrupted:
        budget = durable.PrepaidBudget()

        async def generate(self, messages, before_request, **kwargs):
            await before_request()
            async with database() as db:
                state = (await db.get(Investigation, identity)).state
                assert len(state["attempts"]) == 1 and state["attempts"][0]["state"] == "reserved"
            raise asyncio.CancelledError()

    provider = durable.DurableProvider(journal, Interrupted())

    async def before():
        pass

    with pytest.raises(asyncio.CancelledError):
        await provider.generate([], before)
    async with database() as db:
        await durable.process(
            db,
            identity,
            uid,
            attempt=attempt,
            make_planner=lambda *a: pytest.fail("ambiguous inference repeated"),
        )
    async with database() as db:
        row = await durable.status(db, uid, repo, rev, identity)
        assert row["status"] == "complete" and row["outcome"] == "unknown_provider_outcome"
        assert row["inference_spent"] == 1 and row["result"] is None


async def test_replay_search_and_decision_without_inference(database, actors, monkeypatch):
    uid, repo, rev, identity = await accepted(database, actors, monkeypatch)
    attempt = await claim(database, identity)
    first = durable.Journal(attempt)
    async with database() as db:
        initial = await first.retrieve(
            db, uid, repo, rev, {"query": "guard", "mode": "keyword", "top_k": 5}
        )
    assert initial.matches
    async with database() as db:
        restored = await durable.Journal(attempt).retrieve(
            db, uid, repo, rev, {"query": "guard", "mode": "keyword", "top_k": 5}
        )
    assert restored == initial
    async with database() as db:
        record = await db.get(Investigation, identity)
        assert "excerpt" not in str(record.state)


async def test_deadline_never_reset(database, actors, monkeypatch):
    uid, repo, rev, identity = await accepted(database, actors, monkeypatch)
    attempt = await claim(database, identity)
    async with database() as db:
        record = await db.get(Investigation, identity)
        record.created_at = queue.now() - timedelta(seconds=10)
        record.deadline_at = queue.now() - timedelta(seconds=1)
        await db.commit()
        await durable.process(
            db,
            identity,
            uid,
            attempt=attempt,
            make_planner=lambda *a: pytest.fail("expired execution planned"),
        )
    async with database() as db:
        assert (await durable.status(db, uid, repo, rev, identity))["outcome"] == "timeout"


async def test_completed_stop_is_terminal(database, actors, monkeypatch):
    uid, repo, rev, identity = await accepted(database, actors, monkeypatch)
    attempt = await claim(database, identity)

    class Planner:
        provider = None

        def set_readiness(self, value):
            pass

        async def __call__(self, *args):
            return Decision(operation="stop")

    async with database() as db:
        await durable.process(db, identity, uid, attempt=attempt, make_planner=lambda *a: Planner())
    assert await claim(database, identity) is None
    async with database() as db:
        result = await durable.status(db, uid, repo, rev, identity)
        assert result["status"] == "complete" and result["outcome"] == "insufficient_evidence"
        with pytest.raises(HTTPException) as exc:
            await durable.cancel(db, uid, repo, rev, identity)
        assert exc.value.status_code == 409


@pytest.mark.parametrize("checkpoint", ["search/0", "search/1", "plan/1"])
async def test_actual_worker_process_crash_and_grounded_recovery(
    database, actors, monkeypatch, gemini, checkpoint
):
    import os
    import sys
    from pathlib import Path

    from sqlalchemy import text

    uid, repo, rev, identity = await accepted(database, actors, monkeypatch)
    async with database() as db:
        schema = await db.scalar(text("SELECT current_schema()"))
    env = dict(
        os.environ,
        APP_ENV="test",
        PYTHONDONTWRITEBYTECODE="1",
        VERIFY_SCHEMA=schema,
        VERIFY_JOB=identity,
        VERIFY_CHECKPOINT=checkpoint,
    )
    child = await asyncio.create_subprocess_exec(
        sys.executable,
        "-B",
        "-m",
        "tests.durable_investigation_child",
        env=env,
        cwd=str(Path(__file__).resolve().parents[2]),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(child.communicate(), 30)
    assert child.returncode == 73, stderr.decode()[-2000:]
    async with database() as db:
        record = await db.get(Investigation, identity)
        assert checkpoint in record.state["events"]
        job = await db.get(Review, identity)
        job.lease_expires_at = queue.now() - timedelta(seconds=1)
        await db.commit()
        assert await queue.recover(db) == 1
        job = await db.get(Review, identity)
        job.retry_at = queue.now() - timedelta(seconds=1)
        await db.commit()
    recovered = await claim(database, identity)
    gemini.budget = durable.PrepaidBudget()

    class Planner:
        def __init__(self, journal):
            self.provider = durable.DurableProvider(journal, gemini)

        def set_readiness(self, value):
            pass

        async def __call__(self, *args):
            return Decision(operation="answer")

    async with database() as db:
        await durable.process(
            db,
            identity,
            uid,
            attempt=recovered,
            make_planner=lambda journal, deadline: durable.DurablePlanner(
                journal, Planner(journal)
            ),
        )
    async with database() as db:
        record = await db.get(Investigation, identity)
        assert record.outcome == "completed", record.final
        assert record.final["final"]["citations"]
        assert record.final["final"]["answer"]
        assert len(record.state["attempts"]) == 2 and len(gemini.calls) == 2
        assert checkpoint in record.state["events"]
        assert (await db.get(Review, identity)).attempt_count == 2
    env["VERIFY_EXPECT_TERMINAL"] = "1"
    restarted = await asyncio.create_subprocess_exec(
        sys.executable,
        "-B",
        "-m",
        "tests.durable_investigation_child",
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, errors = await asyncio.wait_for(restarted.communicate(), 30)
    assert restarted.returncode == 74, errors.decode()[-2000:]


async def test_api_create_get_cancel_and_foreign_denials(client, database, actors, monkeypatch):
    from tests.integration.test_auth_authorization import bearer

    uid, repo, rev, identity = await accepted(database, actors, monkeypatch)
    endpoint = f"/repositories/{repo}/revisions/{rev}/durable-investigations"
    body = {
        "objective": "Describe guard.",
        "generation_provider": "gemini",
        "hosted_processing_consent": "gemini-source-processing-v1",
    }
    assert (await client.post(endpoint, json=body)).status_code == 401
    assert (
        await client.post(endpoint, json=body, headers=bearer(actors["users"][1].id))
    ).status_code == 404
    created = await client.post(endpoint, json=body, headers=bearer(uid))
    assert created.status_code == 202, created.text
    path = endpoint + "/" + created.json()["id"]
    assert (await client.get(path, headers=bearer(uid))).json()["status"] == "pending"
    assert (await client.get(path, headers=bearer(actors["users"][1].id))).status_code == 404
    assert (
        await client.post(path + "/cancel", headers=bearer(actors["users"][1].id))
    ).status_code == 404
    assert (await client.post(path + "/cancel", headers=bearer(uid))).json()[
        "status"
    ] == "cancelled"


async def test_six_call_budget_survives_new_journal_and_callback_once(
    database, actors, monkeypatch
):
    uid, repo, rev, identity = await accepted(database, actors, monkeypatch)
    attempt = await claim(database, identity)

    class Provider:
        budget = durable.PrepaidBudget()

        async def generate(self, messages, before, **kwargs):
            await before()
            return "synthetic response"

    calls = []

    async def before():
        calls.append(1)

    for _ in range(6):
        journal = durable.Journal(attempt)
        journal.phase = "plan/0"
        await durable.DurableProvider(journal, Provider()).generate([], before)
    from rag.generator.local_provider import GenerationError

    with pytest.raises(GenerationError):
        await durable.DurableProvider(durable.Journal(attempt), Provider()).generate([], before)
    async with database() as db:
        assert len((await db.get(Investigation, identity)).state["attempts"]) == 6
    assert len(calls) == 7  # One callback per attempted call; seventh cannot POST.


@pytest.mark.parametrize("fault", ["source", "owner", "inactive"])
async def test_recovered_scope_change_never_reaches_planner(database, actors, monkeypatch, fault):
    from sqlalchemy import update

    from core.models import Profile, SourceChunk, User

    uid, repo, rev, identity = await accepted(database, actors, monkeypatch)
    attempt = await claim(database, identity)
    async with database() as db:
        if fault == "source":
            await db.execute(
                update(SourceChunk)
                .where(SourceChunk.profile_id == actors["profiles"][0].id)
                .values(text="corrupt")
            )
        elif fault == "owner":
            await db.execute(
                update(Profile)
                .where(Profile.id == actors["profiles"][0].id)
                .values(user_id=actors["users"][1].id)
            )
        else:
            await db.execute(update(User).where(User.id == uid).values(is_active=False))
        await db.commit()
        await durable.process(
            db,
            identity,
            uid,
            attempt=attempt,
            make_planner=lambda *a: pytest.fail("revoked scope planned"),
        )
    async with database() as db:
        record = await db.get(Investigation, identity)
        assert record.outcome == "scope_changed" and record.final is None


async def test_inflight_cancellation_committed_and_no_publication(database, actors, monkeypatch):
    uid, repo, rev, identity = await accepted(database, actors, monkeypatch)
    attempt = await claim(database, identity)
    entered, release = asyncio.Event(), asyncio.Event()

    class Provider:
        budget = durable.PrepaidBudget()

        async def generate(self, messages, before, **kwargs):
            await before()
            entered.set()
            await release.wait()
            return "synthetic response"

    async def before():
        pass

    task = asyncio.create_task(
        durable.DurableProvider(durable.Journal(attempt), Provider()).generate([], before)
    )
    await asyncio.wait_for(entered.wait(), 5)
    async with database() as db:
        await durable.cancel(db, uid, repo, rev, identity)
    release.set()
    with pytest.raises(queue.CancellationRequestedError) as exc:
        await task
    await queue.fail(database, attempt, exc.value)
    assert await claim(database, identity) is None
    async with database() as db:
        record, job = await db.get(Investigation, identity), await db.get(Review, identity)
        assert job.status == "cancelled" and record.final is None
        assert len(record.state["attempts"]) == 1


@pytest.mark.parametrize("fault", ["partial", "id", "unsupported"])
async def test_durable_final_validation_cannot_be_bypassed(
    database, actors, monkeypatch, gemini, fault
):
    uid, repo, rev, identity = await accepted(database, actors, monkeypatch)
    attempt = await claim(database, identity)
    gemini.fault, gemini.budget = fault, durable.PrepaidBudget()

    class Planner:
        def __init__(self, journal):
            self.provider = durable.DurableProvider(journal, gemini)

        def set_readiness(self, value):
            pass

        async def __call__(self, *args):
            return Decision(operation="answer")

    async with database() as db:
        await durable.process(
            db,
            identity,
            uid,
            attempt=attempt,
            make_planner=lambda journal, deadline: durable.DurablePlanner(
                journal, Planner(journal)
            ),
        )
    async with database() as db:
        result = (await db.get(Investigation, identity)).final
        assert result["status"] == "validation_failure"
        assert result["final"] is None


async def test_step_budget_replayed_without_refill(database, actors, monkeypatch):
    uid, repo, rev, identity = await accepted(database, actors, monkeypatch)
    attempt = await claim(database, identity)
    async with database() as db:
        record = await db.get(Investigation, identity)
        record.request = {**record.request, "max_steps": 1}
        await db.commit()
    journal = durable.Journal(attempt)
    await journal.save("plan/0", {"operation": "search", "query": "guard", "reason": ""})
    async with database() as db:
        await journal.retrieve(
            db, uid, repo, rev, {"query": "guard", "mode": "keyword", "top_k": 5}
        )

    class Planner:
        provider = None

        def set_readiness(self, value):
            pass

        async def __call__(self, *args):
            pytest.fail("Restart refilled step allowance")

    async with database() as db:
        await durable.process(
            db,
            identity,
            uid,
            attempt=attempt,
            make_planner=lambda j, d: durable.DurablePlanner(j, Planner()),
        )
    async with database() as db:
        assert (await db.get(Investigation, identity)).outcome == "step_limit"


async def test_checker_cancellation_wins_finalization_race(database, actors, monkeypatch, gemini):
    from workers.run import run_attempt

    uid, repo, rev, identity = await accepted(database, actors, monkeypatch)
    attempt = await claim(database, identity)
    gemini.pause, gemini.budget = "checker", durable.PrepaidBudget()

    class Planner:
        def __init__(self, journal):
            self.provider = durable.DurableProvider(journal, gemini)

        def set_readiness(self, value):
            pass

        async def __call__(self, *args):
            return Decision(operation="answer")

    async def execute(factory, current):
        async with factory() as db:
            await durable.process(
                db,
                identity,
                uid,
                attempt=current,
                make_planner=lambda j, d: durable.DurablePlanner(j, Planner(j)),
            )

    task = asyncio.create_task(run_attempt(database, attempt, execute))
    await asyncio.wait_for(gemini.entered.wait(), 15)
    async with database() as db:
        await durable.cancel(db, uid, repo, rev, identity)
    gemini.release.set()
    await asyncio.wait_for(task, 15)
    async with database() as db:
        record, job = await db.get(Investigation, identity), await db.get(Review, identity)
        assert job.status == "cancelled" and record.final is None
        assert len(record.state["attempts"]) == 2


async def test_api_wrong_revision_inactive_and_disabled(client, database, actors, monkeypatch):
    from uuid import uuid4

    from sqlalchemy import update

    from core.models import User
    from core.security import create_access_token

    uid, repo, rev, identity = await accepted(database, actors, monkeypatch)
    headers = {"Authorization": "Bearer " + create_access_token({"sub": uid})}
    base = f"/repositories/{repo}/revisions/{rev}/durable-investigations"
    wrong = base.replace(rev, str(uuid4())) + "/" + identity
    assert (await client.get(wrong, headers=headers)).status_code == 404
    monkeypatch.setattr(durable.settings, "durable_investigation_enabled", False)
    body = {
        "objective": "Describe guard.",
        "generation_provider": "gemini",
        "hosted_processing_consent": "gemini-source-processing-v1",
    }
    assert (await client.post(base, json=body, headers=headers)).status_code == 409
    async with database() as db:
        await db.execute(update(User).where(User.id == uid).values(is_active=False))
        await db.commit()
    assert (await client.get(base + "/" + identity, headers=headers)).status_code == 401
    assert (
        await client.post(base + "/" + identity + "/cancel", headers=headers)
    ).status_code == 401


async def test_native_gemini_adapter_reserves_post_once(database, actors, monkeypatch):
    import httpx

    from rag.generator.gemini_provider import Gemini
    from tests.unit.test_gemini_experiment import response

    uid, repo, rev, identity = await accepted(database, actors, monkeypatch)
    attempt = await claim(database, identity)
    posts, callbacks = [], []

    async def handler(request):
        async with database() as db:
            reservations = (await db.get(Investigation, identity)).state["attempts"]
            assert len(reservations) == 1 and reservations[0]["state"] == "reserved"
        posts.append(request.method)
        return httpx.Response(200, json=response())

    async def before():
        callbacks.append(1)

    native = Gemini("TEST_ONLY", durable.PrepaidBudget(), httpx.MockTransport(handler))
    provider = durable.DurableProvider(durable.Journal(attempt), native)
    result = await provider.generate(
        [{"role": "system", "content": "synthetic"}, {"role": "user", "content": "synthetic"}],
        before,
        response_schema={"type": "object"},
    )
    assert result.model == "gemini-3.5-flash-lite"
    assert posts == ["POST"] and callbacks == [1]
    async with database() as db:
        state = (await db.get(Investigation, identity)).state
        assert state["attempts"][0]["state"] == "response_received"
        assert "TEST_ONLY" not in str(state)


async def test_actual_process_crash_cannot_refill_inference_budget(database, actors, monkeypatch):
    import os
    import sys

    from sqlalchemy import text

    uid, repo, rev, identity = await accepted(database, actors, monkeypatch)
    async with database() as db:
        schema = await db.scalar(text("SELECT current_schema()"))
    env = dict(
        os.environ,
        APP_ENV="test",
        PYTHONDONTWRITEBYTECODE="1",
        VERIFY_SCHEMA=schema,
        VERIFY_JOB=identity,
        VERIFY_CHECKPOINT="inference",
    )
    child = await asyncio.create_subprocess_exec(
        sys.executable,
        "-B",
        "-m",
        "tests.durable_investigation_child",
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await asyncio.wait_for(child.communicate(), 30)
    assert child.returncode == 73, stderr.decode()[-2000:]
    async with database() as db:
        record = await db.get(Investigation, identity)
        assert len(record.state["attempts"]) == 1
        job = await db.get(Review, identity)
        job.lease_expires_at = queue.now() - timedelta(seconds=1)
        await db.commit()
        assert await queue.recover(db) == 1
        job = await db.get(Review, identity)
        job.retry_at = queue.now() - timedelta(seconds=1)
        await db.commit()
    attempt = await claim(database, identity)
    async with database() as db:
        await durable.process(
            db,
            identity,
            uid,
            attempt=attempt,
            make_planner=lambda *args: pytest.fail("Ambiguous POST replayed"),
        )
    async with database() as db:
        record = await db.get(Investigation, identity)
        assert record.outcome == "unknown_provider_outcome" and record.final is None
        assert len(record.state["attempts"]) == 1


async def test_initial_scope_time_does_not_extend_absolute_deadline(database, actors, monkeypatch):
    uid, repo, rev, identity = await accepted(database, actors, monkeypatch)
    attempt = await claim(database, identity)
    async with database() as db:
        deadline = queue.aware((await db.get(Investigation, identity)).deadline_at)
        job = await db.get(Review, identity)
        job.lease_expires_at = deadline + timedelta(seconds=30)
        await db.commit()
    original = durable.retrieval._snapshot

    async def slow_scope(*args):
        result = await original(*args)
        monkeypatch.setattr(queue, "now", lambda: deadline + timedelta(seconds=1))
        return result

    monkeypatch.setattr(durable.retrieval, "_snapshot", slow_scope)
    async with database() as db:
        await durable.process(
            db,
            identity,
            uid,
            attempt=attempt,
            make_planner=lambda *args: pytest.fail("Initial scope reset absolute deadline"),
        )
    async with database() as db:
        record = await db.get(Investigation, identity)
        assert record.outcome == "timeout" and record.final is None
