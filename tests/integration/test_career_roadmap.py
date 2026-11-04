"""Real PostgreSQL/API/canonical J2 retrieval. All external model/GitHub I/O doubled."""

import asyncio
import json
from datetime import timedelta
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy import select, update
from sqlalchemy.exc import DBAPIError

from api.schemas.career_matching import CONSENT
from core.config import settings
from core.models import CareerAIBudget, CareerAssessment, CareerRoadmap, Profile, User
from core.services import career_roadmap as service
from core.services import career_roadmap_policy as policy
from tests.integration import test_career_matching as j2
from tests.integration.test_auth_authorization import bearer

model = j2.model
portfolio = j2.portfolio
create = j2.create
match = j2.match

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


@pytest.fixture
def road_model(model):
    original = model.output
    model.road_failure = None
    model.check_invalid = False
    model.bad_project = False

    def output(messages):
        system = messages[0]["content"]
        body = json.loads(messages[1]["content"])
        if system == policy.GENERATE:
            if model.road_failure == "malformed":
                return {"unrecognized": "output"}
            return {
                "items": [
                    dict(
                        requirement_id=r["id"],
                        action_type="documenting" if r["citations"] else "learning",
                        action="Write proposed "
                        + r["quote"]
                        + " examples with executable verification.",
                        gap="This requirement is not fully demonstrated within inspected evidence.",
                        rationale="The job explicitly includes " + r["quote"],
                        completion_proof=(
                            "Publish a reproducible example and its verification output."
                        ),
                        project_citation=4
                        if model.bad_project
                        else (0 if r["citations"] else None),
                    )
                    for r in body["requirements"]
                ]
            }
        if system == policy.CHECK:
            return {
                "complete": not model.check_invalid,
                "items": [
                    dict(
                        requirement_id=r["id"],
                        grounded=True,
                        individualized=True,
                        gap_accurate=True,
                        future_only=True,
                        observable_proof=True,
                        project_supported=True,
                        action_type_appropriate=True,
                    )
                    for r in body["requirements"]
                ],
            }
        return original(messages)

    model.output = output
    return model


@pytest_asyncio.fixture
async def assessed(client, actors, portfolio, road_model):
    response, _ = await create(client, actors, portfolio)
    assert response.status_code == 201, response.text
    identity = response.json()["id"]
    for _ in range(3):
        response = await match(client, actors, identity)
        assert response.status_code == 200, response.text
    assert response.json()["status"] == "complete"
    return response.json()


async def generate(client, actors, assessed, **kwargs):
    return await client.post(
        f"/career-assessments/{assessed['id']}/roadmap",
        headers=bearer(actors["users"][0].id),
        json={"hosted_processing_consent": CONSENT, **kwargs},
    )


async def test_real_assessment_to_roadmap_persistence_and_idempotency(
    client, database, actors, assessed, road_model
):
    response = await generate(client, actors, assessed)
    assert response.status_code == 201, response.text
    data = response.json()
    assert data["status"] == "complete", data
    assert data["inference_spent"] == 8 and data["roadmap_inference_spent"] == 2
    recommendations = data["result"]["recommendations"]
    assert len(recommendations) == 2
    assert recommendations[0]["priority"] == "high" and recommendations[0]["project"] is None
    documented = next(
        x for x in recommendations if x["observation"]["classification"] == "documented/claimed"
    )
    assert documented["project"]["full_name"] == "example/beta"
    assert documented["observation"]["citations"] == assessed["matches"][1]["citations"]
    assert documented["effort"] == "unknown"
    count = len(road_model.posts)
    for _ in range(2):
        replay = await generate(client, actors, assessed)
        assert replay.json() == data
    for url in [f"/career-roadmaps/{data['id']}", f"/career-assessments/{assessed['id']}/roadmap"]:
        saved = await client.get(url, headers=bearer(actors["users"][0].id))
        assert saved.json() == data and saved.headers["cache-control"] == "no-store"
    assert len(road_model.posts) == count
    async with database() as db:
        row = await db.get(CareerRoadmap, data["id"])
        assert row.version == 1
        budget = await db.get(CareerAIBudget, 1)
        assert budget.spent == 8
        parent = await db.get(CareerAssessment, assessed["id"])
        assert parent.status == "complete"


@pytest.mark.parametrize("mode", ["incomplete", "bad_project", "malformed", "provider"])
async def test_rejections_publish_no_result(client, database, actors, assessed, road_model, mode):
    if mode == "incomplete":
        road_model.check_invalid = True
    if mode == "bad_project":
        road_model.bad_project = True
    if mode == "malformed":
        road_model.road_failure = "malformed"
    if mode == "provider":
        road_model.failure = "network"
    response = await generate(client, actors, assessed)
    assert response.status_code in {502, 503}, response.text
    result = response.json()
    assert result["status"] == "failed" and result["result"] is None
    assert result["inference_spent"] > assessed["inference_spent"]
    posts = len(road_model.posts)
    assert (await generate(client, actors, assessed)).json()["status"] == "failed"
    assert len(road_model.posts) == posts


async def test_auth_consent_and_direct_owner_checks(client, database, actors, assessed):
    url = f"/career-assessments/{assessed['id']}/roadmap"
    assert (await client.post(url, json={"hosted_processing_consent": CONSENT})).status_code == 401
    assert (
        await client.post(
            url, headers=bearer(actors["users"][1].id), json={"hosted_processing_consent": CONSENT}
        )
    ).status_code == 404
    assert (
        await client.post(url, headers=bearer(actors["users"][0].id), json={})
    ).status_code == 422
    assert (await generate(client, actors, assessed, reset_budget=True)).status_code == 422
    async with database() as db:
        with pytest.raises(Exception) as denied:
            await service.create(
                db, actors["users"][1].id, assessed["id"], {"hosted_processing_consent": CONSENT}
            )
        assert denied.value.status_code == 404
        await db.execute(
            update(User).where(User.id == actors["users"][0].id).values(is_active=False)
        )
        await db.commit()
    assert (await generate(client, actors, assessed)).status_code == 401


@pytest.mark.parametrize("kind", ["assessment", "owner", "aggregate", "stopped", "rate"])
async def test_shared_budget_is_never_refilled(
    client, database, actors, assessed, road_model, kind
):
    async with database() as db:
        if kind == "assessment":
            parent = await db.get(CareerAssessment, assessed["id"])
            # 6 recorded + 5 = 11 of 12: fewer than two left for the roadmap (12-call policy).
            parent.calls = parent.calls + [{"phase": "spent", "state": "reserved"}] * 5
        else:
            b = await db.get(CareerAIBudget, 1)
            if kind == "owner":
                b.owners = {actors["users"][0].id: 40}
            if kind == "aggregate":
                b.spent = 100
            if kind == "stopped":
                b.stopped = True
            if kind == "rate":
                b.recent = [service.assessment.time.time()] * 10
        await db.commit()
    count = len(road_model.posts)
    response = await generate(client, actors, assessed)
    assert response.status_code in {409, 429, 503}, response.text
    assert len(road_model.posts) == count
    async with database() as db:
        b = await db.get(CareerAIBudget, 1)
        if kind == "aggregate":
            assert b.spent == 100
        if kind == "owner":
            assert b.owners[actors["users"][0].id] == 40


async def test_cancellation_fences_inflight_result(client, database, actors, assessed, road_model):
    road_model.entered.clear()
    road_model.release = asyncio.Event()
    pending = asyncio.create_task(generate(client, actors, assessed))
    await asyncio.wait_for(road_model.entered.wait(), 15)
    async with database() as db:
        row = await db.scalar(select(CareerRoadmap))
        identity = row.id
    cancelled = await client.post(
        f"/career-roadmaps/{identity}/cancel", headers=bearer(actors["users"][0].id)
    )
    assert cancelled.json()["status"] == "cancelled"
    road_model.release.set()
    response = await pending
    assert response.status_code == 409, response.text
    saved = await client.get(f"/career-roadmaps/{identity}", headers=bearer(actors["users"][0].id))
    assert saved.json()["status"] == "cancelled" and saved.json()["result"] is None
    assert saved.json()["roadmap_inference_spent"] == 1


async def test_ownership_transfer_during_inference(client, database, actors, assessed, road_model):
    async def transfer():
        async with database() as db:
            await db.execute(
                update(Profile)
                .where(Profile.id == actors["profiles"][0].id)
                .values(user_id=actors["users"][1].id)
            )
            await db.commit()

    road_model.hook = transfer
    response = await generate(client, actors, assessed)
    assert response.status_code == 404, response.text
    async with database() as db:
        row = await db.scalar(select(CareerRoadmap))
        assert row.status == "failed" and row.result is None
        identity = row.id  # rollback expires ORM state; preserve the identifier, not a lazy load
        for uid in [u.id for u in actors["users"]]:
            with pytest.raises(Exception) as denied:
                await service.detail(db, uid, identity)
            assert denied.value.status_code == 404
            await db.rollback()


async def test_expired_unknown_never_resends_and_terminal_immutable(
    client, database, actors, assessed, road_model
):
    async with database() as db:
        p = await db.get(CareerAssessment, assessed["id"])
        row = CareerRoadmap(
            assessment_id=p.id,
            source_signature=service.signature(p),
            operation_token=str(uuid4()),
            lease_until=service.assessment.now() - timedelta(seconds=1),
        )
        db.add(row)
        await db.commit()
        identity = row.id
    posts = len(road_model.posts)
    response = await generate(client, actors, assessed)
    assert (
        response.json()["status"] == "failed"
        and response.json()["error_category"] == "interrupted_unknown"
    )
    assert len(road_model.posts) == posts
    async with database() as db:
        with pytest.raises(DBAPIError):
            await db.execute(
                update(CareerRoadmap)
                .where(CareerRoadmap.id == identity)
                .values(status="complete", result={})
            )
            await db.commit()
        await db.rollback()


async def test_citation_and_snapshot_tamper_rejected(
    client, database, actors, assessed, road_model
):
    async with database() as db:
        p = await db.get(CareerAssessment, assessed["id"])
        matches = json.loads(json.dumps(p.matches))
        matches[1]["citations"][0]["quote"] = "invented nonexistent quote"
        p.matches = matches
        await db.commit()
    posts = len(road_model.posts)
    response = await generate(client, actors, assessed)
    assert response.status_code == 409 and len(road_model.posts) == posts


async def test_incomplete_assessment_and_disabled_provider(
    client, database, actors, portfolio, road_model, monkeypatch
):
    response, _ = await create(client, actors, portfolio)
    row = response.json()
    assert (await generate(client, actors, row)).status_code == 409
    for _ in range(3):
        response = await match(client, actors, row["id"])
    monkeypatch.setattr(settings, "career_j2_enabled", False)
    assert (await generate(client, actors, response.json())).status_code == 409


@pytest.mark.parametrize("partial", [False, True])
async def test_demonstrated_no_action_and_partial_project_proposal(
    client, database, actors, portfolio, road_model, monkeypatch, partial
):
    monkeypatch.setattr(j2, "REQUIREMENTS", j2.REQUIREMENTS[:1])
    road_model.partial = partial
    created, _ = await create(client, actors, portfolio)
    assessed = (await match(client, actors, created.json()["id"])).json()
    assert assessed["status"] == "complete"
    before = len(road_model.posts)
    response = await generate(client, actors, assessed)
    assert response.status_code == 201, response.text
    value = response.json()
    assert value["status"] == "complete", value
    if partial:
        assert (
            value["result"]["recommendations"][0]["observation"]["classification"]
            == "partially demonstrated"
        )
        assert value["result"]["recommendations"][0]["project"]
        assert len(road_model.posts) == before + 2
    else:
        assert value["result"]["recommendations"] == []
        assert value["result"]["demonstrated_requirement_ids"] == [
            assessed["requirements"][0]["id"]
        ]
        assert len(road_model.posts) == before and value["roadmap_inference_spent"] == 0


async def test_foreign_read_cancel_inactive_and_dirty_direct_service(
    client, database, actors, assessed, road_model
):
    response = await generate(client, actors, assessed)
    assert response.json()["status"] == "complete", response.text
    identity = response.json()["id"]
    for method, path in [
        ("get", f"/career-roadmaps/{identity}"),
        ("post", f"/career-roadmaps/{identity}/cancel"),
        ("get", f"/career-assessments/{assessed['id']}/roadmap"),
    ]:
        assert (
            await getattr(client, method)(path, headers=bearer(actors["users"][1].id))
        ).status_code == 404
        assert (await getattr(client, method)(path)).status_code == 401
    assert (
        await client.post(
            f"/career-roadmaps/{identity}/cancel", headers=bearer(actors["users"][0].id)
        )
    ).status_code == 409
    async with database() as db:
        profile = await db.get(Profile, actors["profiles"][0].id)
        profile.github_username = "pending-change"
        with pytest.raises(RuntimeError, match="clean session"):
            await service.create(
                db, actors["users"][0].id, assessed["id"], {"hosted_processing_consent": CONSENT}
            )
        assert profile in db.dirty
        await db.rollback()
        await db.execute(
            update(User).where(User.id == actors["users"][0].id).values(is_active=False)
        )
        await db.commit()
        with pytest.raises(Exception) as denied:
            await service.detail(db, actors["users"][0].id, identity)
        assert denied.value.status_code == 401
    assert (
        await client.get(f"/career-roadmaps/{identity}", headers=bearer(actors["users"][0].id))
    ).status_code == 401


async def test_all_canonical_dimensions_and_later_snapshot_change(
    client, database, actors, assessed, road_model
):
    original = assessed["matches"]
    changes = [
        {"quote": "fabricated quote"},
        {"path": "another.py"},
        {"chunk_id": str(uuid4())},
        {"repository_id": str(uuid4())},
        {"revision_id": str(uuid4())},
        {"commit_sha": "f" * 40},
        {"line_start": 1000, "line_end": 1001},
    ]
    count = len(road_model.posts)
    for change in changes:
        altered = json.loads(json.dumps(original))
        altered[1]["citations"][0].update(change)
        async with database() as db:
            await db.execute(
                update(CareerAssessment)
                .where(CareerAssessment.id == assessed["id"])
                .values(matches=altered)
            )
            await db.commit()
        assert (await generate(client, actors, assessed)).status_code == 409
        assert len(road_model.posts) == count
    async with database() as db:
        await db.execute(
            update(CareerAssessment)
            .where(CareerAssessment.id == assessed["id"])
            .values(matches=original)
        )
        await db.commit()
    result = await generate(client, actors, assessed)
    assert result.json()["status"] == "complete", result.text
    async with database() as db:
        altered = json.loads(json.dumps(original))
        altered[1]["uncertainty"] = "Changed assessment meaning."
        await db.execute(
            update(CareerAssessment)
            .where(CareerAssessment.id == assessed["id"])
            .values(matches=altered)
        )
        await db.commit()
    assert (
        await client.get(
            f"/career-roadmaps/{result.json()['id']}", headers=bearer(actors["users"][0].id)
        )
    ).status_code == 409


async def test_concurrent_duplicate_submission_has_two_total_attempts(
    client, database, actors, assessed, road_model
):
    road_model.entered.clear()
    road_model.release = asyncio.Event()
    pending = asyncio.create_task(generate(client, actors, assessed))
    await asyncio.wait_for(road_model.entered.wait(), 15)
    replay = await generate(client, actors, assessed)
    assert replay.json()["status"] == "generating" and replay.json()["result"] is None
    road_model.release.set()
    finished = await pending
    assert finished.json()["status"] == "complete", finished.text
    assert finished.json()["roadmap_inference_spent"] == 2


async def test_used_roadmap_downgrade_and_provenance_guard(client, database, actors, assessed):
    from alembic.config import Config

    from alembic import command

    response = await generate(client, actors, assessed)
    assert response.json()["status"] == "complete", response.text
    async with database() as db:
        with pytest.raises(DBAPIError, match="provenance is immutable"):
            await db.execute(
                update(CareerRoadmap)
                .where(CareerRoadmap.id == response.json()["id"])
                .values(source_signature="0" * 64)
            )
            await db.commit()
        await db.rollback()
        connection = await db.connection()

        def downgrade(conn):
            config = Config("alembic.ini")
            config.attributes["connection"] = conn
            with pytest.raises(RuntimeError, match="Roadmap history exists"):
                command.downgrade(config, "010")

        await connection.run_sync(downgrade)
        await db.rollback()
        assert (await db.get(CareerRoadmap, response.json()["id"])).status == "complete"


@pytest.mark.parametrize("kind", ["assessment-eight", "owner-thirty-nine", "aggregate-ninety-nine"])
async def test_last_available_attempts_are_preserved(
    client, database, actors, assessed, road_model, kind
):
    async with database() as db:
        parent = await db.get(CareerAssessment, assessed["id"])
        budget = await db.get(CareerAIBudget, 1)
        if kind == "assessment-eight":
            parent.calls = parent.calls + [{"phase": "prior-reserved", "state": "consumed"}] * 2
            budget.spent = 8
            budget.owners = {actors["users"][0].id: 8}
        elif kind == "owner-thirty-nine":
            budget.owners = {actors["users"][0].id: 39}
        else:
            budget.spent = 99
        await db.commit()
    response = await generate(client, actors, assessed)
    value = response.json()
    if kind == "assessment-eight":
        assert response.status_code == 201 and value["status"] == "complete", response.text
        assert value["inference_spent"] == 10 and value["roadmap_inference_spent"] == 2
    else:
        # One shared attempt cannot fund generate + check: refused before any roadmap row or
        # POST (2026-11-04), so the last attempt is preserved unspent and nothing is left failed.
        assert response.status_code == 409 and "fewer than two" in response.text, response.text
        missing = await client.get(
            f"/career-assessments/{assessed['id']}/roadmap", headers=bearer(actors["users"][0].id)
        )
        assert missing.status_code == 404
    async with database() as db:
        budget = await db.get(CareerAIBudget, 1)
        if kind == "owner-thirty-nine":
            assert budget.owners[actors["users"][0].id] == 39
        if kind == "aggregate-ninety-nine":
            assert budget.spent == 99
