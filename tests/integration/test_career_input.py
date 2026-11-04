"""Real isolated PostgreSQL/JWT/ingestion workers; only GitHub is doubled."""

import httpx
import pytest
from fastapi import HTTPException
from sqlalchemy import func, select, text, update

from core.models import CareerInput, CareerSelection, Profile, Review, User
from core.services import career_input as service
from core.services import github_ingestion
from ingestion.github import GitHubClient
from ingestion.github_profile import Discovery
from tests.github_responses import GitHubResponses
from tests.integration.test_auth_authorization import bearer
from tests.unit.test_career_input import repository
from workers.run import tick

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]
JD = "We need a Python engineer to build reliable APIs and maintain automated tests."


@pytest.fixture
def discovery(monkeypatch):
    def respond(request):
        if request.url.path == "/users/example":
            return httpx.Response(200, json={"login": "example", "type": "User"})
        return httpx.Response(
            200, json=[repository(i) for i in range(1, 7)] + [repository(7, fork=True)]
        )

    monkeypatch.setattr(
        service, "make_discovery", lambda: Discovery(transport=httpx.MockTransport(respond))
    )


async def create(client, actors):
    result = await client.post(
        "/career-inputs",
        headers=bearer(actors["users"][0].id),
        json={
            "profile_id": actors["profiles"][0].id,
            "github_profile_url": "https://github.com/Example",
            "job_description": JD,
        },
    )
    assert result.status_code == 201, result.text
    return result.json()


async def test_snapshot_selection_auth_and_rollback(client, database, actors, discovery):
    item = await create(client, actors)
    path = "/career-inputs/" + item["id"]
    owner, foreign = [bearer(u.id) for u in actors["users"]]
    assert item["job_description"] == JD and item["future_hosted_consent"] == "not_granted"
    assert (await client.get(path)).status_code == 401
    assert (await client.get(path, headers=foreign)).status_code == 404
    for names in [
        ["example/project7"],
        ["foreign/repo"],
        ["example/project1"] * 2,
        [f"example/project{i}" for i in range(1, 7)],
    ]:
        assert (
            await client.put(path + "/selection", headers=owner, json={"repositories": names})
        ).status_code == 422
    assert (await client.get(path, headers=owner)).json()["selection"] == []
    selected = await client.put(
        path + "/selection",
        headers=owner,
        json={"repositories": ["example/project2", "example/project1"]},
    )
    assert selected.status_code == 200 and selected.json()["status"] == "selection_saved"
    listing = await client.get("/career-inputs", headers=owner)
    assert "job_description" not in listing.text and JD not in listing.text
    assert (await client.get("/career-inputs", headers=foreign)).json()["items"] == []
    async with database() as db:
        row = await db.get(CareerInput, item["id"])
        row.job_description = JD + " changed"
        with pytest.raises(ValueError):
            await db.commit()
        await db.rollback()
        assert (await db.get(CareerInput, item["id"])).job_description == JD
    async with database() as db:
        await db.execute(
            update(Profile)
            .where(Profile.id == actors["profiles"][0].id)
            .values(user_id=actors["users"][1].id)
        )
        await db.commit()
    assert (await client.get(path, headers=owner)).status_code == 404
    assert (await client.get(path, headers=foreign)).status_code == 404
    async with database() as db:
        with pytest.raises(HTTPException):
            await service.detail(db, actors["users"][1].id, item["id"])


async def test_sequential_existing_worker_pins_and_gets_do_not_schedule(
    client, database, actors, discovery, monkeypatch
):
    item = await create(client, actors)
    path = "/career-inputs/" + item["id"]
    headers = bearer(actors["users"][0].id)
    fixtures = {f"example/project{i}": GitHubResponses(name=f"example/project{i}") for i in [1, 2]}

    def respond(request):
        name = "/".join(request.url.path.split("/")[2:4])
        return fixtures[name](request)

    monkeypatch.setattr(
        github_ingestion,
        "make_github_client",
        lambda: GitHubClient(transport=httpx.MockTransport(respond)),
    )
    assert (
        await client.put(
            path + "/selection", headers=headers, json={"repositories": list(fixtures)}
        )
    ).status_code == 200
    first = (await client.post(path + "/prepare", headers=headers)).json()
    job_id = first["selection"][0]["job_id"]
    assert job_id and first["selection"][1]["job_id"] is None
    for _ in range(2):
        again = await client.post(path + "/prepare", headers=headers)
        assert again.status_code == 202 and again.json()["selection"][0]["job_id"] == job_id
        assert (await client.get(path, headers=headers)).json()["status"] == "preparing"
    assert (
        await client.put(
            path + "/selection", headers=headers, json={"repositories": ["example/project2"]}
        )
    ).status_code == 409
    await tick(database, "career-test-worker")
    # Terminal ingestion alone doesn't mutate/finalize the J1 snapshot on a read.
    before = (await client.get(path, headers=headers)).json()
    assert (
        before["selection"][0]["status"] == "awaiting_pin"
        and before["selection"][1]["job_id"] is None
    )
    second = await client.post(path + "/prepare", headers=headers)
    assert second.status_code == 202, second.text
    assert second.json()["selection"][0]["commit_sha"] == fixtures["example/project1"].sha
    await tick(database, "career-test-worker")
    final = await client.post(path + "/prepare", headers=headers)
    assert final.status_code == 202, final.text
    assert final.json()["status"] == "ready"
    manifest = final.json()["selection"]
    assert all(x["revision_id"] and x["repository_id"] for x in manifest)
    assert (await client.post(path + "/prepare", headers=headers)).json()["selection"] == manifest
    async with database() as db:
        assert (
            await db.scalar(
                select(func.count())
                .select_from(Review)
                .where(Review.job_type == "github_ingestion")
            )
            == 2
        )
        assert await db.scalar(text("SELECT version_num FROM alembic_version")) == "012"


async def test_failed_partial_and_inactive(client, database, actors, discovery):
    item = await create(client, actors)
    path = "/career-inputs/" + item["id"]
    headers = bearer(actors["users"][0].id)
    assert (await client.post(path + "/prepare", headers=headers)).status_code == 409
    await client.put(
        path + "/selection", headers=headers, json={"repositories": ["example/project1"]}
    )
    prepared = await client.post(path + "/prepare", headers=headers)
    job = prepared.json()["selection"][0]["job_id"]
    async with database() as db:
        row = await db.get(Review, job)
        row.status = "failed"
        row.error_category = "not_public"
        await db.commit()
    assert (await client.post(path + "/prepare", headers=headers)).json()["status"] == "failed"
    async with database() as db:
        await db.execute(
            update(User).where(User.id == actors["users"][0].id).values(is_active=False)
        )
        await db.commit()
    assert (await client.get(path, headers=headers)).status_code == 401
    assert (await client.post(path + "/prepare", headers=headers)).status_code == 401


async def test_sensitive_validation_and_foreign_creation(client, actors, discovery):
    headers = bearer(actors["users"][0].id)
    data = {
        "profile_id": actors["profiles"][1].id,
        "github_profile_url": "https://github.com/example",
        "job_description": JD,
    }
    assert (await client.post("/career-inputs", headers=headers, json=data)).status_code == 404
    data["profile_id"] = actors["profiles"][0].id
    for value in [" ", "SENSITIVE_JD_" * 2000]:
        data["job_description"] = value
        r = await client.post("/career-inputs", headers=headers, json=data)
        assert r.status_code in {413, 422}
        assert "SENSITIVE_JD_" not in r.text
        assert "input" not in str(r.json().get("detail", [])) if r.status_code == 422 else True


async def test_concurrent_prepare_reuses_one_job(client, database, actors, discovery):
    import asyncio

    item = await create(client, actors)
    path = "/career-inputs/" + item["id"]
    headers = bearer(actors["users"][0].id)
    await client.put(
        path + "/selection", headers=headers, json={"repositories": ["example/project1"]}
    )
    results = await asyncio.gather(
        *[client.post(path + "/prepare", headers=headers) for _ in range(3)]
    )
    assert all(r.status_code == 202 for r in results), [r.text for r in results]
    assert len({r.json()["selection"][0]["job_id"] for r in results}) == 1
    async with database() as db:
        assert (
            await db.scalar(
                select(func.count())
                .select_from(Review)
                .where(Review.job_type == "github_ingestion")
            )
            == 1
        )


async def test_partial_and_database_snapshot_guards(
    client, database, actors, discovery, monkeypatch
):
    from alembic.config import Config
    from sqlalchemy.exc import DBAPIError

    from alembic import command

    item = await create(client, actors)
    path = "/career-inputs/" + item["id"]
    headers = bearer(actors["users"][0].id)
    await client.put(
        path + "/selection",
        headers=headers,
        json={"repositories": ["example/project1", "example/project2"]},
    )
    fixture = GitHubResponses(name="example/project1")
    monkeypatch.setattr(
        github_ingestion,
        "make_github_client",
        lambda: GitHubClient(transport=httpx.MockTransport(fixture)),
    )
    await client.post(path + "/prepare", headers=headers)
    await tick(database, "career-partial-worker")
    next_result = await client.post(path + "/prepare", headers=headers)
    second = next_result.json()["selection"][1]["job_id"]
    async with database() as db:
        row = await db.get(Review, second)
        row.status = "failed"
        row.error_category = "not_public"
        await db.commit()
    result = await client.post(path + "/prepare", headers=headers)
    assert result.json()["status"] == "partial"
    assert [r["status"] for r in result.json()["selection"]] == ["ready", "failed"]
    async with database() as db:
        with pytest.raises(DBAPIError):
            await db.execute(
                update(CareerInput)
                .where(CareerInput.id == item["id"])
                .values(job_description=JD + " overwritten")
            )
        await db.rollback()
        # Wrong job/revision cannot form a manifest, even through direct SQL.
        first = result.json()["selection"][0]
        with pytest.raises(DBAPIError):
            await db.execute(
                update(CareerSelection)
                .where(CareerSelection.input_id == item["id"], CareerSelection.position == 1)
                .values(revision_id=first["revision_id"], commit_sha=first["commit_sha"])
            )
        await db.rollback()
        connection = await db.connection()

        def refuse(conn):
            cfg = Config("alembic.ini")
            cfg.attributes["connection"] = conn
            with pytest.raises(RuntimeError, match="history"):
                command.downgrade(cfg, "008")

        await connection.run_sync(refuse)
        await db.rollback()
        assert (await db.get(CareerInput, item["id"])).job_description == JD


async def test_direct_invalid_selection_rolls_back(client, database, actors, discovery):
    item = await create(client, actors)
    async with database() as db:
        with pytest.raises(HTTPException):
            await service.select_repositories(
                db, actors["users"][0].id, item["id"], {"repositories": ["foreign/repository"]}
            )
        assert not db.in_transaction()
        assert (await service.detail(db, actors["users"][0].id, item["id"]))["selection"] == []
