"""New read-only workspace contracts: real PostgreSQL, no model inference."""

from uuid import uuid4

import pytest
from sqlalchemy import update

from core.config import settings
from core.models import Profile, User
from core.services import durable_investigation
from tests.embedding_fixtures import ingested
from tests.integration.test_auth_authorization import bearer

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


async def test_catalog_authorization_pagination_and_available_revisions(client, database, actors):
    own = await ingested(database, actors)
    foreign = await ingested(database, actors, index=1)
    headers = bearer(actors["users"][0].id)
    assert (await client.get("/repositories")).status_code == 401
    result = await client.get("/repositories?limit=1", headers=headers)
    assert result.status_code == 200 and result.headers["cache-control"] == "no-store"
    assert [r["id"] for r in result.json()["items"]] == [own.ingestion_result["repository_id"]]
    assert result.json()["next_offset"] is None
    assert (await client.get("/repositories?offset=1", headers=headers)).json()["items"] == []
    assert (await client.get("/repositories?limit=26", headers=headers)).status_code == 422
    assert (await client.get("/repositories?offset=10001", headers=headers)).status_code == 422
    repo = own.ingestion_result["repository_id"]
    revisions = await client.get(f"/repositories/{repo}/revisions", headers=headers)
    assert revisions.json()["items"][0]["id"] == own.revision_id
    assert revisions.json()["items"][0]["commit_sha"] == own.ingestion_result["commit_sha"]
    assert (
        await client.get(
            f"/repositories/{foreign.ingestion_result['repository_id']}/revisions", headers=headers
        )
    ).status_code == 404
    async with database() as db:
        await db.execute(
            update(User).where(User.id == actors["users"][0].id).values(is_active=False)
        )
        await db.commit()
    assert (await client.get("/repositories", headers=headers)).status_code == 401


async def test_history_owned_revision_bounds_and_transfer(client, database, actors, monkeypatch):
    source = await ingested(database, actors)
    uid, repo, rev = (
        actors["users"][0].id,
        source.ingestion_result["repository_id"],
        source.revision_id,
    )
    monkeypatch.setattr(settings, "durable_investigation_enabled", True)
    ids = []
    for i in range(2):
        async with database() as db:
            ids.append(
                await durable_investigation.submit(
                    db,
                    uid,
                    repo,
                    rev,
                    {
                        "objective": f"Describe guard {i}",
                        "hosted_processing_consent": "gemini-source-processing-v1",
                    },
                )
            )
    path = f"/repositories/{repo}/revisions/{rev}/durable-investigations"
    headers = bearer(uid)
    result = await client.get(path + "?limit=1", headers=headers)
    assert result.status_code == 200 and result.json()["next_offset"] == 1
    row = result.json()["items"][0]
    assert row["id"] == ids[-1] and row["objective"] == "Describe guard 1"
    assert set(row) == {"id", "objective", "status", "outcome", "created_at", "completed_at"}
    second = await client.get(path + "?limit=1&offset=1", headers=headers)
    assert second.json()["items"][0]["id"] == ids[0]
    assert (await client.get(path)).status_code == 401
    assert (await client.get(path, headers=bearer(actors["users"][1].id))).status_code == 404
    assert (await client.get(path.replace(rev, str(uuid4())), headers=headers)).status_code == 404
    assert (await client.get(path + "?limit=21", headers=headers)).status_code == 422
    async with database() as db:
        await db.execute(
            update(Profile)
            .where(Profile.id == actors["profiles"][0].id)
            .values(user_id=actors["users"][1].id)
        )
        await db.commit()
    assert (await client.get(path, headers=headers)).status_code == 404
    moved = await client.get(path, headers=bearer(actors["users"][1].id))
    assert moved.status_code == 200 and moved.json()["items"] == []


async def test_policy_reports_flags_without_enabling_features(
    client, database, actors, monkeypatch
):
    source = await ingested(database, actors)
    path = (
        f"/repositories/{source.ingestion_result['repository_id']}/revisions/"
        f"{source.revision_id}/question-policy"
    )
    monkeypatch.setattr(settings, "qa_gemini_enabled", True)
    monkeypatch.setattr(settings, "durable_investigation_enabled", False)
    headers = bearer(actors["users"][0].id)
    assert (await client.get(path, headers=headers)).json()[
        "durable_investigations_available"
    ] is False
    monkeypatch.setattr(settings, "durable_investigation_enabled", True)
    body = (await client.get(path, headers=headers)).json()
    assert body["durable_investigations_available"] is True
    assert not {"api_key", "secret_key", "database_url"} & body.keys()
