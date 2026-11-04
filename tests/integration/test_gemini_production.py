"""Ordinary authenticated API, real PostgreSQL, external inference doubled only."""

import asyncio

import pytest
from sqlalchemy import select, update

from core.config import settings
from core.models import Profile, SourceChunk, User
from core.services import repository_qa
from rag.generator import hosted
from tests.integration import test_grounding_prototype as prior
from tests.integration import test_semantic_retrieval as base

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


@pytest.fixture
def gemini(monkeypatch):
    model = prior.Model()
    monkeypatch.setattr(settings, "qa_gemini_enabled", True)
    monkeypatch.setattr(hosted, "identity", lambda: "GEMINI_TEST_CONFIG")
    monkeypatch.setattr(hosted, "make_provider", lambda: model)
    return model


async def call(client, actors, source, index=0, **extra):
    return await client.post(
        base.endpoint(source).replace("/search", "/questions"),
        headers=base.auth(actors, index),
        json={
            "question": "Describe guard.",
            "retrieval_mode": "keyword",
            "generation_provider": "gemini",
            "hosted_processing_consent": hosted.CONSENT,
            **extra,
        },
    )


async def test_real_route_consent_owner_and_canonical_no_writes(client, database, actors, gemini):
    source = await prior.setup(database, actors)
    before = await base.counts(database)
    denied = await call(client, actors, source, hosted_processing_consent=None)
    assert denied.status_code == 422 and not gemini.calls
    assert (await call(client, actors, source, index=1)).status_code == 404 and not gemini.calls
    response = await call(client, actors, source)
    assert response.status_code == 200, response.text
    body = response.json()
    assert (
        body["provider"] == "gemini"
        and body["model"] == "gemini-3.5-flash-lite"
        and body["policy_version"] == "grounded-claims-v4"
    )
    assert body["citations"][0]["chunk_id"] and body["citations"][0]["source_id"]
    assert await base.counts(database) == before and len(gemini.calls) == 2


@pytest.mark.parametrize(
    "fault",
    ["id", "wrong_revision_quote", "malformed", "unsupported", "partial", "index", "timeout"],
)
async def test_gemini_failures_no_publication(client, database, actors, gemini, fault):
    source = await prior.setup(database, actors)
    gemini.fault = fault
    response = await call(client, actors, source)
    assert response.status_code in (502, 503), response.text
    assert not response.json()["answer"] and not response.json()["citations"]


@pytest.mark.parametrize("stage", ["generation", "checker"])
@pytest.mark.parametrize("fault", ["owner", "inactive", "source"])
async def test_gemini_inflight_owner_source_active_revalidation(
    client, database, actors, gemini, stage, fault
):
    source = await prior.setup(database, actors)
    gemini.pause = stage
    task = asyncio.create_task(call(client, actors, source))
    try:
        await asyncio.wait_for(gemini.entered.wait(), 10)
        async with database() as db:
            if fault == "owner":
                await db.execute(
                    update(Profile)
                    .where(Profile.id == actors["profiles"][0].id)
                    .values(user_id=actors["users"][1].id)
                )
            elif fault == "inactive":
                await db.execute(
                    update(User).where(User.id == actors["users"][0].id).values(is_active=False)
                )
            else:
                row = await db.scalar(
                    select(SourceChunk)
                    .where(SourceChunk.profile_id == actors["profiles"][0].id)
                    .limit(1)
                )
                row.text = "corrupt"
            await asyncio.wait_for(db.commit(), 5)
    finally:
        gemini.release.set()
    response = await task
    assert response.status_code in (401, 404, 409), response.text


async def test_gemini_direct_foreign_disabled_and_policy(
    client, database, actors, gemini, monkeypatch
):
    source = await prior.setup(database, actors)
    path = base.endpoint(source).replace("/search", "/question-policy")
    policy = await client.get(path, headers=base.auth(actors))
    assert policy.status_code == 200
    assert policy.json()["consent_version"] == hosted.CONSENT
    assert (await client.get(path, headers=base.auth(actors, 1))).status_code == 404
    monkeypatch.setattr(settings, "qa_gemini_enabled", False)
    assert (await call(client, actors, source)).status_code == 409 and not gemini.calls
    from fastapi import HTTPException

    async with database() as db:
        with pytest.raises(HTTPException) as error:
            await repository_qa.answer(
                db,
                actors["users"][1].id,
                source.ingestion_result["repository_id"],
                source.revision_id,
                {
                    "question": "x",
                    "retrieval_mode": "keyword",
                    "generation_provider": "gemini",
                    "hosted_processing_consent": hosted.CONSENT,
                },
            )
        assert error.value.status_code == 404
