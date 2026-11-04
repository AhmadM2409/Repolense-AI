"""Real PostgreSQL scopes; only the planner and external generation are doubled."""

import asyncio
from uuid import uuid4

import pytest
from fastapi import HTTPException

from core.services import investigation as svc
from tests.integration import test_grounding_prototype as prior
from tests.integration import test_semantic_retrieval as base

pytest_plugins = ["tests.integration.test_gemini_production"]

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("one", "completed"),
        ("two", "completed"),
        ("stop", "insufficient_evidence"),
        ("limit", "step_limit"),
        ("bad", "invalid_decision"),
        ("cancel", "cancelled"),
    ],
)
async def test_lifecycle(database, actors, gemini, mode, expected):
    source = await prior.setup(database, actors)
    before = await base.counts(database)
    event = asyncio.Event()
    if mode == "cancel":
        event.set()

    async def planner(objective, steps):
        if mode == "bad":
            return {"operation": "shell", "query": "echo malicious"}
        if mode == "stop":
            return svc.Decision(operation="stop")
        if mode == "two" and not steps or mode == "limit":
            return svc.Decision(operation="search", query="guard")
        return svc.Decision(operation="answer")

    async with database() as db:
        result = await svc.investigate(
            db,
            actors["users"][0].id,
            source.ingestion_result["repository_id"],
            source.revision_id,
            dict(
                objective="Describe guard.",
                max_steps=1 if mode == "limit" else 3,
                hosted_processing_consent="gemini-source-processing-v1",
            ),
            planner=planner,
            cancelled=event,
        )
    assert result.status == expected
    if expected == "completed":
        assert result.final.citations and len(gemini.calls) == 2
        assert len(result.steps) == (2 if mode == "two" else 1)
    else:
        assert result.final is None and not gemini.calls
    assert await base.counts(database) == before


@pytest.mark.parametrize("fault", ["foreign", "revision"])
async def test_scope_denied_before_planner(database, actors, gemini, fault):
    source = await prior.setup(database, actors)

    async def forbidden(*args):
        raise AssertionError("planner reached for foreign scope")

    async with database() as db:
        with pytest.raises(HTTPException) as err:
            await svc.investigate(
                db,
                actors["users"][1 if fault == "foreign" else 0].id,
                source.ingestion_result["repository_id"],
                str(uuid4()) if fault == "revision" else source.revision_id,
                dict(objective="Describe guard."),
                planner=forbidden,
            )
    assert err.value.status_code == 404 and not gemini.calls


@pytest.mark.parametrize("fault", ["id", "malformed", "unsupported", "partial", "timeout"])
async def test_invalid_final_never_published(database, actors, gemini, fault):
    source = await prior.setup(database, actors)
    gemini.fault = fault

    async def planner(*args):
        return svc.Decision(operation="answer")

    async with database() as db:
        result = await svc.investigate(
            db,
            actors["users"][0].id,
            source.ingestion_result["repository_id"],
            source.revision_id,
            dict(
                objective="Describe guard.", hosted_processing_consent="gemini-source-processing-v1"
            ),
            planner=planner,
        )
    assert (
        result.status == ("provider_failure" if fault == "timeout" else "validation_failure")
        and result.final is None
    )


@pytest.mark.parametrize("fault", ["owner", "source", "inactive"])
async def test_scope_rechecked_after_planning(database, actors, gemini, fault):
    from sqlalchemy import select, update

    from core.models import Profile, SourceChunk, User

    source = await prior.setup(database, actors)

    async def planner(*args):
        async with database() as other:
            if fault == "owner":
                await other.execute(
                    update(Profile)
                    .where(Profile.id == actors["profiles"][0].id)
                    .values(user_id=actors["users"][1].id)
                )
            elif fault == "inactive":
                await other.execute(
                    update(User).where(User.id == actors["users"][0].id).values(is_active=False)
                )
            else:
                chunk = await other.scalar(
                    select(SourceChunk)
                    .where(SourceChunk.profile_id == actors["profiles"][0].id)
                    .limit(1)
                )
                chunk.text = "changed"
            await other.commit()
        return svc.Decision(operation="answer")

    async with database() as db:
        if fault == "source":
            result = await svc.investigate(
                db,
                actors["users"][0].id,
                source.ingestion_result["repository_id"],
                source.revision_id,
                dict(objective="Describe guard."),
                planner=planner,
            )
            assert result.status == "scope_changed" and result.final is None
        else:
            with pytest.raises(HTTPException):
                await svc.investigate(
                    db,
                    actors["users"][0].id,
                    source.ingestion_result["repository_id"],
                    source.revision_id,
                    dict(objective="Describe guard."),
                    planner=planner,
                )
    assert not gemini.calls


async def test_timeout_and_provider_failure(database, actors, gemini):
    from rag.generator.local_provider import GenerationError

    source = await prior.setup(database, actors)

    async def slow(*args):
        await asyncio.sleep(2)

    async def broken(*args):
        raise GenerationError("transport")

    for planner, expected in [(slow, "timeout"), (broken, "provider_failure")]:
        async with database() as db:
            result = await svc.investigate(
                db,
                actors["users"][0].id,
                source.ingestion_result["repository_id"],
                source.revision_id,
                dict(objective="Describe guard.", timeout_seconds=1),
                planner=planner,
            )
        assert result.status == expected and result.final is None
    assert not gemini.calls
