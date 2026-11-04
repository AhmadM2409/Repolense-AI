"""Real PG canonical transfer and premature-answer interception; external inference doubled."""

import json
from uuid import uuid4

import pytest
from fastapi import HTTPException

from core.services import investigation, repository_qa
from rag.generator import hosted
from rag.generator.local_provider import GenerationResult
from tests.integration import test_grounding_prototype as prior
from tests.integration import test_semantic_retrieval as base

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]
pytest_plugins = ["tests.integration.test_gemini_production"]


class CompleteModel:
    calls = []

    async def generate(self, messages, before_request, **kwargs):
        await before_request()
        data = json.loads(messages[1]["content"])
        self.calls.append(data)
        if "claims" not in data:
            evidence = data["evidence"]
            gate = next(k for k, v in evidence.items() if v["path"] == "admit.py")
            limits = next(k for k, v in evidence.items() if v["path"] == "limits.py")
            value = {
                "status": "answered",
                "claims": [
                    {
                        "text": "admit permits n when n is less than CEILING, which is 29.",
                        "kind": "observed",
                        "references": [
                            {"evidence_id": gate, "quote": "return n < CEILING"},
                            {"evidence_id": limits, "quote": "CEILING = 29"},
                        ],
                    }
                ],
                "uncertainty": "",
            }
        else:
            value = {
                "support": [{"claim_index": 0, "verdict": "supported"}],
                "coverage": [
                    {"requirement_id": u["requirement_id"], "complete": True, "claim_indices": [0]}
                    for u in data["requirements"]
                ],
            }
        return GenerationResult(
            json.dumps(value), "gemini", "gemini-3.5-flash-lite", 0.01, None, "stop"
        )


@pytest.mark.parametrize(
    "missing,max_steps,expected",
    [(False, 3, "completed"), (True, 3, "insufficient_evidence"), (False, 1, "step_limit")],
)
async def test_premature_answer_is_intercepted(
    database, actors, gemini, monkeypatch, missing, max_steps, expected
):
    files = {"admit.py": b"from limits import CEILING\ndef admit(n):\n    return n < CEILING\n"}
    if not missing:
        files["limits.py"] = b"CEILING = 29\n"
    source = await prior.setup(database, actors, files)
    model = CompleteModel()
    model.calls = []
    monkeypatch.setattr(hosted, "make_provider", lambda: model)

    async def premature(*args):
        return investigation.Decision(operation="answer")

    before = await base.counts(database)
    async with database() as db:
        result = await investigation.investigate(
            db,
            actors["users"][0].id,
            source.ingestion_result["repository_id"],
            source.revision_id,
            dict(
                objective="What numeric threshold does admit compare to?",
                max_steps=max_steps,
                hosted_processing_consent=hosted.CONSENT,
            ),
            planner=premature,
        )
    assert result.status == expected
    assert result.steps[0].operation == "search" and result.steps[0].origin == "readiness"
    if expected == "completed":
        assert len(model.calls) == 2
        assert {c.path for c in result.final.citations} == {"admit.py", "limits.py"}
        assert len(result.steps) == 2
    else:
        assert not model.calls and result.final is None
    assert await base.counts(database) == before


async def test_foreign_chunk_handoff_cannot_replace_evidence(database, actors, gemini):
    source = await prior.setup(database, actors)
    async with database() as db:
        scope = await repository_qa.capture(
            db,
            str(actors["users"][0].id),
            source.ingestion_result["repository_id"],
            source.revision_id,
            type("R", (), {"retrieval_mode": "keyword"})(),
        )
        await db.rollback()
        with pytest.raises(HTTPException) as error:
            await repository_qa.answer(
                db,
                actors["users"][0].id,
                source.ingestion_result["repository_id"],
                source.revision_id,
                dict(
                    question="Describe guard.",
                    retrieval_mode="keyword",
                    generation_provider="gemini",
                    hosted_processing_consent=hosted.CONSENT,
                ),
                investigation_chunk_ids=(uuid4(),),
                investigation_digest=scope.signature,
            )
    assert error.value.status_code == 404 and not gemini.calls


@pytest.mark.parametrize(
    "objective", ["Does guard use Kafka?", "Describe guard and how storage is configured"]
)
async def test_unresolved_relationship_never_generates(database, actors, gemini, objective):
    source = await prior.setup(database, actors)

    async def premature(*args):
        return investigation.Decision(operation="answer")

    async with database() as db:
        result = await investigation.investigate(
            db,
            actors["users"][0].id,
            source.ingestion_result["repository_id"],
            source.revision_id,
            dict(objective=objective, hosted_processing_consent=hosted.CONSENT),
            planner=premature,
        )
    assert result.status == "insufficient_evidence" and not gemini.calls and result.final is None


@pytest.mark.parametrize("fault", ["foreign", "revision", "digest", "malformed"])
async def test_scoped_evidence_handoff_validation(database, actors, gemini, fault):
    from sqlalchemy import select

    from core.models import SourceChunk
    from tests.embedding_fixtures import ingested

    source = await prior.setup(database, actors)
    other = await ingested(
        database,
        actors,
        index=1 if fault == "foreign" else 0,
        files={"other.py": b"def other():\n    return 8\n"},
    )
    async with database() as db:
        from core.models import IngestedSource

        chunk = await db.scalar(
            select(SourceChunk)
            .join(IngestedSource, SourceChunk.source_id == IngestedSource.id)
            .where(IngestedSource.revision_id == other.revision_id)
            .limit(1)
        )
        chunk_id = str(chunk.id)
        scope = await repository_qa.capture(
            db,
            str(actors["users"][0].id),
            source.ingestion_result["repository_id"],
            source.revision_id,
            type("R", (), {"retrieval_mode": "keyword"})(),
        )
        await db.rollback()
        with pytest.raises(HTTPException) as error:
            await repository_qa.answer(
                db,
                actors["users"][0].id,
                source.ingestion_result["repository_id"],
                source.revision_id,
                dict(
                    question="Describe guard.",
                    retrieval_mode="keyword",
                    generation_provider="gemini",
                    hosted_processing_consent=hosted.CONSENT,
                ),
                investigation_chunk_ids=("not-a-uuid",) if fault == "malformed" else (chunk_id,),
                investigation_digest="changed" if fault == "digest" else scope.signature,
            )
    assert (
        error.value.status_code == ({"digest": 409, "malformed": 422}.get(fault, 404))
        and not gemini.calls
    )


async def test_concurrent_evidence_sets_remain_owner_scoped(database, actors, gemini):
    import asyncio

    from tests.embedding_fixtures import ingested

    sources = [
        await ingested(
            database, actors, index=i, files={"guard.py": b"def guard():\n    return 1\n"}
        )
        for i in range(2)
    ]

    async def planner(*args):
        return investigation.Decision(operation="answer")

    async def run(i):
        async with database() as db:
            return await investigation.investigate(
                db,
                actors["users"][i].id,
                sources[i].ingestion_result["repository_id"],
                sources[i].revision_id,
                dict(objective="Describe guard.", hosted_processing_consent=hosted.CONSENT),
                planner=planner,
            )

    results = await asyncio.gather(run(0), run(1))
    for i, result in enumerate(results):
        assert result.status == "completed" and str(result.user_id) == str(actors["users"][i].id)
        assert all(
            str(c.revision_id) == sources[i].revision_id
            and str(c.repository_id) == sources[i].ingestion_result["repository_id"]
            for c in result.final.citations
        )
    assert {str(c.source_id) for c in results[0].final.citations}.isdisjoint(
        {str(c.source_id) for c in results[1].final.citations}
    )


async def test_explicit_search_then_server_definition_lookup(database, actors, gemini, monkeypatch):
    source = await prior.setup(
        database,
        actors,
        {
            "admit.py": b"from limits import CEILING\ndef admit(n):\n    return n < CEILING\n",
            "limits.py": b"CEILING = 29\n",
        },
    )
    model = CompleteModel()
    model.calls = []
    monkeypatch.setattr(hosted, "make_provider", lambda: model)

    async def planner(objective, steps):
        return (
            investigation.Decision(operation="search", query="admit")
            if not steps
            else investigation.Decision(operation="answer")
        )

    async with database() as db:
        result = await investigation.investigate(
            db,
            actors["users"][0].id,
            source.ingestion_result["repository_id"],
            source.revision_id,
            dict(
                objective="What numeric threshold does admit compare to?",
                max_steps=3,
                hosted_processing_consent=hosted.CONSENT,
            ),
            planner=planner,
        )
    assert result.status == "completed" and len(model.calls) == 2
    assert [s.operation for s in result.steps] == ["search", "search", "answer"]
    assert result.steps[0].evidence_paths == ["admit.py"]
    assert result.steps[1].origin == "readiness" and "limits.py" in result.steps[1].evidence_paths
    assert {c.path for c in result.final.citations} == {"admit.py", "limits.py"}
