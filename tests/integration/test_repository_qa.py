# Token-shaped values are synthetic security fixtures, never real credentials.
"""Real PostgreSQL Q&A security; only external model generation is doubled."""

import asyncio
import json
from pathlib import Path

import pytest
from sqlalchemy import select, update

from core.models import Profile, SourceChunk, User
from core.services import repository_qa as service
from rag.generator.local_provider import GenerationError, GenerationResult
from tests.embedding_fixtures import ingested
from tests.integration import test_semantic_retrieval as base

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]
FILES = json.loads((Path(__file__).parents[1] / "fixtures/phase22_qa.json").read_text())[
    "documents"
]


class FakeGeneration:
    calls = []
    supported = True
    fault = None

    async def generate(self, messages, before_request, **kwargs):
        await before_request()
        type(self).calls.append(messages)
        if self.fault == "provider":
            raise GenerationError("transport")
        if "claims" in json.loads(messages[1]["content"]):
            payload = json.loads(messages[1]["content"])
            first = payload["claims"][0]
            text = json.dumps(
                {
                    "support": [
                        {
                            "claim_index": i,
                            "source_fact": "The referenced source defines this function.",
                            "verdict": "supported" if self.supported else "contradicted",
                            "evidence_ids": [r["evidence_id"] for r in c["references"]],
                        }
                        for i, c in enumerate(payload["claims"])
                    ],
                    "coverage": [
                        {
                            "requirement_id": unit["requirement_id"],
                            "covered": self.supported,
                            "claim_indices": [0] if self.supported else [],
                            "answer_quote": first["text"] if self.supported else "",
                            "evidence_ids": [r["evidence_id"] for r in first["references"]]
                            if self.supported
                            else [],
                        }
                        for unit in payload["requirements"]
                    ],
                }
            )
        else:
            payload = json.loads(messages[1]["content"])
            evidence = payload["evidence"]
            key = next(iter(evidence))
            source = evidence[key]["text"]
            quote = source.splitlines()[0]
            text = json.dumps(
                {
                    "status": "answered",
                    "claims": [
                        {
                            "text": "The source defines a function.",
                            "kind": "observed",
                            "references": [{"evidence_id": key, "quote": quote}],
                        }
                    ],
                    "uncertainty": "",
                }
            )
            if self.fault == "id":
                text = text.replace(key, "E999")
            if self.fault == "malformed":
                text = "not json PRIVATE_MODEL_OUTPUT"
            if self.fault == "abstain":
                text = json.dumps(
                    {"status": "insufficient_evidence", "claims": [], "uncertainty": "No support."}
                )
        return GenerationResult(text, "lmstudio", "qwen/qwen3.5-9b", 0.01, None, "stop")


@pytest.fixture
def model(monkeypatch):
    FakeGeneration.calls = []
    FakeGeneration.supported = True
    FakeGeneration.fault = None
    monkeypatch.setattr(service, "make_provider", FakeGeneration)
    return FakeGeneration


async def setup_source(database, actors):
    return await ingested(database, actors, files={"auth.py": FILES["auth.py"].encode()})


async def query(client, actors, source, index=0, **extra):
    return await client.post(
        base.endpoint(source).replace("/search", "/questions"),
        headers=base.auth(actors, index),
        json={"question": "How is the user checked?", "retrieval_mode": "keyword", **extra},
    )


async def test_answer_validates_source_no_writes(client, database, actors, model):
    source = await setup_source(database, actors)
    before = await base.counts(database)
    response = await query(client, actors, source)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "answered" and len(model.calls) == 2
    cite = body["citations"][0]
    async with database() as db:
        chunk = await db.get(SourceChunk, cite["chunk_id"])
        assert cite["quote"] == chunk.text.splitlines()[0] and cite["line_start"] == 1
    assert await base.counts(database) == before


async def test_foreign_inactive_and_unauthenticated_no_model(client, database, actors, model):
    source = await setup_source(database, actors)
    assert (await query(client, actors, source, index=1)).status_code == 404
    assert (
        await client.post(
            base.endpoint(source).replace("/search", "/questions"), json={"question": "x"}
        )
    ).status_code == 401
    async with database() as db:
        await db.execute(
            update(User).where(User.id == actors["users"][0].id).values(is_active=False)
        )
        await db.commit()
    assert (await query(client, actors, source)).status_code == 401
    assert not model.calls


@pytest.mark.parametrize(
    "fault,code,status",
    [
        ("id", 502, "validation_failure"),
        ("malformed", 502, "validation_failure"),
        ("provider", 503, "provider_failure"),
        ("abstain", 200, "insufficient_evidence"),
        ("support", 502, "validation_failure"),
    ],
)
async def test_failure_and_abstention_no_fake_success(
    client, database, actors, model, fault, code, status
):
    source = await setup_source(database, actors)
    model.fault = fault
    model.supported = fault != "support"
    response = await query(client, actors, source)
    assert response.status_code == code, response.text
    assert response.json()["status"] == status and not response.json()["citations"]
    assert "PRIVATE_MODEL_OUTPUT" not in response.text


@pytest.mark.parametrize("fault", ["owner", "inactive", "source"])
async def test_reauthorization_across_generation_no_held_http_locks(
    client, database, actors, model, monkeypatch, fault
):
    source = await setup_source(database, actors)
    entered = asyncio.Event()
    release = asyncio.Event()

    class Paused(FakeGeneration):
        async def generate(self, messages, before_request, **kwargs):
            await before_request()
            entered.set()
            await release.wait()

            async def noop():
                pass

            return await super().generate(messages, noop)

    monkeypatch.setattr(service, "make_provider", Paused)
    task = asyncio.create_task(query(client, actors, source))
    try:
        await asyncio.wait_for(entered.wait(), 10)
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
        release.set()
    response = await task
    assert response.status_code in (401, 404, 409), response.text
    assert "answer" not in response.json()


async def test_missing_index_dirty_session_and_wrong_revision(client, database, actors, model):
    source = await setup_source(database, actors)
    assert (await query(client, actors, source, retrieval_mode="hybrid")).status_code == 409
    other = await ingested(database, actors, files={"x.py": b"def unrelated(): pass"})
    path = (
        base.endpoint(source)
        .replace("/search", "/questions")
        .replace(source.revision_id, other.revision_id)
    )
    response = await client.post(
        path, headers=base.auth(actors), json={"question": "x", "retrieval_mode": "keyword"}
    )
    # Both revisions may share a repository, but the selected revision is never substituted.
    assert response.status_code in (200, 404)
    if response.status_code == 200:
        assert response.json()["revision_id"] == other.revision_id
    async with database() as db:
        user = await db.get(User, actors["users"][0].id)
        user.is_active = False
        with pytest.raises(RuntimeError):
            await service.answer(
                db,
                user.id,
                source.ingestion_result["repository_id"],
                source.revision_id,
                {"question": "x", "retrieval_mode": "keyword"},
            )
        assert db.dirty


async def test_revocation_during_independent_support_check(
    client, database, actors, model, monkeypatch
):
    source = await setup_source(database, actors)
    entered = asyncio.Event()
    release = asyncio.Event()

    class PausedSupport(FakeGeneration):
        async def generate(self, messages, before_request, **kwargs):
            if "claims" in json.loads(messages[1]["content"]):
                await before_request()
                entered.set()
                await release.wait()

                async def noop():
                    pass

                return await super().generate(messages, noop, **kwargs)
            return await super().generate(messages, before_request, **kwargs)

    monkeypatch.setattr(service, "make_provider", PausedSupport)
    task = asyncio.create_task(query(client, actors, source))
    try:
        await asyncio.wait_for(entered.wait(), 10)
        async with database() as db:
            await db.execute(
                update(Profile)
                .where(Profile.id == actors["profiles"][0].id)
                .values(user_id=actors["users"][1].id)
            )
            await asyncio.wait_for(db.commit(), 5)
    finally:
        release.set()
    result = await task
    assert result.status_code == 404 and "citations" not in result.json()


async def test_secret_source_never_sent_to_model(client, database, actors, model):
    source = await ingested(
        database, actors, files={"settings.py": b'api_key = "ghp_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"'}
    )
    result = await query(client, actors, source, question="What is the api_key?")
    assert result.status_code == 200 and result.json()["status"] == "insufficient_evidence"
    assert not model.calls and "ghp_" not in result.text


async def test_uncited_definition_reaches_checker_and_partial_rejection_is_not_published(
    client, database, actors, model, monkeypatch
):
    source = await ingested(
        database,
        actors,
        files={
            "upload.py": (
                b"from options import MAX_PAYLOAD\ndef accept(size):\n"
                b"    return size <= MAX_PAYLOAD"
            ),
            "options.py": b"MAX_PAYLOAD = 8192",
        },
    )

    class CompletenessCheck(FakeGeneration):
        async def generate(self, messages, before_request, **kwargs):
            payload = json.loads(messages[1]["content"])
            if "claims" in payload:
                await before_request()
                type(self).calls.append(messages)
                evidence = payload["canonical_evidence"]
                cited = {
                    ref["evidence_id"] for claim in payload["claims"] for ref in claim["references"]
                }
                assert {item["path"] for item in evidence.values()} == {"upload.py", "options.py"}
                assert set(evidence) - cited  # Available uncited source must not disappear.
                return GenerationResult(
                    json.dumps(
                        {
                            "support": [
                                {
                                    "claim_index": 0,
                                    "evidence_ids": list(cited),
                                    "source_fact": "Source does not establish the full response.",
                                    "verdict": "insufficient",
                                }
                            ],
                            "coverage": [
                                {
                                    "requirement_id": u["requirement_id"],
                                    "covered": False,
                                    "claim_indices": [],
                                    "answer_quote": "",
                                    "evidence_ids": [],
                                }
                                for u in payload["requirements"]
                            ],
                        }
                    ),
                    "lmstudio",
                    "qwen/qwen3.5-9b",
                    0.01,
                    None,
                    "stop",
                )
            return await super().generate(messages, before_request, **kwargs)

    monkeypatch.setattr(service, "make_provider", CompletenessCheck)
    result = await query(
        client,
        actors,
        source,
        question="How does MAX_PAYLOAD limit upload size and what is its byte value?",
    )
    assert result.status_code == 502
    assert result.json()["status"] == "validation_failure"
    assert result.json()["answer"] == "" and result.json()["citations"] == []


async def test_model_supported_flag_cannot_publish_missing_numeric_part(
    client, database, actors, model, monkeypatch
):
    source = await ingested(
        database,
        actors,
        files={
            "upload.py": (
                b"from options import MAX_PAYLOAD\ndef accept(size):\n"
                b"    return size <= MAX_PAYLOAD"
            ),
            "options.py": b"MAX_PAYLOAD = 8192",
        },
    )

    class FalseComplete(FakeGeneration):
        async def generate(self, messages, before_request, **kwargs):
            await before_request()
            type(self).calls.append(messages)
            payload = json.loads(messages[1]["content"])
            if "claims" in payload:
                key = payload["claims"][0]["references"][0]["evidence_id"]
                value = {
                    "support": [
                        {
                            "claim_index": 0,
                            "source_fact": "The source uses the comparison.",
                            "verdict": "supported",
                            "evidence_ids": [key],
                        }
                    ],
                    "coverage": [
                        {
                            "requirement_id": "R1",
                            "covered": True,
                            "claim_indices": [0],
                            "answer_quote": "size <= MAX_PAYLOAD",
                            "evidence_ids": [key],
                        },
                        {
                            "requirement_id": "R2",
                            "covered": True,
                            "claim_indices": [0],
                            "answer_quote": "MAX_PAYLOAD",
                            "evidence_ids": [key],
                        },
                    ],
                }
            else:
                key = next(k for k, v in payload["evidence"].items() if v["path"] == "upload.py")
                value = {
                    "status": "answered",
                    "claims": [
                        {
                            "text": "Uploads are allowed when size <= MAX_PAYLOAD.",
                            "kind": "observed",
                            "references": [{"evidence_id": key}],
                        }
                    ],
                    "uncertainty": "",
                }
            return GenerationResult(
                json.dumps(value), "lmstudio", "qwen/qwen3.5-9b", 0.01, None, "stop"
            )

    monkeypatch.setattr(service, "make_provider", FalseComplete)
    before = await base.counts(database)
    result = await query(
        client,
        actors,
        source,
        question="When does MAX_PAYLOAD permit uploads and how many bytes are allowed?",
    )
    assert result.status_code == 502 and result.json()["error_category"] == "incomplete_output"
    assert result.json()["answer"] == "" and result.json()["citations"] == []
    assert len(model.calls) == 2 and await base.counts(database) == before


async def test_excess_requirements_are_bounded_before_any_model_request(
    client, database, actors, model
):
    source = await setup_source(database, actors)
    result = await query(client, actors, source, question=" and ".join(["which function"] * 7))
    assert result.status_code == 422 and not model.calls


@pytest.mark.parametrize(
    "verdict,category",
    [
        ("contradicted", "contradicted_output"),
        ("insufficient", "insufficient_support"),
        ("invalid", "unsupported_output"),
    ],
)
async def test_semantic_verdicts_cannot_publish_even_with_relevant_citations(
    client, database, actors, model, monkeypatch, verdict, category
):
    source = await setup_source(database, actors)

    class SemanticModel(FakeGeneration):
        async def generate(self, messages, before_request, **kwargs):
            result = await super().generate(messages, before_request, **kwargs)
            if "claims" in json.loads(messages[1]["content"]):
                value = json.loads(result.content)
                value["support"][0].update(
                    verdict=verdict, source_fact="Evidence is relevant but not entailing."
                )
                return GenerationResult(
                    json.dumps(value),
                    result.provider,
                    result.model,
                    result.latency_seconds,
                    result.usage,
                    result.finish_reason,
                )
            return result

    monkeypatch.setattr(service, "make_provider", SemanticModel)
    result = await query(client, actors, source)
    assert result.status_code == 502 and result.json()["error_category"] == category
    assert result.json()["answer"] == "" and result.json()["citations"] == []
    assert "source_fact" not in result.text


async def test_direct_service_foreign_owner_cannot_trigger_inference(database, actors, model):
    from fastapi import HTTPException

    source = await setup_source(database, actors)
    async with database() as db:
        with pytest.raises(HTTPException) as error:
            await service.answer(
                db,
                actors["users"][1].id,
                source.ingestion_result["repository_id"],
                source.revision_id,
                {"question": "Describe the function.", "retrieval_mode": "keyword"},
            )
        assert error.value.status_code == 404
    assert model.calls == []


# Deferred stabilization (2026-11-04): the local provider path now applies the same deterministic
# comparison-direction obligation as the hosted policy. A model double that states the WRONG
# relation, and a checker double that calls it supported and complete, cannot publish.
@pytest.mark.parametrize(
    ("claim", "published"),
    [
        ("permits accepts size not less than LIMIT.", False),
        ("permits accepts size less than or equal to LIMIT.", False),
        ("permits accepts size less than LIMIT.", True),
    ],
)
async def test_local_path_enforces_stated_comparison_direction(
    client, database, actors, model, monkeypatch, claim, published
):
    source = await ingested(
        database, actors, files={"gate.py": b"def permits(size):\n    return size < LIMIT"}
    )

    class DirectionModel(FakeGeneration):
        async def generate(self, messages, before_request, **kwargs):
            await before_request()
            type(self).calls.append(messages)
            payload = json.loads(messages[1]["content"])
            if "claims" in payload:
                key = payload["claims"][0]["references"][0]["evidence_id"]
                value = {
                    "support": [
                        {
                            "claim_index": 0,
                            "source_fact": "The function returns a comparison.",
                            "verdict": "supported",
                            "evidence_ids": [key],
                        }
                    ],
                    "coverage": [
                        {
                            "requirement_id": unit["requirement_id"],
                            "covered": True,
                            "claim_indices": [0],
                            "answer_quote": claim,
                            "evidence_ids": [key],
                        }
                        for unit in payload["requirements"]
                    ],
                }
            else:
                key = next(iter(payload["evidence"]))
                value = {
                    "status": "answered",
                    "claims": [
                        {"text": claim, "kind": "observed", "references": [{"evidence_id": key}]}
                    ],
                    "uncertainty": "",
                }
            return GenerationResult(
                json.dumps(value), "lmstudio", "qwen/qwen3.5-9b", 0.01, None, "stop"
            )

    monkeypatch.setattr(service, "make_provider", DirectionModel)
    before = await base.counts(database)
    result = await query(client, actors, source, question="What comparison does permits use?")
    if published:
        assert result.status_code == 200 and result.json()["status"] == "answered"
        assert result.json()["citations"]
    else:
        assert result.status_code == 502
        assert result.json()["error_category"] == "incomplete_output"
        assert result.json()["answer"] == "" and result.json()["citations"] == []
    assert await base.counts(database) == before
