"""Real PostgreSQL isolation; Gemini HTTP is a deterministic external double."""

import json

import httpx
import pytest
import pytest_asyncio

from core.database import get_db
from tests.support.gemini.provider import MODEL, Budget, Gemini
from tests.support.gemini.service import bind, private_app
from tests.integration import test_grounding_prototype as prior

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


@pytest_asyncio.fixture(params=["production", "prototype"])
async def experiment(database, tmp_path, request):
    model = prior.Model()
    budget = Budget(tmp_path / "ledger.json")

    async def external(req):
        body = json.loads(req.content)
        messages = [
            {"role": "system", "content": body["systemInstruction"]["parts"][0]["text"]},
            {"role": "user", "content": body["contents"][0]["parts"][0]["text"]},
        ]

        async def noop():
            pass

        generated = await model.generate(messages, noop)
        value = json.loads(generated.content)
        if "support" in value and request.param == "production":
            value = {
                "support": [
                    {
                        "claim_index": 0,
                        "verdict": "supported",
                        "evidence_ids": ["E1"],
                        "source_fact": "guard returns 1.",
                    }
                ],
                "coverage": [
                    {
                        "requirement_id": "R1",
                        "covered": True,
                        "claim_indices": [0],
                        "answer_quote": "guard returns 1.",
                        "evidence_ids": ["E1"],
                    }
                ],
            }
        return httpx.Response(
            200,
            json={
                "modelVersion": MODEL,
                "candidates": [
                    {
                        "finishReason": "STOP",
                        "content": {"role": "model", "parts": [{"text": json.dumps(value)}]},
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 10,
                    "candidatesTokenCount": 5,
                    "totalTokenCount": 15,
                },
            },
        )

    provider = Gemini("SYNTHETIC_ONLY", budget, httpx.MockTransport(external))
    answer = bind(lambda: provider, request.param)
    app = private_app(answer)

    async def session():
        async with database() as db:
            yield db

    app.dependency_overrides[get_db] = session
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False), base_url="http://test"
    ) as client:
        yield client, model, answer


async def test_authorized_canonical_no_writes(experiment, database, actors):
    await prior.test_prototype_publishes_canonical_references_and_no_writes(
        experiment, database, actors
    )


async def test_foreign_inactive_unauthenticated_wrong_revision(experiment, database, actors):
    await prior.test_foreign_inactive_unauthenticated_and_wrong_revision_make_no_calls(
        experiment, database, actors
    )


async def test_direct_service_denial_and_dirty_session(experiment, database, actors):
    await prior.test_direct_foreign_and_dirty_session_checks(experiment, database, actors)


async def test_secret_exclusion(experiment, database, actors):
    await prior.test_secret_evidence_is_omitted_and_no_inference(experiment, database, actors)


async def test_injection_no_execution_or_writes(experiment, database, actors):
    await prior.test_prompt_injection_remains_untrusted_data_without_writes(
        experiment, database, actors
    )


@pytest.mark.parametrize("stage", ["generation", "checker"])
@pytest.mark.parametrize("fault", ["owner", "inactive", "source"])
async def test_inflight_revalidation(experiment, database, actors, stage, fault):
    await prior.test_inflight_revalidation_and_unlocked_http(
        experiment, database, actors, stage, fault
    )
