import json
from pathlib import Path

import httpx
import pytest

from tests.support.gemini.checker import contract_fixture, validate
from tests.support.gemini.provider import MODEL, URL, Budget, Gemini
from rag.generator.local_provider import GenerationError

ROOT = Path(__file__).resolve().parents[2]


def decision():
    answer, _, _, _, _ = contract_fixture(ROOT)
    return {
        "support": [
            {
                "claim_index": i,
                "evidence_ids": ["E" + str(i + 1)],
                "source_fact": claim.text,
                "verdict": "supported",
            }
            for i, claim in enumerate(answer.claims)
        ],
        "coverage": [
            {
                "requirement_id": "R" + str(i + 1),
                "covered": True,
                "claim_indices": [i],
                "answer_quote": claim.text,
                "evidence_ids": ["E" + str(i + 1)],
            }
            for i, claim in enumerate(answer.claims)
        ],
    }


@pytest.mark.asyncio
async def test_exact_checker_no_generation_single_post(tmp_path):
    _, _, _, messages, schema = contract_fixture(ROOT)
    calls = []

    def http(request):
        calls.append(request)
        body = json.loads(request.content)
        assert str(request.url) == URL
        assert body["systemInstruction"]["parts"][0]["text"] == messages[0]["content"]
        assert body["contents"][0]["parts"][0]["text"] == messages[1]["content"]
        assert body["generationConfig"]["responseJsonSchema"] == schema
        return httpx.Response(
            200,
            json={
                "modelVersion": MODEL,
                "candidates": [
                    {
                        "finishReason": "STOP",
                        "content": {
                            "role": "model",
                            "parts": [
                                {"text": json.dumps(decision()), "thoughtSignature": "OPAQUE"}
                            ],
                        },
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 10,
                    "candidatesTokenCount": 10,
                    "totalTokenCount": 20,
                },
            },
        )

    budget = Budget(tmp_path / "ledger.json", maximum=1)
    provider = Gemini("TEST_ONLY", budget, httpx.MockTransport(http))

    async def reserve():
        pass

    result = await provider.generate(messages, reserve, response_schema=schema)
    validate(result.content, ROOT)
    with pytest.raises(GenerationError):
        await provider.generate(messages, reserve, response_schema=schema)
    assert len(calls) == 1 and len(budget.entries) == 1
    assert "OPAQUE" not in budget.path.read_text()


@pytest.mark.parametrize("fault", ["unsupported", "wrong_id", "incomplete", "missing_requirement"])
def test_unchanged_gates_reject(fault):
    value = decision()
    if fault == "unsupported":
        value["support"][0]["verdict"] = "insufficient"
    if fault == "wrong_id":
        value["support"][0]["evidence_ids"] = ["E2"]
    if fault == "incomplete":
        value["coverage"][0]["covered"] = False
    if fault == "missing_requirement":
        value["coverage"].pop()
    with pytest.raises(ValueError):
        validate(json.dumps(value), ROOT)
