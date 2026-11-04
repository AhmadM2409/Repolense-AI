import json

import httpx
import pytest

from core.services import repository_qa
from tests.support.gemini.provider import MODEL, URL, Budget, Gemini, load_secret
from tests.support.gemini.service import bind
from rag.generator.local_provider import GenerationError


def response():
    return {
        "modelVersion": MODEL,
        "candidates": [
            {
                "finishReason": "STOP",
                "content": {"role": "model", "parts": [{"text": '{"ok":true}'}]},
            }
        ],
        "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5, "totalTokenCount": 15},
    }


async def reserve():
    pass


@pytest.mark.asyncio
async def test_success_and_exact_allowlist(tmp_path):
    requests = []

    def http(request):
        requests.append(request)
        assert str(request.url) == URL and request.headers["x-goog-api-key"] == "TEST_ONLY"
        assert not request.url.query
        data = json.loads(request.content)
        assert not {"tools", "toolConfig", "cachedContent"} & data.keys()
        assert data["generationConfig"]["maxOutputTokens"] == 512
        assert data["generationConfig"]["responseJsonSchema"] == {"type": "object"}
        return httpx.Response(200, json=response())

    budget = Budget(tmp_path / "ledger.json")
    result = await Gemini("TEST_ONLY", budget, httpx.MockTransport(http)).generate(
        [{"role": "system", "content": "synthetic"}, {"role": "user", "content": "synthetic"}],
        reserve,
        response_schema={"type": "object"},
    )
    assert result.model == MODEL and result.provider == "gemini" and len(requests) == 1
    assert "TEST_ONLY" not in budget.path.read_text()


@pytest.mark.parametrize(
    "fault",
    [
        "429",
        "403",
        "402",
        "500",
        "302",
        "model",
        "finish",
        "json",
        "tools",
        "thought",
        "usage",
        "blocked",
        "many",
        "large",
        "timeout",
    ],
)
@pytest.mark.asyncio
async def test_errors_no_retry_or_leak(tmp_path, fault):
    calls = []

    def http(request):
        calls.append(request)
        data = response()
        if fault.isdigit():
            return httpx.Response(int(fault), text="PRIVATE_PROVIDER_ERROR TEST_ONLY")
        if fault == "model":
            data["modelVersion"] = "other"
        if fault == "finish":
            data["candidates"][0]["finishReason"] = "MAX_TOKENS"
        if fault == "json":
            data["candidates"][0]["content"]["parts"][0]["text"] = "bad"
        if fault == "tools":
            data["candidates"][0]["content"]["parts"] = [{"functionCall": {}}]
        if fault == "thought":
            data["candidates"][0]["content"]["parts"][0]["thought"] = True
        if fault == "usage":
            data["usageMetadata"]["promptTokenCount"] = True
        if fault == "blocked":
            data["promptFeedback"] = {"blockReason": "SAFETY"}
        if fault == "many":
            data["candidates"] *= 2
        if fault == "large":
            return httpx.Response(200, content=b"x" * 65537)
        if fault == "timeout":
            raise httpx.ReadTimeout("TEST_ONLY")
        return httpx.Response(200, json=data)

    budget = Budget(tmp_path / "ledger.json")
    with pytest.raises(GenerationError) as error:
        await Gemini("TEST_ONLY", budget, httpx.MockTransport(http)).generate(
            [{"role": "system", "content": "x"}, {"role": "user", "content": "x"}],
            reserve,
            response_schema={},
        )
    assert len(calls) == 1 and len(budget.entries) == 1
    assert "TEST_ONLY" not in str(error.value) + budget.path.read_text()
    if fault in {"429", "403", "402"}:
        with pytest.raises(GenerationError):
            budget.reserve()


def test_budget_durable_no_resume(tmp_path):
    path = tmp_path / "ledger.json"
    budget = Budget(path)
    budget.reserve()
    budget.reserve()
    with pytest.raises(GenerationError):
        budget.reserve()
    with pytest.raises(FileExistsError):
        Budget(path)
    with pytest.raises(GenerationError):
        Budget(tmp_path / "other", 20)


@pytest.mark.asyncio
async def test_ownership_reservation_precedes_transmission(tmp_path):
    calls = []

    async def denied():
        raise RuntimeError("denied")

    budget = Budget(tmp_path / "ledger.json")
    with pytest.raises(RuntimeError):
        await Gemini("TEST_ONLY", budget, httpx.MockTransport(lambda r: calls.append(r))).generate(
            [{"role": "system", "content": "x"}, {"role": "user", "content": "x"}],
            denied,
            response_schema={},
        )
    assert calls == [] and budget.entries == []


@pytest.mark.asyncio
async def test_input_bound(tmp_path):
    budget = Budget(tmp_path / "ledger.json")
    with pytest.raises(GenerationError, match="Local generation unavailable"):
        await Gemini("TEST_ONLY", budget).generate(
            [{"role": "system", "content": "x" * 4097}, {"role": "user", "content": "x"}],
            reserve,
            response_schema={},
        )
    assert not budget.entries


def test_secret_configuration(tmp_path):
    path = tmp_path / ".env.gemini.local"
    path.write_text("GEMINI_API_KEY=TEST_ONLY\nGEMINI_MODEL=" + MODEL)
    assert load_secret(tmp_path) == "TEST_ONLY"
    path.write_text("GEMINI_API_KEY=TEST_ONLY\nGEMINI_MODEL=other")
    with pytest.raises(GenerationError):
        load_secret(tmp_path)


@pytest.mark.parametrize("arm", ["production", "prototype"])
def test_private_binding_preserves_production(arm):
    before = dict(repository_qa.answer.__globals__)
    answer = bind(lambda: None, arm)
    assert answer is not repository_qa.answer
    assert answer.__globals__["QuestionResponse"].model_fields["provider"].default == "gemini"
    assert repository_qa.answer.__globals__ == before


@pytest.mark.asyncio
async def test_official_text_signature_is_metadata_not_answer(tmp_path):
    data = response()
    data["candidates"][0]["content"]["parts"][0].update(
        thoughtSignature="OPAQUE_NOT_RETAINED", thought=False
    )
    budget = Budget(tmp_path / "ledger.json")
    result = await Gemini(
        "TEST_ONLY", budget, httpx.MockTransport(lambda r: httpx.Response(200, json=data))
    ).generate(
        [{"role": "system", "content": "x"}, {"role": "user", "content": "x"}],
        reserve,
        response_schema={},
    )
    assert json.loads(result.content) == {"ok": True}
    assert "OPAQUE_NOT_RETAINED" not in budget.path.read_text()
    assert budget.entries[0]["receipt"]["candidates"][0]["has_signature"] is True


@pytest.mark.asyncio
async def test_rejected_json_retained_for_safe_diagnostics(tmp_path):
    data = response()
    data["candidates"][0]["content"]["parts"][0]["text"] = "not json"
    budget = Budget(tmp_path / "ledger.json")
    with pytest.raises(GenerationError):
        await Gemini(
            "TEST_ONLY", budget, httpx.MockTransport(lambda r: httpx.Response(200, json=data))
        ).generate(
            [{"role": "system", "content": "x"}, {"role": "user", "content": "x"}],
            reserve,
            response_schema={},
        )
    assert budget.entries[0]["receipt"]["candidates"][0]["text"] == ["not json"]


@pytest.mark.asyncio
async def test_credential_echo_is_never_archived(tmp_path):
    data = response()
    data["candidates"][0]["content"]["parts"][0]["text"] = '{"echo":"TEST_ONLY"}'
    budget = Budget(tmp_path / "ledger.json")
    with pytest.raises(GenerationError):
        await Gemini(
            "TEST_ONLY", budget, httpx.MockTransport(lambda r: httpx.Response(200, json=data))
        ).generate(
            [{"role": "system", "content": "x"}, {"role": "user", "content": "x"}],
            reserve,
            response_schema={},
        )
    assert "TEST_ONLY" not in budget.path.read_text()


def test_single_post_budget_and_safe_part_diagnostics(tmp_path):
    from tests.support.gemini.receipt import receipt

    budget = Budget(tmp_path / "one.json", maximum=1)
    budget.reserve()
    with pytest.raises(GenerationError):
        budget.reserve()
    data = response()
    data["candidates"][0]["content"]["parts"].extend(
        [
            {"text": "DO_NOT_KEEP_THOUGHT", "thought": True},
            {"text": "TEST_ONLY", "thoughtSignature": "DO_NOT_KEEP_SIGNATURE"},
        ]
    )
    value = receipt(json.dumps(data).encode(), "TEST_ONLY")
    serialized = json.dumps(value)
    assert "DO_NOT_KEEP_THOUGHT" not in serialized
    assert "DO_NOT_KEEP_SIGNATURE" not in serialized and "TEST_ONLY" not in serialized
    assert value["candidate_count"] == 1
    assert value["candidates"][0]["parts"][0]["keys_types"] == {"text": "str"}


@pytest.mark.parametrize("raw", [b"not json", b"null", b'{"candidates":[{"content":null}]}'])
def test_receipt_is_safe_on_malformed_shapes(raw):
    from tests.support.gemini.receipt import receipt

    assert isinstance(receipt(raw, "TEST_ONLY"), dict)
