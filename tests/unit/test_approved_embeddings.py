"""Strict provider contracts, non-disclosure and explicit hosted approval."""

import copy
import json
from dataclasses import replace

import httpx
import pytest
from pydantic import SecretStr

from core.config import settings
from ingestion.embeddings.approved import (
    AsyncEmbeddingProvider,
    EmbeddingError,
    approved_config,
)


def config(monkeypatch, provider="lmstudio"):
    if provider == "mistral":
        monkeypatch.setattr(settings, "embedding_hosted_source_allowed", True)
        monkeypatch.setattr(
            settings, "embedding_mistral_api_key", SecretStr("fixture-secret-not-real")
        )
        monkeypatch.setattr(settings, "embedding_mistral_max_requests", 10)
    return approved_config(provider)


def response(config, count=2):
    return {
        "object": "list",
        "model": config.model,
        "data": [
            {
                "object": "embedding",
                "index": i,
                "embedding": [1.0] + [0.0] * (config.dimensions - 1),
            }
            for i in range(count)
        ],
    }


def transport(config, payload, captured=None, status=200):
    def handler(request):
        if request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "id": config.model,
                            "type": "embeddings",
                            "loaded_context_length": 2048,
                            "quantization": "Q4_K_M",
                        }
                    ]
                },
            )
        if captured is not None:
            captured.append(json.loads(request.content))
        return httpx.Response(status, json=payload)

    return httpx.MockTransport(handler)


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["lmstudio", "mistral"])
async def test_provider_input_policy_and_query(monkeypatch, provider):
    cfg = config(monkeypatch, provider)
    captured = []
    reservations = []

    async def reserve():
        reservations.append(1)

    adapter = AsyncEmbeddingProvider(cfg, transport(cfg, response(cfg), captured))
    vectors = await adapter.embed_documents(["private source", "other"], reserve)
    assert len(vectors) == 2 and len(vectors[0]) == cfg.dimensions
    assert (
        captured[0]["input"][0]
        == ("search_document: " if provider == "lmstudio" else "") + "private source"
    )
    adapter = AsyncEmbeddingProvider(cfg, transport(cfg, response(cfg, 1), captured))
    assert len(await adapter.embed_query("query", reserve)) == cfg.dimensions
    assert captured[1]["input"][0] == ("search_query: " if provider == "lmstudio" else "") + "query"
    assert len(reservations) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "extra",
        "duplicate",
        "missing_index",
        "wrong_dimension",
        "nan",
        "inf",
        "empty",
        "model",
        "shape",
        "bool",
        "zero",
        "huge",
        "row_shape",
    ],
)
async def test_malformed_responses_rejected(monkeypatch, fault):
    cfg = config(monkeypatch)
    data = response(cfg)
    if fault == "missing":
        data["data"].pop()
    if fault == "extra":
        data["data"].append(copy.deepcopy(data["data"][0]))
    if fault == "duplicate":
        data["data"][1]["index"] = 0
    if fault == "missing_index":
        del data["data"][0]["index"]
    if fault == "wrong_dimension":
        data["data"][0]["embedding"].pop()
    if fault in {"nan", "inf"}:
        data["data"][0]["embedding"][0] = float(fault)
    if fault == "empty":
        data["data"][0]["embedding"] = []
    if fault == "model":
        data["model"] = "qwen/qwen3.5-9b"
    if fault == "shape":
        data = []
    if fault == "bool":
        data["data"][0]["index"] = True
    if fault == "zero":
        data["data"][0]["embedding"] = [0.0] * cfg.dimensions
    if fault == "huge":
        data["data"][0]["embedding"][0] = 1e100
    if fault == "row_shape":
        data["data"][0] = []

    # JSON's standard serializer rejects NaN, so raw bytes exercise strict numeric validation.
    def handler(request):
        if request.method == "GET":
            return transport(cfg, {}).handle_request(request)
        return httpx.Response(200, content=json.dumps(data).encode())

    calls = []

    async def reserve():
        calls.append(1)

    with pytest.raises(EmbeddingError) as error:
        await AsyncEmbeddingProvider(cfg, httpx.MockTransport(handler)).embed_documents(
            ["a", "b"], reserve
        )
    assert not error.value.retryable and len(calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status,retryable,expected",
    [(401, False, 1), (403, False, 1), (400, False, 1), (429, True, 3), (503, True, 3)],
)
async def test_status_retries_bounded_and_sanitized(monkeypatch, status, retryable, expected):
    cfg = config(monkeypatch, "mistral")
    calls = []

    async def reserve():
        calls.append(1)

    with pytest.raises(EmbeddingError) as error:
        await AsyncEmbeddingProvider(
            cfg, transport(cfg, {"error": "SOURCE_SECRET"}, status=status)
        ).embed_documents(["private"], reserve)
    assert error.value.retryable == retryable and len(calls) == expected
    assert "SOURCE_SECRET" not in str(error.value) and "fixture-secret" not in str(error.value)


@pytest.mark.asyncio
async def test_transport_timeout_retry(monkeypatch):
    cfg = config(monkeypatch, "mistral")
    calls = []

    async def reserve():
        calls.append(1)

    def fail(request):
        raise httpx.ReadTimeout("PRIVATE_SOURCE_SECRET")

    with pytest.raises(EmbeddingError) as error:
        await AsyncEmbeddingProvider(cfg, httpx.MockTransport(fail)).embed_documents(
            ["text"], reserve
        )
    assert error.value.retryable and len(calls) == 3
    assert "PRIVATE_SOURCE_SECRET" not in str(error.value)


@pytest.mark.asyncio
async def test_budget_callback_stops_retries(monkeypatch):
    cfg = config(monkeypatch, "mistral")
    calls = []

    async def reserve():
        calls.append(1)
        if len(calls) > 1:
            raise EmbeddingError("request_budget")

    with pytest.raises(EmbeddingError, match="request budget"):
        await AsyncEmbeddingProvider(cfg, transport(cfg, {}, status=429)).embed_documents(
            ["text"], reserve
        )
    assert len(calls) == 2


@pytest.mark.parametrize(
    "field,value",
    [
        ("embedding_lmstudio_base_url", "http://169.254.169.254:1235/v1"),
        ("embedding_lmstudio_base_url", "http://localhost:1235/v1/redirect"),
        ("embedding_lmstudio_base_url", "http://secret@localhost:1235/v1"),
        ("embedding_lmstudio_model", "qwen/qwen3.5-9b"),
        ("embedding_lmstudio_dimensions", 1024),
        ("embedding_mistral_base_url", "http://api.mistral.ai/v1"),
    ],
)
def test_unapproved_configuration(monkeypatch, field, value):
    monkeypatch.setattr(settings, field, value)
    with pytest.raises(EmbeddingError):
        approved_config("mistral" if "mistral" in field else "lmstudio")


def test_hosted_disabled_and_config_version_isolation(monkeypatch):
    monkeypatch.setattr(settings, "embedding_hosted_source_allowed", False)
    with pytest.raises(EmbeddingError, match="hosted"):
        approved_config("mistral")
    local = approved_config("lmstudio")
    assert local.fingerprint != replace(local, version="new-artifact-v2").fingerprint
    assert local.digest("text") != local.digest("different")
    assert len(local.digest("text")) == 64


@pytest.mark.asyncio
async def test_input_and_batch_limits_before_network(monkeypatch):
    cfg = config(monkeypatch)

    async def no_reservation():
        raise AssertionError("No network budget should be used")

    adapter = AsyncEmbeddingProvider(cfg, transport(cfg, {}))
    for texts in (["x" * 2048], ["a"] * (cfg.batch_size + 1), [""]):
        with pytest.raises(EmbeddingError):
            await adapter.embed_documents(texts, no_reservation)


@pytest.mark.asyncio
async def test_model_context_and_quantization_guard(monkeypatch):
    cfg = config(monkeypatch)

    async def reserve():
        raise AssertionError("No embedding request")

    def wrong(request):
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "id": cfg.model,
                        "type": "embeddings",
                        "loaded_context_length": 512,
                        "quantization": "different",
                    }
                ]
            },
        )

    with pytest.raises(EmbeddingError, match="configuration mismatch"):
        await AsyncEmbeddingProvider(cfg, httpx.MockTransport(wrong)).embed_documents(
            ["source"], reserve
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [[], {"data": [1]}, {"data": "invalid"}])
async def test_bad_native_metadata_fails_closed(monkeypatch, payload):
    cfg = config(monkeypatch)

    async def reserve():
        raise AssertionError("No embedding request allowed")

    def invalid(request):
        return httpx.Response(200, json=payload)

    with pytest.raises(EmbeddingError, match="invalid model metadata"):
        await AsyncEmbeddingProvider(cfg, httpx.MockTransport(invalid)).embed_documents(
            ["text"], reserve
        )


@pytest.mark.asyncio
async def test_native_and_embedding_share_server_side_auth(monkeypatch):
    cfg = config(monkeypatch)
    monkeypatch.setattr(settings, "embedding_lmstudio_api_key", SecretStr("private-fixture-key"))
    seen = []
    base = transport(cfg, response(cfg, 1))

    def authenticated(request):
        seen.append(request.headers.get("authorization"))
        return base.handle_request(request)

    async def reserve():
        pass

    await AsyncEmbeddingProvider(cfg, httpx.MockTransport(authenticated)).embed_documents(
        ["text"], reserve
    )
    assert seen == ["Bearer private-fixture-key", "Bearer private-fixture-key"]


def test_unicode_compatibility_expansion_is_bounded_without_changing_input():
    cfg = approved_config("lmstudio")
    text = "ﷺ" * 100
    assert cfg.preprocess(text) == "search_document: " + text
    with pytest.raises(EmbeddingError, match="input limit"):
        cfg.bound(text)
