"""Local generation contracts with external HTTP only mocked."""

import asyncio
import copy
import json

import httpx
import pytest

from core.config import settings
from rag.generator.local_provider import GenerationError, LocalGenerationProvider

MODEL = "qwen/qwen3.5-9b"
MESSAGES = [
    {"role": "system", "content": "Return JSON only."},
    {"role": "user", "content": 'Return {"ok":true}.'},
]


@pytest.fixture(autouse=True)
def openai_contract(monkeypatch):
    monkeypatch.setattr(settings, "generation_protocol", "openai")


def transport(fault=None):
    def respond(request):
        if fault == "connection":
            raise httpx.ConnectError("PRIVATE", request=request)
        if fault == "timeout":
            raise httpx.ReadTimeout("PRIVATE", request=request)
        if request.url.path == "/api/v0/models":
            m = {"id": MODEL, "type": "llm", "state": "loaded", "loaded_context_length": 8192}
            if fault == "model":
                m["id"] = "other"
            if fault == "unloaded":
                m["state"] = "not-loaded"
            if fault == "context":
                m["loaded_context_length"] = 128
            return httpx.Response(200, json={"data": [m]})
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": MODEL}]})
        assert request.url.path == "/v1/chat/completions"
        data = json.loads(request.content)
        assert data["model"] == MODEL and "tools" not in data
        value = {
            "model": MODEL,
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": '{"ok":true}'},
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        if fault == "status":
            return httpx.Response(503, text="PRIVATE")
        if fault == "malformed":
            return httpx.Response(200, content=b"not json PRIVATE")
        if fault == "wrong_model":
            value["model"] = "other"
        if fault == "empty":
            value["choices"][0]["message"]["content"] = ""
        if fault == "tool":
            value["choices"][0]["message"]["tool_calls"] = [{}]
        if fault == "length":
            value["choices"][0]["finish_reason"] = "length"
        if fault == "usage":
            value["usage"]["total_tokens"] = True
        if fault == "output":
            value["usage"]["completion_tokens"] = 99999
        if fault == "oversize":
            return httpx.Response(200, content=b"x" * 70000)
        if fault == "choices":
            value["choices"] = []
        return httpx.Response(200, json=value)

    return httpx.MockTransport(respond)


@pytest.mark.asyncio
async def test_success_and_metadata():
    calls = []

    async def before():
        calls.append(1)

    result = await LocalGenerationProvider(transport()).generate(copy.deepcopy(MESSAGES), before)
    assert json.loads(result.content) == {"ok": True} and result.model == MODEL
    assert result.usage["total_tokens"] == 15 and result.latency_seconds >= 0 and calls == [1]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault",
    [
        "connection",
        "timeout",
        "model",
        "unloaded",
        "context",
        "status",
        "malformed",
        "wrong_model",
        "empty",
        "tool",
        "length",
        "usage",
        "output",
        "oversize",
        "choices",
    ],
)
async def test_failures_sanitized_without_fallback(fault):
    async def before():
        pass

    with pytest.raises(GenerationError) as error:
        await LocalGenerationProvider(transport(fault)).generate(MESSAGES, before)
    assert "PRIVATE" not in str(error.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    [
        "https://evil.test/v1",
        "http://localhost:1235/other",
        "http://localhost:1235/v1?url=evil",
        "http://a:b@localhost:1235/v1",
        "http://localhost:443/v1",
        "file:///v1",
    ],
)
async def test_invalid_endpoint(monkeypatch, url):
    monkeypatch.setattr(settings, "generation_base_url", url)

    async def before():
        raise AssertionError()

    with pytest.raises(GenerationError):
        await LocalGenerationProvider(transport()).generate(MESSAGES, before)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "messages",
    [
        [],
        [None],
        [{"role": "tool", "content": "x"}],
        [{"role": "user", "content": "x" * 4097}],
        [{"role": "user", "content": ""}],
        [{"role": "user", "content": "x", "url": "evil"}],
    ],
)
async def test_invalid_input(messages):
    async def before():
        raise AssertionError()

    with pytest.raises(GenerationError):
        await LocalGenerationProvider(transport()).generate(messages, before)


@pytest.mark.asyncio
async def test_cancellation_releases_capacity():
    entered = asyncio.Event()

    async def before():
        entered.set()
        await asyncio.sleep(30)

    task = asyncio.create_task(LocalGenerationProvider(transport()).generate(MESSAGES, before))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    async def ready():
        pass

    assert (await LocalGenerationProvider(transport()).generate(MESSAGES, ready)).model == MODEL


@pytest.mark.asyncio
async def test_capacity_is_bounded_and_recovers():
    entered = asyncio.Event()
    release = asyncio.Event()

    async def paused():
        entered.set()
        await release.wait()

    task = asyncio.create_task(LocalGenerationProvider(transport()).generate(MESSAGES, paused))
    await entered.wait()

    async def noop():
        pass

    try:
        with pytest.raises(GenerationError) as error:
            await LocalGenerationProvider(transport()).generate(MESSAGES, noop)
        assert error.value.code == "capacity"
    finally:
        release.set()
    await task


@pytest.mark.asyncio
async def test_structured_schema_is_server_supplied_and_charged_to_context(monkeypatch):
    seen = []
    base = transport()

    async def handle(request):
        if request.method == "POST":
            seen.append(json.loads(request.content))
        return await base.handle_async_request(request)

    async def noop():
        pass

    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}
    await LocalGenerationProvider(httpx.MockTransport(handle)).generate(
        MESSAGES, noop, response_schema=schema
    )
    assert seen[0]["response_format"]["json_schema"]["schema"] == schema
    monkeypatch.setattr(settings, "generation_max_input_bytes", 256)
    with pytest.raises(GenerationError) as error:
        await LocalGenerationProvider(transport()).generate(
            MESSAGES, noop, response_schema={"description": "x" * 300}
        )
    assert error.value.code == "input_limit"
