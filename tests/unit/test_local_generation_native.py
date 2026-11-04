"""Native local protocol has explicit no-storage/no-tool/no-reasoning policy."""

import httpx
import pytest

from core.config import settings
from rag.generator.local_provider import GenerationError, LocalGenerationProvider

MODEL = "qwen/qwen3.5-9b"


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", [None, "identity", "tool", "reasoning", "limit", "usage"])
async def test_native_bounds_and_identity(monkeypatch, fault):
    monkeypatch.setattr(settings, "generation_protocol", "lmstudio-native")

    def handler(request):
        if request.url.path == "/api/v0/models":
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "id": MODEL,
                            "state": "loaded",
                            "type": "llm",
                            "loaded_context_length": 8192,
                        }
                    ]
                },
            )
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": MODEL}]})
        import json

        body = json.loads(request.content)
        assert request.url.path == "/api/v1/chat" and body["reasoning"] == "off"
        assert (
            body["integrations"] == []
            and body["store"] is False
            and "previous_response_id" not in body
        )
        value = {
            "model_instance_id": MODEL,
            "output": [{"type": "message", "content": '{"ok":true}'}],
            "stats": {"input_tokens": 20, "total_output_tokens": 5, "reasoning_output_tokens": 0},
        }
        if fault == "identity":
            value["model_instance_id"] = "other"
        if fault == "tool":
            value["output"][0]["type"] = "tool_call"
        if fault == "reasoning":
            value["stats"]["reasoning_output_tokens"] = 1
        if fault == "limit":
            value["stats"]["total_output_tokens"] = settings.generation_max_output_tokens
        if fault == "usage":
            value["stats"]["input_tokens"] = True
        return httpx.Response(200, json=value)

    async def before():
        pass

    provider = LocalGenerationProvider(httpx.MockTransport(handler))
    if fault:
        with pytest.raises(GenerationError):
            await provider.generate([{"role": "user", "content": "Return JSON"}], before)
    else:
        result = await provider.generate([{"role": "user", "content": "Return JSON"}], before)
        assert result.protocol == "lmstudio-native" and result.finish_reason == "not_reported"
        assert result.configuration_fingerprint
