"""Bounded local-only async chat generation. No tools, fallback or hosted transmission."""

import asyncio
import hashlib
import json
import threading
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from urllib.parse import urlsplit

import httpx

from core.config import settings

_SLOTS = threading.BoundedSemaphore(settings.generation_concurrency)


class GenerationError(Exception):
    def __init__(self, code):
        self.code = code
        super().__init__("Local generation unavailable")


@dataclass(frozen=True)
class GenerationResult:
    content: str
    provider: str
    model: str
    latency_seconds: float
    usage: dict[str, int] | None
    finish_reason: str
    protocol: str = "openai"
    configuration_fingerprint: str = ""


def configuration():
    try:
        url = urlsplit(settings.generation_base_url)
        valid = (
            url.scheme in {"http", "https"}
            and url.hostname in {"localhost", "127.0.0.1", "host.docker.internal"}
            and url.port == 1235
            and url.path == "/v1"
            and not any((url.username, url.password, url.query, url.fragment))
            and settings.generation_model == "qwen/qwen3.5-9b"
            and settings.generation_protocol in {"openai", "lmstudio-native"}
        )
    except ValueError:
        valid = False
    if not valid:
        raise GenerationError("configuration")
    return settings.generation_base_url, settings.generation_model


def fingerprint():
    configuration()
    names = (
        "generation_base_url",
        "generation_model",
        "generation_protocol",
        "generation_timeout_seconds",
        "generation_max_input_bytes",
        "generation_max_output_tokens",
        "generation_max_response_bytes",
        "generation_concurrency",
        "generation_temperature",
    )
    return hashlib.sha256(
        json.dumps({name: getattr(settings, name) for name in names}, sort_keys=True).encode()
    ).hexdigest()


class LocalGenerationProvider:
    def __init__(self, transport=None):
        self.transport = transport

    async def generate(
        self,
        messages: list[dict],
        before_request: Callable[[], Awaitable[None]],
        *,
        response_schema=None,
    ):
        base, model = configuration()
        protocol = settings.generation_protocol
        identity = fingerprint()
        if (
            not isinstance(messages, list)
            or not 1 <= len(messages) <= 3
            or any(
                not isinstance(m, dict)
                or set(m) != {"role", "content"}
                or not isinstance(m["role"], str)
                or m["role"] not in {"system", "user"}
                or not isinstance(m["content"], str)
                or not m["content"].strip()
                for m in messages
            )
        ):
            raise GenerationError("input")
        try:
            schema_bytes = (
                len(json.dumps(response_schema, allow_nan=False).encode()) if response_schema else 0
            )
        except (ValueError, TypeError):
            raise GenerationError("input") from None
        if response_schema is not None and not isinstance(response_schema, dict):
            raise GenerationError("input")
        schema_bytes += (
            len(b"\nRequired JSON schema: ")
            if response_schema is not None and protocol == "lmstudio-native"
            else 0
        )
        size = sum(len(m["content"].encode()) for m in messages) + schema_bytes
        if size > settings.generation_max_input_bytes:
            raise GenerationError("input_limit")
        if not _SLOTS.acquire(blocking=False):
            raise GenerationError("capacity")
        started = time.perf_counter()
        key = settings.generation_local_api_key.get_secret_value()
        headers = {"Authorization": "Bearer " + key} if key else {}
        try:
            async with (
                asyncio.timeout(settings.generation_timeout_seconds),
                httpx.AsyncClient(
                    timeout=settings.generation_timeout_seconds,
                    trust_env=False,
                    follow_redirects=False,
                    transport=self.transport,
                    headers=headers,
                ) as client,
            ):

                async def request(method, url, **kwargs):
                    async with client.stream(method, url, **kwargs) as response:
                        if response.status_code != 200:
                            raise GenerationError("provider_status")
                        raw = bytearray()
                        async for part in response.aiter_bytes():
                            raw.extend(part)
                            if len(raw) > settings.generation_max_response_bytes:
                                raise GenerationError("response_limit")
                        try:
                            return json.loads(raw)
                        except (ValueError, UnicodeError):
                            raise GenerationError("malformed") from None

                # Native runtime context and OpenAI-compatible identity are both verified.
                native = await request("GET", base.removesuffix("/v1") + "/api/v0/models")
                advertised = await request("GET", base + "/models")
                models = native.get("data") if isinstance(native, dict) else None
                ids = advertised.get("data") if isinstance(advertised, dict) else None
                if (
                    not isinstance(models, list)
                    or not isinstance(ids, list)
                    or not any(isinstance(item, dict) and item.get("id") == model for item in ids)
                ):
                    raise GenerationError("model_unavailable")
                match = next(
                    (m for m in models if isinstance(m, dict) and m.get("id") == model), None
                )
                if (
                    not match
                    or match.get("state") != "loaded"
                    or match.get("type") not in {"llm", "vlm"}
                ):
                    raise GenerationError("model_unavailable")
                context = match.get("loaded_context_length")
                if (
                    type(context) is not int
                    or context < size + settings.generation_max_output_tokens + 512
                ):
                    raise GenerationError("context_limit")
                await before_request()  # Caller releases DB locks before this returns.
                if configuration() != (base, model) or fingerprint() != identity:
                    raise GenerationError("configuration_changed")
                if protocol == "lmstudio-native":
                    system = "\n".join(m["content"] for m in messages if m["role"] == "system")
                    if response_schema is not None:
                        system += "\nRequired JSON schema: " + json.dumps(response_schema)
                    native_result = await request(
                        "POST",
                        base.removesuffix("/v1") + "/api/v1/chat",
                        json={
                            "model": model,
                            "system_prompt": system,
                            "input": [
                                {"type": "text", "content": m["content"]}
                                for m in messages
                                if m["role"] == "user"
                            ],
                            "reasoning": "off",
                            "store": False,
                            "integrations": [],
                            "max_output_tokens": settings.generation_max_output_tokens,
                            "temperature": settings.generation_temperature,
                        },
                    )
                    if (
                        not isinstance(native_result, dict)
                        or native_result.get("model_instance_id") != model
                    ):
                        raise GenerationError("model_mismatch")
                    items = native_result.get("output")
                    stats = native_result.get("stats")
                    if (
                        not isinstance(items, list)
                        or len(items) != 1
                        or not isinstance(items[0], dict)
                        or items[0].get("type") != "message"
                        or not isinstance(items[0].get("content"), str)
                        or not items[0]["content"].strip()
                        or not isinstance(stats, dict)
                    ):
                        raise GenerationError("malformed")
                    values = [
                        stats.get(name)
                        for name in (
                            "input_tokens",
                            "total_output_tokens",
                            "reasoning_output_tokens",
                        )
                    ]
                    if any(type(n) is not int or n < 0 for n in values):
                        raise GenerationError("usage")
                    if values[1] >= settings.generation_max_output_tokens:
                        raise GenerationError("incomplete")
                    if values[2] != 0:
                        raise GenerationError("reasoning_policy")
                    return GenerationResult(
                        items[0]["content"].strip(),
                        "lmstudio",
                        model,
                        time.perf_counter() - started,
                        {"prompt_tokens": values[0], "completion_tokens": values[1]},
                        "not_reported",
                        protocol,
                        identity,
                    )
                value = await request(
                    "POST",
                    base + "/chat/completions",
                    json={
                        "model": model,
                        "messages": messages,
                        "temperature": settings.generation_temperature,
                        "max_tokens": settings.generation_max_output_tokens,
                        "stream": False,
                        **(
                            {
                                "response_format": {
                                    "type": "json_schema",
                                    "json_schema": {
                                        "name": "repository_response",
                                        "strict": True,
                                        "schema": response_schema,
                                    },
                                }
                            }
                            if response_schema is not None
                            else {}
                        ),
                    },
                )
                if not isinstance(value, dict) or value.get("model") != model:
                    raise GenerationError("model_mismatch")
                choices = value.get("choices")
                if (
                    not isinstance(choices, list)
                    or len(choices) != 1
                    or not isinstance(choices[0], dict)
                ):
                    raise GenerationError("malformed")
                choice = choices[0]
                message = choice.get("message")
                if choice.get("finish_reason") != "stop" or not isinstance(message, dict):
                    raise GenerationError("incomplete")
                text = message.get("content")
                if (
                    message.get("role") != "assistant"
                    or message.get("tool_calls")
                    or message.get("function_call")
                    or not isinstance(text, str)
                    or not text.strip()
                    or len(text.encode()) > settings.generation_max_response_bytes
                ):
                    raise GenerationError("malformed")
                usage = value.get("usage")
                if usage is not None:
                    if not isinstance(usage, dict):
                        raise GenerationError("usage")
                    usage = {
                        k: usage[k]
                        for k in ("prompt_tokens", "completion_tokens", "total_tokens")
                        if k in usage
                    }
                    if any(type(n) is not int or n < 0 for n in usage.values()):
                        raise GenerationError("usage")
                    if usage.get("completion_tokens", 0) > settings.generation_max_output_tokens:
                        raise GenerationError("output_limit")
                    if "total_tokens" in usage and usage["total_tokens"] < usage.get(
                        "prompt_tokens", 0
                    ) + usage.get("completion_tokens", 0):
                        raise GenerationError("usage")
                return GenerationResult(
                    text.strip(),
                    "lmstudio",
                    model,
                    time.perf_counter() - started,
                    usage,
                    "stop",
                    protocol,
                    identity,
                )
        except (TimeoutError, httpx.TimeoutException):
            raise GenerationError("timeout") from None
        except httpx.HTTPError:
            raise GenerationError("transport") from None
        except (TypeError, KeyError, AttributeError):
            raise GenerationError("malformed") from None
        finally:
            _SLOTS.release()
