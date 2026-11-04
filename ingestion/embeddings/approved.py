"""Approved async embedding contracts. No fallback, chat or legacy vector writes."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import random
import unicodedata
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
from urllib.parse import urlsplit

import httpx

from core.config import settings


class EmbeddingError(Exception):
    def __init__(self, code: str, retryable: bool = False):
        self.code, self.retryable = code, retryable
        super().__init__("Embedding " + code.replace("_", " ") + "; no result published")


@dataclass(frozen=True)
class EmbeddingConfig:
    provider: Literal["lmstudio", "mistral"]
    base_url: str
    model: str
    dimensions: int
    version: str
    batch_size: int
    max_input_tokens: int
    max_batch_tokens: int
    timeout: float
    retries: int
    max_requests: int
    input_policy: str = "utf8-nfkd-byte-upper-bound-plus16-reject-v1"
    metric: str = "cosine"
    artifact: str = ""

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(
            json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def preprocess(self, text: str, query: bool = False) -> str:
        if not isinstance(text, str) or not text.strip():
            raise EmbeddingError("empty_input")
        if self.provider == "lmstudio":
            return ("search_query: " if query else "search_document: ") + text
        return text

    def bound(self, text: str, query: bool = False) -> int:
        # Conservative token upper bound, NOT an invented model-tokenizer count.
        # UTF-8 byte vocabulary bound + prefix and special-token allowance.
        prepared = self.preprocess(text, query)
        count = (
            max(
                len(prepared.encode("utf-8")),
                len(unicodedata.normalize("NFKD", prepared).encode("utf-8")),
            )
            + 16
        )
        if count > self.max_input_tokens:
            raise EmbeddingError("input_limit")
        return count

    def digest(self, text: str) -> str:
        return hashlib.sha256(
            (self.fingerprint + "\n" + self.preprocess(text)).encode()
        ).hexdigest()


def approved_config(name: str = "default") -> EmbeddingConfig:
    provider = settings.embedding_provider if name == "default" else name
    if provider not in {"lmstudio", "mistral"}:
        raise EmbeddingError("configuration_unavailable")
    local = provider == "lmstudio"
    base = (
        settings.embedding_lmstudio_base_url if local else settings.embedding_mistral_base_url
    ).rstrip("/")
    try:
        parts = urlsplit(base)
        port = parts.port
    except ValueError:
        raise EmbeddingError("configuration_unavailable") from None
    if local:
        allowed = (
            parts.scheme in {"http", "https"}
            and parts.hostname in {"localhost", "127.0.0.1", "host.docker.internal", "::1"}
            and port == 1235
            and parts.path == "/v1"
        )
    else:
        allowed = base == "https://api.mistral.ai/v1"
    if not allowed or parts.username or parts.password or parts.query or parts.fragment:
        raise EmbeddingError("configuration_unavailable")
    model = settings.embedding_lmstudio_model if local else settings.embedding_mistral_model
    dimension = (
        settings.embedding_lmstudio_dimensions if local else settings.embedding_mistral_dimensions
    )
    if (model, dimension) != (
        ("text-embedding-nomic-embed-text-v1.5", 768) if local else ("mistral-embed", 1024)
    ):
        raise EmbeddingError("configuration_unavailable")
    if not local and (
        not settings.embedding_hosted_source_allowed
        or not settings.embedding_mistral_api_key.get_secret_value()
        or settings.embedding_mistral_max_requests <= 0
    ):
        raise EmbeddingError("hosted_not_permitted")
    return EmbeddingConfig(
        provider=provider,
        base_url=base,
        model=model,
        dimensions=dimension,
        version=settings.embedding_lmstudio_version
        if local
        else settings.embedding_mistral_version,
        batch_size=settings.embedding_batch_size,
        max_input_tokens=settings.embedding_max_input_tokens,
        max_batch_tokens=settings.embedding_max_batch_tokens,
        timeout=settings.embedding_timeout_seconds,
        retries=settings.embedding_retries,
        max_requests=settings.embedding_lmstudio_max_requests
        if local
        else settings.embedding_mistral_max_requests,
        artifact=settings.embedding_lmstudio_quantization if local else "mistral-embed-2312",
    )


def validate_vectors(vectors, expected: int, dimensions: int) -> list[list[float]]:
    if not isinstance(vectors, list) or len(vectors) != expected:
        raise EmbeddingError("invalid_response")
    result = []
    for vector in vectors:
        if not isinstance(vector, list) or len(vector) != dimensions:
            raise EmbeddingError("dimension_mismatch")
        if any(
            type(value) not in {int, float}
            or not math.isfinite(value)
            or abs(value) > 3.402823466e38
            for value in vector
        ):
            raise EmbeddingError("invalid_vector")
        values = [float(value) for value in vector]
        if math.fsum(value * value for value in values) <= 0:
            raise EmbeddingError("zero_vector")
        result.append(values)
    return result


class AsyncEmbeddingProvider:
    """HTTP-only async adapter; bounded response, retry and exact index validation."""

    def __init__(self, config: EmbeddingConfig, transport=None):
        self.config = config
        self.transport = transport

    async def embed_documents(
        self, texts: list[str], before_request: Callable[[], Awaitable[None]]
    ):
        return await self._embed(texts, False, before_request)

    async def embed_query(
        self, text: str, before_request: Callable[[], Awaitable[None]]
    ) -> list[float]:
        return (await self._embed([text], True, before_request))[0]

    async def _embed(self, texts, query, before_request):
        config = self.config
        if not texts or len(texts) > config.batch_size:
            raise EmbeddingError("batch_limit")
        if sum(config.bound(text, query) for text in texts) > config.max_batch_tokens:
            raise EmbeddingError("batch_limit")
        payload = {
            "model": config.model,
            "input": [config.preprocess(t, query) for t in texts],
            "encoding_format": "float",
        }
        key = (
            settings.embedding_lmstudio_api_key
            if config.provider == "lmstudio"
            else settings.embedding_mistral_api_key
        ).get_secret_value()
        if config.provider == "mistral" and (
            not settings.embedding_hosted_source_allowed or not key or config.max_requests <= 0
        ):
            raise EmbeddingError("hosted_not_permitted")
        headers = {"Authorization": "Bearer " + key} if key else {}
        limit = config.batch_size * config.dimensions * 32 + 65536
        async with httpx.AsyncClient(
            timeout=config.timeout,
            transport=self.transport,
            trust_env=False,
            follow_redirects=False,
        ) as client:
            if config.provider == "lmstudio":
                try:
                    async with client.stream(
                        "GET",
                        config.base_url.removesuffix("/v1") + "/api/v0/models",
                        headers=headers,
                    ) as response:
                        if response.status_code != 200:
                            raise EmbeddingError(
                                "local_model_unavailable",
                                response.status_code in {429, 500, 502, 503, 504},
                            )
                        body = bytearray()
                        async for piece in response.aiter_bytes():
                            body.extend(piece)
                            if len(body) > 65536:
                                raise EmbeddingError("invalid_model_metadata")
                    metadata = json.loads(body)
                    if (
                        not isinstance(metadata, dict)
                        or not isinstance(metadata.get("data"), list)
                        or any(not isinstance(model, dict) for model in metadata["data"])
                    ):
                        raise EmbeddingError("invalid_model_metadata")
                    models = metadata["data"]
                    matches = [m for m in models if m.get("id") == config.model]
                    if (
                        len(matches) != 1
                        or matches[0].get("type") != "embeddings"
                        or type(matches[0].get("loaded_context_length")) is not int
                        or matches[0]["loaded_context_length"] < config.max_input_tokens
                        or matches[0].get("quantization") != config.artifact
                    ):
                        raise EmbeddingError("local_model_configuration_mismatch")
                except httpx.TransportError:
                    raise EmbeddingError("local_model_unavailable", True) from None
                except (KeyError, TypeError, ValueError, RecursionError):
                    raise EmbeddingError("invalid_model_metadata") from None
            for number in range(config.retries + 1):
                await before_request()  # durable reservation BEFORE every billable attempt
                try:
                    async with client.stream(
                        "POST", config.base_url + "/embeddings", json=payload, headers=headers
                    ) as response:
                        if response.status_code != 200:
                            transient = response.status_code == 429 or response.status_code in {
                                500,
                                502,
                                503,
                                504,
                            }
                            raise EmbeddingError(
                                "provider_unavailable" if transient else "provider_rejected",
                                transient,
                            )
                        body = bytearray()
                        async for piece in response.aiter_bytes():
                            body.extend(piece)
                            if len(body) > limit:
                                raise EmbeddingError("response_limit")
                    try:
                        data = json.loads(body)
                        if not isinstance(data, dict) or data.get("object") != "list":
                            raise EmbeddingError("invalid_response")
                        if data.get("model") != config.model:
                            raise EmbeddingError("model_mismatch")
                        rows = data["data"]
                        if not isinstance(rows, list) or len(rows) != len(texts):
                            raise EmbeddingError("invalid_response")
                        by_index = {}
                        for row in rows:
                            if (
                                not isinstance(row, dict)
                                or row.get("object") != "embedding"
                                or type(row.get("index")) is not int
                                or row["index"] in by_index
                            ):
                                raise EmbeddingError("invalid_response")
                            by_index[row["index"]] = row["embedding"]
                        if set(by_index) != set(range(len(texts))):
                            raise EmbeddingError("invalid_response")
                        return validate_vectors(
                            [by_index[i] for i in range(len(texts))], len(texts), config.dimensions
                        )
                    except (KeyError, TypeError, ValueError, RecursionError):
                        raise EmbeddingError("invalid_response") from None
                except httpx.TransportError:
                    error = EmbeddingError("provider_unavailable", True)
                except EmbeddingError as exc:
                    error = exc
                if not error.retryable or number == config.retries:
                    raise error from None
                await asyncio.sleep(random.uniform(0.05, min(2, 0.25 * 2**number)))
        raise EmbeddingError("provider_unavailable", True)


class LMStudioEmbeddingProvider(AsyncEmbeddingProvider):
    """Local Nomic task prefixes and native runtime identity/context checks."""

    def __init__(self, config: EmbeddingConfig, transport=None):
        if config.provider != "lmstudio":
            raise EmbeddingError("configuration_unavailable")
        super().__init__(config, transport)


class MistralEmbeddingProvider(AsyncEmbeddingProvider):
    """Hosted identity preprocessing; never selected as an automatic fallback."""

    def __init__(self, config: EmbeddingConfig, transport=None):
        if config.provider != "mistral":
            raise EmbeddingError("configuration_unavailable")
        super().__init__(config, transport)
