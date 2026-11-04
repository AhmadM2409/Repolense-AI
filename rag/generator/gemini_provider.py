"""Isolated Gemini REST adapter. No retries, tools, configurable URLs or fallback."""

import asyncio
import hashlib
import json
import time
from pathlib import Path

import httpx

from rag.generator.gemini_receipt import receipt
from rag.generator.local_provider import GenerationError, GenerationResult

MODEL = "gemini-3.5-flash-lite"
URL = "https://generativelanguage.googleapis.com/v1beta/models/" + MODEL + ":generateContent"
IDENTITY = hashlib.sha256((URL + ":synthetic-experiment-v1").encode()).hexdigest()


def load_secret(root):
    """Read only the explicitly ignored private source; never return credential metadata."""
    values = {}
    for line in (Path(root) / ".env.gemini.local").read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if separator and key.strip() in {"GEMINI_API_KEY", "GEMINI_MODEL"}:
            values[key.strip()] = value.strip().strip("\"'")
    if not values.get("GEMINI_API_KEY") or values.get("GEMINI_MODEL") != MODEL:
        raise GenerationError("configuration")
    return values["GEMINI_API_KEY"]


class Budget:
    """One evaluation process and an exclusive persistent ledger; errors consume a slot."""

    def __init__(self, path, maximum=2):
        if type(maximum) is not int or maximum not in {1, 2}:
            raise GenerationError("unverified_quota")
        self.maximum = maximum
        self.path = Path(path)
        self.entries = []
        self.stopped = False
        with self.path.open("x", encoding="utf-8") as handle:
            json.dump([], handle)

    def save(self):
        self.path.write_text(json.dumps(self.entries, indent=2), encoding="utf-8")

    def reserve(self):
        if self.stopped or len(self.entries) >= self.maximum:
            raise GenerationError("experiment_budget")
        row = {"number": len(self.entries) + 1, "state": "reserved_before_post"}
        self.entries.append(row)
        self.save()
        return row


class Gemini:
    def __init__(self, key, budget, transport=None, *, thinking_level=None, deadline=None):
        if not key:
            raise GenerationError("configuration")
        if thinking_level not in {None, "minimal", "low"}:
            raise GenerationError("configuration")
        self.thinking_level = thinking_level
        self.deadline = deadline
        self._key = key
        self.budget = budget
        self.transport = transport

    async def generate(self, messages, before_request, *, response_schema=None):
        if (
            not isinstance(messages, list)
            or len(messages) != 2
            or [m.get("role") for m in messages] != ["system", "user"]
            or any(
                set(m) != {"role", "content"}
                or not isinstance(m["content"], str)
                or not m["content"].strip()
                for m in messages
            )
            or not isinstance(response_schema, dict)
        ):
            raise GenerationError("input")
        try:
            size = sum(len(m["content"].encode()) for m in messages) + len(
                json.dumps(response_schema, allow_nan=False).encode()
            )
        except (ValueError, TypeError):
            raise GenerationError("input") from None
        if size > 4096:
            raise GenerationError("input_limit")
        # REST JSON mode preserves the supplied JSON schema; unsupported schema errors
        # are retained rather than silently dropping constraints or retrying.
        body = {
            "systemInstruction": {"parts": [{"text": messages[0]["content"]}]},
            "contents": [{"role": "user", "parts": [{"text": messages[1]["content"]}]}],
            "generationConfig": {
                "temperature": 0,
                "candidateCount": 1,
                "maxOutputTokens": 512,
                "responseMimeType": "application/json",
                "responseJsonSchema": response_schema,
            },
        }
        if self.thinking_level is not None:
            body["generationConfig"]["thinkingConfig"] = {
                "thinkingLevel": self.thinking_level.upper()
            }
        remaining = (
            min(90.0, self.deadline - asyncio.get_running_loop().time())
            if self.deadline is not None
            else 90.0
        )
        if remaining <= 0:
            raise GenerationError("timeout")
        started = time.perf_counter()
        row = None
        try:
            async with (
                asyncio.timeout(remaining),
                httpx.AsyncClient(
                    trust_env=False,
                    follow_redirects=False,
                    timeout=remaining,
                    transport=self.transport,
                ) as client,
            ):
                await before_request()
                row = self.budget.reserve()
                async with client.stream(
                    "POST", URL, headers={"x-goog-api-key": self._key}, json=body
                ) as response:
                    row["http_status"] = response.status_code
                    if response.status_code != 200:
                        if response.status_code in {401, 402, 403, 429}:
                            self.budget.stopped = True
                        raise GenerationError(
                            "authorization"
                            if response.status_code in {401, 403}
                            else "quota_or_billing"
                            if response.status_code in {402, 429}
                            else "provider_status"
                        )
                    raw = bytearray()
                    async for part in response.aiter_bytes():
                        if len(raw) + len(part) > 65536:
                            raise GenerationError("response_limit")
                        raw.extend(part)
                try:
                    row["receipt"] = receipt(raw, self._key)
                    self.budget.save()
                    row["validation_stage"] = "response_json"
                    data = json.loads(raw)
                    if not isinstance(data, dict) or data.get("promptFeedback", {}).get(
                        "blockReason"
                    ):
                        raise ValueError()

                    row["validation_stage"] = "model_identity"
                    if data.get("modelVersion") != MODEL:
                        raise GenerationError("model_identity")
                    row["validation_stage"] = "candidate_count"
                    candidates = data["candidates"]
                    if not isinstance(candidates, list) or len(candidates) != 1:
                        raise ValueError()
                    candidate = candidates[0]
                    row["validation_stage"] = "finish_reason"
                    if candidate["finishReason"] != "STOP":
                        raise GenerationError("finish_reason")
                    row["validation_stage"] = "content_shape"
                    content = candidate["content"]
                    if content.get("role") != "model" or not content["parts"]:
                        raise ValueError()
                    row["validation_stage"] = "part_shape"
                    if any(
                        not isinstance(p, dict)
                        or set(p) - {"text", "thoughtSignature", "thought"}
                        or not isinstance(p.get("text"), str)
                        or ("thought" in p and p["thought"] is not False)
                        or ("thoughtSignature" in p and not isinstance(p["thoughtSignature"], str))
                        for p in content["parts"]
                    ):
                        raise ValueError()
                    text = "".join(p["text"] for p in content["parts"])
                    if self._key in text:
                        raise GenerationError("unsafe_response")
                    row["validation_stage"] = "generated_json"
                    json.loads(text)
                    row["validation_stage"] = "usage_metadata"
                    usage = data["usageMetadata"]
                    if not isinstance(usage, dict) or any(
                        type(v) is not int or v < 0
                        for k, v in usage.items()
                        if k.endswith("TokenCount")
                    ):
                        raise ValueError()
                    if (
                        not {"promptTokenCount", "candidatesTokenCount", "totalTokenCount"}
                        <= usage.keys()
                    ):
                        raise ValueError()
                except (KeyError, TypeError, ValueError, AttributeError, UnicodeError):
                    raise GenerationError("malformed") from None
                elapsed = time.perf_counter() - started
                safe_usage = {k: v for k, v in usage.items() if k.endswith("TokenCount")}
                row.update(
                    state="completed",
                    model=data["modelVersion"],
                    finish_reason="STOP",
                    usage=safe_usage,
                    content=text,
                    input_bytes=size,
                )
                row["validation_stage"] = "complete"
                return GenerationResult(
                    text, "gemini", MODEL, elapsed, safe_usage, "stop", "gemini-rest", IDENTITY
                )
        except GenerationError as exc:
            if row is not None:
                row["rejection_reason"] = exc.code
            raise
        except (httpx.HTTPError, TimeoutError):
            raise GenerationError("transport_or_timeout") from None
        finally:
            if row is not None:
                row["latency_seconds"] = time.perf_counter() - started
                if row["state"] != "completed":
                    row["state"] = "failed_no_retry"
                self.budget.save()
