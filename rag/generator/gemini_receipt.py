"""Bounded safe response diagnostics; no thoughts, signatures or headers retained."""

import json


def receipt(raw, key):
    def safe(value):
        return value.replace(key, "[REDACTED]")[:4096] if isinstance(value, str) else None

    result = {"bytes": len(raw), "json_object": False}
    try:
        data = json.loads(raw)
    except (ValueError, UnicodeError):
        return result
    if not isinstance(data, dict):
        return result
    result["json_object"] = True
    result["modelVersion"] = safe(data.get("modelVersion"))
    candidates = data.get("candidates")
    result["candidate_count"] = len(candidates) if isinstance(candidates, list) else None
    result["candidates"] = []
    for candidate in candidates[:10] if isinstance(candidates, list) else []:
        if not isinstance(candidate, dict):
            result["candidates"].append({"type": type(candidate).__name__})
            continue
        content = candidate.get("content")
        parts = content.get("parts") if isinstance(content, dict) else None
        row = {
            "finishReason": safe(candidate.get("finishReason")),
            "parts": [],
            "text": [],
            "has_signature": False,
        }
        for part in parts[:20] if isinstance(parts, list) else []:
            if not isinstance(part, dict):
                row["parts"].append({"type": type(part).__name__})
                continue
            row["parts"].append(
                {"keys_types": {safe(k): type(v).__name__ for k, v in list(part.items())[:20]}}
            )
            row["has_signature"] |= "thoughtSignature" in part
            if part.get("thought") is not True and isinstance(part.get("text"), str):
                row["text"].append(safe(part["text"]))
        result["candidates"].append(row)
    usage = data.get("usageMetadata")
    result["usage"] = (
        {
            k: v
            for k, v in usage.items()
            if k
            in {"promptTokenCount", "candidatesTokenCount", "totalTokenCount", "thoughtsTokenCount"}
            and type(v) is int
            and v >= 0
        }
        if isinstance(usage, dict)
        else None
    )
    return result
