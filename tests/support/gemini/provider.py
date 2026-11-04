"""Compatibility facade for preserved experimental callers."""

from rag.generator.gemini_provider import (  # noqa: F401
    IDENTITY,
    MODEL,
    URL,
    Budget,
    Gemini,
    load_secret,
)
