"""Regression contracts repaired in 1.6.1; no model/network approximations."""

import pytest

from agent.tools.tech_detector import TechDetector
from ingestion.chunking.semantic_chunker import SemanticChunker
from ingestion.chunking.structural_chunker import StructuralChunker
from rag.evaluator.faithfulness_checker import FaithfulnessChecker
from rag.retriever.keyword_search import KeywordSearcher
from safety.pii_scrubber import PIIScrubber


@pytest.mark.parametrize("kind", [SemanticChunker, StructuralChunker])
def test_real_token_bounds_original_slices_and_full_coverage(kind):
    text = "Preamble kept.\n# Main\n" + ("東京🐍 longword " * 700) + "\n\nLast sentence.\n"
    chunker = kind()
    chunks = chunker.chunk(text, {"source": "fixture"})
    covered = set()
    for index, chunk in enumerate(chunks):
        start, end = chunk.metadata["char_start"], chunk.metadata["char_end"]
        assert chunk.text == text[start:end]
        assert len(chunker.encoder.encode(chunk.text, disallowed_special=())) <= 800
        assert chunk.metadata["chunk_index"] == index
        covered.update(range(start, end))
    assert all(i in covered for i, c in enumerate(text) if not c.isspace())


def test_structural_preamble_and_fenced_heading_are_not_lost():
    text = "Preamble\n# Real\n```python\n# not a heading\nprint(1)\n```\n## Next\nBody\n"
    chunks = StructuralChunker().chunk(text, {})
    assert chunks[0].text == "Preamble\n"
    assert any(
        "# not a heading" in chunk.text and chunk.metadata["heading_path"] == "Real"
        for chunk in chunks
    )


def test_unicode_and_special_token_strings_are_plain_source():
    text = "<|endoftext|> 🐍。" * 500
    chunks = SemanticChunker().chunk(text, {})
    assert len(chunks) > 1
    assert all(
        chunk.text == text[chunk.metadata["char_start"] : chunk.metadata["char_end"]]
        for chunk in chunks
    )


def test_offline_real_tokenizer_cache(monkeypatch):
    import os

    import requests
    import tiktoken
    import tiktoken.registry

    from scripts.prepare_tokenizer import prepare

    prepare(os.environ["TIKTOKEN_CACHE_DIR"], offline=True)

    def no_network(*args, **kwargs):
        raise AssertionError("Offline tokenizer test attempted network access")

    monkeypatch.setattr(requests, "get", no_network)
    previous = tiktoken.registry.ENCODINGS.pop("cl100k_base", None)
    try:
        encoder = tiktoken.get_encoding("cl100k_base")
        assert encoder.decode(encoder.encode("Actual BPE tokens 🐍")) == "Actual BPE tokens 🐍"
    finally:
        if previous is not None:
            tiktoken.registry.ENCODINGS["cl100k_base"] = previous


def test_pii_exact_phone_span_and_non_pii_preservation():
    text = "5 years developing Python applications. Call (555) 123-4567."
    scrubber = PIIScrubber()
    assert scrubber.scrub(text) == "5 years developing Python applications. Call [REDACTED]."
    phone = next(item for item in scrubber.detect(text) if item["type"] == "phone_us")
    assert text[phone["start"] : phone["end"]] == "(555) 123-4567"


@pytest.mark.parametrize("path", ["node_modules/a.js", "build/generated.py", "src/vendor/x.ts"])
def test_vendor_paths_do_not_affect_technology_observations(path):
    assert TechDetector._should_skip_file(path)
    assert not TechDetector._should_skip_file("build.gradle")


def test_empty_text_corpus_and_single_partial_claim_are_not_certified():
    search = KeywordSearcher()
    search.index([{"text": ""}])
    assert search.search("anything") == []
    assert FaithfulnessChecker().check("Python Kubernetes expertise.", [{"text": "Python"}]) == 0
