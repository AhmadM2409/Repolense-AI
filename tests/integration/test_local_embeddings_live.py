"""Explicitly opt-in LOCAL live contract; never sends requests to Mistral."""

import os

import pytest

from ingestion.embeddings.approved import AsyncEmbeddingProvider, approved_config


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("REPOLENS_TEST_LMSTUDIO_LIVE") != "1",
    reason="Opt in to the existing local LM Studio runtime",
)
async def test_live_local_nomic_model_and_prefix_contract():
    config = approved_config("lmstudio")
    calls = []

    async def reserve():
        calls.append(1)
        assert len(calls) <= 2

    provider = AsyncEmbeddingProvider(config)
    vectors = await provider.embed_documents(["RepoLens local contract fixture"], reserve)
    query = await provider.embed_query("local contract", reserve)
    assert config.model == "text-embedding-nomic-embed-text-v1.5"
    assert len(vectors) == 1 and len(vectors[0]) == len(query) == 768
    assert len(calls) == 2
