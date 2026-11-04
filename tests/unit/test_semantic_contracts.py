"""Semantic contracts preserve explicit mode and safe distance representation."""

import pytest
from pydantic import ValidationError

from api.schemas.search import KeywordSearchRequest, SemanticSearchRequest
from core.config import settings
from core.services.semantic_retrieval import configuration
from ingestion.embeddings.approved import EmbeddingError


@pytest.mark.parametrize(
    "data",
    [
        {"query": "x"},
        {"query": "", "mode": "semantic"},
        {"query": "  ", "mode": "semantic"},
        {"query": "x" * 1025, "mode": "semantic"},
        {"query": "x", "mode": "hybrid"},
        {"query": "x", "mode": "semantic", "model": "qwen/qwen3.5-9b"},
        {"query": "x", "mode": "semantic", "owner_id": "foreign"},
        {"query": "x", "mode": "semantic", "top_k": True},
        {"query": "x", "mode": "semantic", "top_k": 21},
        {"query": "x", "mode": "semantic", "index_id": "not-uuid"},
        {"query": "x", "mode": "semantic", "configuration": "http://attacker"},
    ],
)
def test_invalid_semantic_requests(data):
    with pytest.raises(ValidationError):
        SemanticSearchRequest.model_validate(data)


def test_keyword_schema_and_semantic_selection_are_distinct():
    assert KeywordSearchRequest(query="x").mode == "keyword"
    request = SemanticSearchRequest(query="x", mode="semantic")
    assert request.top_k == 5 and request.configuration == "default" and request.index_id is None
    with pytest.raises(ValidationError):
        KeywordSearchRequest(query="x", mode="semantic")


def test_hosted_query_approval_is_not_inferred(monkeypatch):
    from pydantic import SecretStr

    monkeypatch.setattr(settings, "embedding_hosted_source_allowed", True)
    monkeypatch.setattr(settings, "embedding_mistral_api_key", SecretStr("not-real"))
    monkeypatch.setattr(settings, "embedding_mistral_max_requests", 10)
    monkeypatch.setattr(settings, "semantic_search_hosted_query_allowed", False)
    with pytest.raises(EmbeddingError, match="hosted query"):
        configuration("mistral")
