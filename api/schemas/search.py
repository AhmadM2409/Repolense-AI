"""Explicit keyword/semantic contracts; excerpts are exact canonical chunks."""

from typing import Literal
from uuid import UUID

from pydantic import Field, field_validator

from core.contracts import SHA, Contract


class KeywordSearchRequest(Contract):
    query: str = Field(strict=True, min_length=1, max_length=1024)
    top_k: int = Field(default=5, strict=True, ge=1, le=20)
    mode: Literal["keyword"] = "keyword"

    @field_validator("query")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("query must contain non-whitespace text")
        return value.strip()


class QueryMetadata(Contract):
    character_count: int = Field(ge=1)
    term_count: int = Field(ge=0)


class KeywordMatch(Contract):
    source_id: UUID
    chunk_id: UUID
    path: str
    commit_sha: SHA
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)
    excerpt: str
    bm25_score: float = Field(gt=0, allow_inf_nan=False)
    rank: int = Field(ge=1)


class KeywordSearchResponse(Contract):
    repository_id: UUID
    revision_id: UUID
    commit_sha: SHA
    mode: Literal["keyword"] = "keyword"
    query_metadata: QueryMetadata
    total_candidate_count: int = Field(ge=0)
    total_match_count: int = Field(ge=0)
    implementation: str
    matches: list[KeywordMatch]


class SemanticSearchRequest(KeywordSearchRequest):
    mode: Literal["semantic"]
    configuration: Literal["default", "lmstudio", "mistral"] = "default"
    index_id: UUID | None = None


class SemanticQueryMetadata(Contract):
    character_count: int = Field(ge=1)


class SemanticMatch(Contract):
    source_id: UUID
    chunk_id: UUID
    path: str
    commit_sha: SHA
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)
    excerpt: str
    cosine_distance: float = Field(allow_inf_nan=False)
    rank: int = Field(ge=1)
    provenance_verified: Literal[True] = True


class SemanticSearchResponse(Contract):
    repository_id: UUID
    revision_id: UUID
    commit_sha: SHA
    mode: Literal["semantic"] = "semantic"
    provider: Literal["lmstudio", "mistral"]
    model: str
    dimensions: int
    metric: Literal["cosine"] = "cosine"
    index_id: UUID
    configuration_fingerprint: str
    query_metadata: SemanticQueryMetadata
    total_candidate_count: int = Field(ge=0)
    implementation: str
    matches: list[SemanticMatch]


class HybridSearchRequest(SemanticSearchRequest):
    mode: Literal["hybrid"]


class HybridMatch(Contract):
    source_id: UUID
    chunk_id: UUID
    path: str
    commit_sha: SHA
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)
    excerpt: str
    rank: int = Field(strict=True, ge=1)
    rrf_score: float = Field(gt=0, allow_inf_nan=False)
    keyword_rank: int | None = Field(default=None, strict=True, ge=1)
    semantic_rank: int | None = Field(default=None, strict=True, ge=1)
    bm25_score: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    cosine_distance: float | None = Field(default=None, allow_inf_nan=False)
    retrieval_source: Literal["keyword", "semantic", "both"]
    provenance_verified: Literal[True] = True


class HybridSearchResponse(Contract):
    repository_id: UUID
    revision_id: UUID
    commit_sha: SHA
    mode: Literal["hybrid"] = "hybrid"
    provider: Literal["lmstudio", "mistral"]
    model: str
    dimensions: int
    index_id: UUID
    configuration_fingerprint: str
    fusion_algorithm: Literal["rrf"] = "rrf"
    fusion_version: Literal["equal-rank-v1"] = "equal-rank-v1"
    rrf_constant: Literal[60] = 60
    requested_top_k: int = Field(ge=1, le=20)
    candidate_limit: int = Field(ge=1, le=20)
    corpus_chunk_count: int = Field(ge=0)
    keyword_total_match_count: int = Field(ge=0)
    keyword_candidate_count: int = Field(ge=0)
    semantic_candidate_count: int = Field(ge=0)
    unique_candidate_count: int = Field(ge=0)
    matches: list[HybridMatch]
