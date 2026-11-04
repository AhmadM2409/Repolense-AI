"""Versioned stateless grounded Q&A contracts; model references never select source paths."""

from typing import Annotated, Literal
from uuid import UUID

from pydantic import Field

from core.contracts import SHA, Contract


class QuestionRequest(Contract):
    question: str = Field(strict=True, min_length=1, max_length=1024)
    generation_provider: Literal["lmstudio", "gemini"] | None = None
    hosted_processing_consent: Literal["gemini-source-processing-v1"] | None = None
    retrieval_mode: Literal["keyword", "semantic", "hybrid"] = "hybrid"
    index_id: UUID | None = None


class Reference(Contract):
    evidence_id: str = Field(pattern=r"^E[1-5]$")
    quote: str | None = Field(default=None, strict=True, min_length=1, max_length=1024)


class GeneratedClaim(Contract):
    text: str = Field(strict=True, min_length=1, max_length=800)
    kind: Literal["observed", "inference"]
    references: list[Reference] = Field(min_length=1, max_length=5)


class GeneratedAnswer(Contract):
    status: Literal["answered", "insufficient_evidence"]
    claims: list[GeneratedClaim] = Field(max_length=4)
    uncertainty: str = Field(strict=True, max_length=512)


class RequirementCoverage(Contract):
    requirement_id: str = Field(pattern=r"^R[1-6]$")
    covered: bool = Field(strict=True)
    claim_indices: list[Annotated[int, Field(strict=True, ge=0, le=3)]] = Field(max_length=4)
    answer_quote: str = Field(strict=True, max_length=1600)
    evidence_ids: list[Annotated[str, Field(pattern=r"^E[1-5]$")]] = Field(max_length=5)


class ClaimSupport(Contract):
    claim_index: int = Field(strict=True, ge=0, le=3)
    evidence_ids: list[Annotated[str, Field(pattern=r"^E[1-5]$")]] = Field(max_length=5)
    source_fact: str = Field(strict=True, min_length=1, max_length=512)
    verdict: Literal["supported", "contradicted", "insufficient", "invalid"]


class SupportDecision(Contract):
    support: list[ClaimSupport] = Field(min_length=1, max_length=4)
    coverage: list[RequirementCoverage] = Field(min_length=1, max_length=6)


class AnswerCitation(Contract):
    evidence_id: str
    chunk_id: UUID
    source_id: UUID
    repository_id: UUID
    revision_id: UUID
    commit_sha: SHA
    path: str
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)
    quote: str


class QuestionResponse(Contract):
    contract_version: Literal["repository-qa-v1"] = "repository-qa-v1"
    repository_id: UUID
    revision_id: UUID
    commit_sha: SHA
    question: str
    status: Literal["answered", "insufficient_evidence", "validation_failure", "provider_failure"]
    answer: str = ""
    claims: list[GeneratedClaim] = Field(default_factory=list)
    citations: list[AnswerCitation] = Field(default_factory=list)
    uncertainty: str = ""
    provider: Literal["lmstudio", "gemini"] = "lmstudio"
    model: Literal["qwen/qwen3.5-9b", "gemini-3.5-flash-lite"] = "qwen/qwen3.5-9b"
    policy_version: Literal["grounded-claims-v3", "grounded-claims-v4"] = "grounded-claims-v3"
    generation_metadata: list[dict] = Field(default_factory=list)
    error_category: str | None = None


def checker_schema():
    """Remove descriptive metadata only; preserve every validation constraint and required field."""

    def compact(value):
        if isinstance(value, dict):
            return {k: compact(v) for k, v in value.items() if k not in {"title", "description"}}
        if isinstance(value, list):
            return [compact(v) for v in value]
        return value

    return compact(SupportDecision.model_json_schema())
