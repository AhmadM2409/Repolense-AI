"""Versioned bounded J2 contracts; no fit score or recommendations."""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import Field, field_validator

from api.schemas.questions import AnswerCitation, Reference
from core.contracts import Contract

CONSENT = "career-gemini-job-and-source-v1"
Classification = Literal[
    "demonstrated", "partially demonstrated", "documented/claimed", "not evidenced"
]


class AssessmentCreate(Contract):
    request_id: UUID
    hosted_processing_consent: Literal["career-gemini-job-and-source-v1"] = Field(
        description="Explicitly authorize Google Gemini processing of the saved job text, "
        "requirements and selected repository excerpts for this assessment. Do not send "
        "sensitive/private data; unpaid-service inputs may be reviewed/used to improve products."
    )
    retrieval_mode: Literal["keyword", "semantic", "hybrid"] = "keyword"


class MatchNext(Contract):
    requirement_id: str = Field(strict=True, pattern=r"^R[0-9a-f]{12}$")
    reviewed: Literal[True]

    @field_validator("reviewed", mode="before")
    @classmethod
    def explicit_review(cls, value):
        if value is not True:
            raise ValueError("Explicit review confirmation required")
        return value

    hosted_processing_consent: Literal["career-gemini-job-and-source-v1"] = Field(
        description="Explicitly authorize Google Gemini processing of the saved job text, "
        "requirements and selected repository excerpts for this assessment. Do not send "
        "sensitive/private data; unpaid-service inputs may be reviewed/used to improve products."
    )


class RequirementDraft(Contract):
    quote: str = Field(strict=True, min_length=3, max_length=240)
    skill: str = Field(strict=True, min_length=1, max_length=60)
    aliases: list[str] = Field(default_factory=list, max_length=3)
    priority: Literal["required", "preferred"]
    depth: str = Field(strict=True, max_length=100)


class Extraction(Contract):
    requirements: list[RequirementDraft] = Field(min_length=1, max_length=4)


class ExtractionVerdict(Contract):
    requirement_id: str
    supported: bool = Field(strict=True)


class ExtractionCheck(Contract):
    complete: bool = Field(strict=True)
    items: list[ExtractionVerdict] = Field(min_length=1, max_length=4)


class MatchDraft(Contract):
    classification: Classification
    claim: str = Field(strict=True, max_length=256)
    references: list[Reference] = Field(max_length=5)
    uncertainty: str = Field(strict=True, max_length=160)


class MatchCheck(Contract):
    verdict: Literal["supported", "contradicted", "insufficient", "invalid"]
    complete: bool = Field(strict=True)
    classification: Classification


class Requirement(RequirementDraft):
    id: str
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    job_hash: str


class CareerCitation(AnswerCitation):
    evidence_kind: Literal["code", "test", "configuration", "documentation"]


class AssessmentMatch(Contract):
    requirement_id: str
    classification: Classification
    explanation: str
    uncertainty: str
    citations: list[CareerCitation] = Field(max_length=5)
    support_verdict: Literal["supported", "contradicted", "insufficient", "invalid"]
    coverage: dict


class AssessmentDetail(Contract):
    id: UUID
    input_id: UUID
    status: Literal["extracting", "review", "matching", "complete", "failed", "cancelled"]
    error_category: str | None
    manifest: dict
    requirements: list[Requirement] = Field(max_length=4)
    matches: list[AssessmentMatch] = Field(max_length=4)
    inference_spent: int = Field(ge=0, le=12)
    policy_version: Literal["career-evidence-v1"]
    model: Literal["gemini-3.5-flash-lite"]
    retrieval_mode: Literal["keyword", "semantic", "hybrid"]
    created_at: datetime
    operation_expired: bool
    coverage_notice: str


class AssessmentSummary(Contract):
    id: UUID
    status: str
    created_at: datetime
    inference_spent: int


class AssessmentList(Contract):
    items: list[AssessmentSummary] = Field(max_length=20)
