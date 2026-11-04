"""Strict version-one roadmap contract; observed facts are server-bound, actions proposed."""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import Field

from api.schemas.career_matching import AssessmentMatch, Requirement
from core.contracts import Contract


class RoadmapCreate(Contract):
    hosted_processing_consent: Literal["career-gemini-job-and-source-v1"] = Field(
        description=(
            "Authorize Google processing of the saved requirements, J2 observations and source "
            "quotes for this roadmap; avoid sensitive/private data. Existing career consent "
            "and inference limits apply."
        )
    )


class Proposal(Contract):
    requirement_id: str = Field(pattern=r"^R[0-9a-f]{12}$")
    action_type: Literal["learning", "coding", "testing", "documenting", "deployment"]
    action: str = Field(strict=True, min_length=15, max_length=160)
    gap: str = Field(strict=True, min_length=10, max_length=120)
    rationale: str = Field(strict=True, min_length=10, max_length=120)
    completion_proof: str = Field(strict=True, min_length=15, max_length=160)
    project_citation: int | None = Field(strict=True, ge=0, le=4)


class Proposals(Contract):
    items: list[Proposal] = Field(min_length=1, max_length=4)


class CheckItem(Contract):
    requirement_id: str
    grounded: bool = Field(strict=True)
    individualized: bool = Field(strict=True)
    gap_accurate: bool = Field(strict=True)
    future_only: bool = Field(strict=True)
    observable_proof: bool = Field(strict=True)
    project_supported: bool = Field(strict=True)
    action_type_appropriate: bool = Field(strict=True)


class ProposalCheck(Contract):
    complete: bool = Field(strict=True)
    items: list[CheckItem] = Field(min_length=1, max_length=4)


class Recommendation(Proposal):
    proposal_notice: Literal["Proposed future work; not existing portfolio capability."]
    requirement: Requirement
    observation: AssessmentMatch
    project: dict | None
    priority: Literal["high", "medium"]
    priority_rationale: str
    effort: Literal["unknown"]
    effort_uncertainty: str


class RoadmapResult(Contract):
    recommendations: list[Recommendation] = Field(max_length=4)
    demonstrated_requirement_ids: list[str] = Field(max_length=4)
    coverage_notice: str


class RoadmapDetail(Contract):
    id: UUID
    assessment_id: UUID
    version: Literal[1]
    status: Literal["generating", "complete", "failed", "cancelled"]
    error_category: str | None
    result: RoadmapResult | None
    source_signature: str
    manifest: dict
    inference_spent: int = Field(ge=0, le=12)
    roadmap_inference_spent: int = Field(ge=0, le=2)
    policy_version: Literal["career-roadmap-v1"]
    model: Literal["gemini-3.5-flash-lite"]
    created_at: datetime
    operation_expired: bool
