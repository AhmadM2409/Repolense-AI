from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field
from core.contracts import JobStatus
from typing import Literal


class FeedbackSection(BaseModel):
    section_name: str
    content: str
    confidence: float = Field(ge=0.0, le=1.0)
    suggestions: list[str]


class ReviewCreate(BaseModel):
    profile_id: UUID
    revision_id: UUID | None = None


class ReviewResponse(BaseModel):
    id: UUID
    profile_id: UUID
    status: JobStatus
    sections: list[FeedbackSection] | None
    overall_score: float | None
    error_message: str | None = None
    revision_id: UUID | None = None
    report_kind: Literal["legacy_unverified", "demo", "evidence"] = "legacy_unverified"
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ReviewListResponse(BaseModel):
    items: list[ReviewResponse]
    total: int
    page: int
    page_size: int


class ReviewStatusResponse(BaseModel):
    review_id: UUID
    status: JobStatus
    progress_pct: int | None = Field(default=None, ge=0, le=100)
    error_message: str | None = None
