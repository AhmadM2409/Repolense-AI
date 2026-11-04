from uuid import UUID
from datetime import datetime
from pydantic import Field, field_validator
from core.contracts import Contract, JobStatus, SourceDocument, TextChunk
from ingestion.github import repository_name


class IngestionCreate(Contract):
    profile_id: UUID
    repository: str = Field(min_length=3, max_length=300)

    @field_validator("repository")
    @classmethod
    def validate_repository(cls, value):
        return repository_name(value)


class IngestionResponse(Contract):
    id: UUID
    profile_id: UUID
    status: JobStatus
    progress_stage: str = "queued"
    progress_pct: int | None = Field(default=None, ge=0, le=100)
    attempt_count: int = 0
    max_attempts: int = 3
    retry_at: datetime | None = None
    cancel_requested_at: datetime | None = None
    error_category: str | None = None
    requested_repository: str
    revision_id: UUID | None
    error_message: str | None
    ingestion_result: dict | None
    created_at: datetime
    updated_at: datetime


class SourceDetail(Contract):
    source: SourceDocument
    chunks: list[TextChunk]
