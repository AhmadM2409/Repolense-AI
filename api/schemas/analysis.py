from datetime import datetime
from uuid import UUID

from pydantic import Field

from core.contracts import SHA, AnalysisJob, Contract, ReportContract


class AnalysisCreate(Contract):
    ingestion_id: UUID


class AnalysisStatus(AnalysisJob):
    progress_stage: str = "queued"
    attempt_count: int = 0
    max_attempts: int = 3
    retry_at: datetime | None = None
    cancel_requested_at: datetime | None = None
    error_category: str | None = None
    engine_version: str
    producer: str = "deterministic"
    created_at: datetime
    updated_at: datetime
    progress_pct: int | None = Field(default=None, ge=0, le=100)


class AnalysisReport(ReportContract):
    revision_id: UUID
    commit_sha: SHA
    repository: str
    engine_version: str
    producer: str = "deterministic"
    summary: dict
