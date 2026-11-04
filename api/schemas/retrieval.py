"""Server-approved creation and safe publication status."""

from datetime import datetime
from typing import Literal
from uuid import UUID

from core.contracts import SHA, Contract, JobStatus


class IndexCreate(Contract):
    ingestion_id: UUID
    configuration: Literal["default", "lmstudio", "mistral"] = "default"


class IndexStatus(Contract):
    index_id: UUID
    job_id: UUID
    repository_id: UUID
    revision_id: UUID
    commit_sha: SHA
    provider: Literal["lmstudio", "mistral"]
    model: str
    dimensions: int
    metric: Literal["cosine"]
    configuration_fingerprint: str
    state: Literal["pending", "building", "ready", "failed", "cancelled"]
    job_status: JobStatus
    expected_chunks: int
    vector_count: int
    published_count: int
    reserved_requests: int
    error_message: str | None
    created_at: datetime
    completed_at: datetime | None
    published_at: datetime | None
