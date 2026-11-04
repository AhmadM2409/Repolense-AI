"""Owner-scoped durable job status and cancellation."""

# FastAPI dependency defaults are intentional.
# ruff: noqa: B008
from datetime import datetime
from uuid import UUID

from fastapi import APIRouter, Depends

from api.middleware.auth import get_current_user
from core.contracts import Contract, JobStatus
from core.database import get_db
from core.services import job_queue


class JobResponse(Contract):
    id: UUID
    profile_id: UUID
    job_type: str
    status: JobStatus
    progress_stage: str
    progress_pct: int | None
    attempt_count: int
    max_attempts: int
    retry_at: datetime | None
    started_at: datetime | None
    completed_at: datetime | None
    cancel_requested_at: datetime | None
    error_category: str | None
    error_message: str | None


router = APIRouter(prefix="/jobs", tags=["durable jobs"])


@router.get("/{job_id}", response_model=JobResponse)
async def status(job_id: UUID, user=Depends(get_current_user), db=Depends(get_db)):
    return JobResponse.model_validate(await job_queue.owned(db, job_id, user.id))


@router.post("/{job_id}/cancel", response_model=JobResponse)
async def cancel(job_id: UUID, user=Depends(get_current_user), db=Depends(get_db)):
    return JobResponse.model_validate(await job_queue.cancel(db, job_id, user.id))
