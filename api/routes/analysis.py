# FastAPI dependency defaults are intentional.
# ruff: noqa: B008
from uuid import UUID

from fastapi import APIRouter, Depends

from api.middleware.auth import get_current_user
from api.schemas.analysis import AnalysisCreate, AnalysisReport, AnalysisStatus
from core.database import get_db
from core.services import repository_analysis as service

router = APIRouter(prefix="/analyses", tags=["deterministic repository analysis"])


def status(job):
    return AnalysisStatus(
        id=job.id,
        profile_id=job.profile_id,
        status=job.status,
        revision_id=job.revision_id,
        error_message=job.error_message,
        engine_version=service.metadata(job)["engine_version"],
        progress_pct=job.progress_pct,
        progress_stage=job.progress_stage,
        attempt_count=job.attempt_count, max_attempts=job.max_attempts,
        retry_at=job.retry_at, cancel_requested_at=job.cancel_requested_at,
        error_category=job.error_category,
        created_at=job.created_at,
        updated_at=job.updated_at,
    )


@router.post("", response_model=AnalysisStatus, status_code=202)
async def start(
    data: AnalysisCreate, user=Depends(get_current_user), db=Depends(get_db)
):
    job, created = await service.submit(db, data.ingestion_id, user.id)
    return status(job)


@router.get("/{job_id}", response_model=AnalysisStatus)
async def get_status(job_id: UUID, user=Depends(get_current_user), db=Depends(get_db)):
    return status(await service.get_job(db, job_id, user.id))


@router.get("/{job_id}/report", response_model=AnalysisReport)
async def report(job_id: UUID, user=Depends(get_current_user), db=Depends(get_db)):
    return await service.report(db, job_id, user.id)


@router.get("/{job_id}/summary")
async def summary(job_id: UUID, user=Depends(get_current_user), db=Depends(get_db)):
    result = await service.report(db, job_id, user.id)
    return {
        "revision_id": result["revision_id"],
        "commit_sha": result["commit_sha"],
        "producer": "deterministic",
        "summary": result["summary"],
    }


@router.get("/{job_id}/findings")
async def findings(job_id: UUID, user=Depends(get_current_user), db=Depends(get_db)):
    result = await service.report(db, job_id, user.id)
    return {
        "revision_id": result["revision_id"],
        "commit_sha": result["commit_sha"],
        "items": result["findings"],
        "total": len(result["findings"]),
    }
