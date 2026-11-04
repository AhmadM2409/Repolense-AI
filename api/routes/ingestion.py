from uuid import UUID
from fastapi import APIRouter, Depends, Query
from api.middleware.auth import get_current_user
from api.schemas.ingestion import IngestionCreate, IngestionResponse, SourceDetail
from core.database import get_db
from core.services import github_ingestion as service

router = APIRouter(prefix="/repository-ingestions", tags=["repository ingestion"])


@router.post("", response_model=IngestionResponse, status_code=202)
async def submit_ingestion(
    data: IngestionCreate,
    user=Depends(get_current_user),
    db=Depends(get_db),
):
    job, created = await service.submit(db, data.profile_id, user.id, data.repository)
    return IngestionResponse.model_validate(job)


@router.get("/{job_id}", response_model=IngestionResponse)
async def get_ingestion(job_id: UUID, user=Depends(get_current_user), db=Depends(get_db)):
    return IngestionResponse.model_validate(await service.get_ingestion(db, job_id, user.id))


@router.get("/{job_id}/sources")
async def list_sources(
    job_id: UUID,
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=20),
    user=Depends(get_current_user),
    db=Depends(get_db),
):
    return await service.sources(db, job_id, user.id, page, page_size)


@router.get("/{job_id}/sources/{source_id}", response_model=SourceDetail)
async def get_source(
    job_id: UUID, source_id: UUID, user=Depends(get_current_user), db=Depends(get_db)
):
    return await service.source_detail(db, job_id, source_id, user.id)
