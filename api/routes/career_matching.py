"""J2 explicit synchronous stages; no request can select a provider or owner."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy.ext.asyncio import AsyncSession

from api.middleware.auth import get_current_user
from api.routes.career import CareerRoute
from api.schemas.career_matching import (
    AssessmentCreate,
    AssessmentDetail,
    AssessmentList,
    MatchNext,
)
from core.database import get_db
from core.models import User
from core.services import career_assessment as service
from rag.generator.local_provider import GenerationError

router = APIRouter(tags=["career assessment"], route_class=CareerRoute)
UserDep = Annotated[User, Depends(get_current_user)]
DbDep = Annotated[AsyncSession, Depends(get_db)]


def status(response, result):
    response.headers["Cache-Control"] = "no-store"
    if result["status"] == "failed":
        response.status_code = 502 if result["error_category"] == "validation_failure" else 503
    return result


@router.post(
    "/career-inputs/{input_id}/assessments", response_model=AssessmentDetail, status_code=201
)
async def create(
    input_id: UUID, data: AssessmentCreate, user: UserDep, db: DbDep, response: Response
):
    try:
        return status(response, await service.create(db, user.id, input_id, data))
    except GenerationError:
        raise HTTPException(409, "Career provider configuration unavailable") from None
    finally:
        await db.rollback()


@router.get("/career-inputs/{input_id}/assessments", response_model=AssessmentList)
async def listing(input_id: UUID, user: UserDep, db: DbDep, response: Response):
    response.headers["Cache-Control"] = "no-store"
    try:
        return await service.listing(db, user.id, input_id)
    finally:
        await db.rollback()


@router.get("/career-assessments/{identity}", response_model=AssessmentDetail)
async def detail(identity: UUID, user: UserDep, db: DbDep, response: Response):
    response.headers["Cache-Control"] = "no-store"
    try:
        return await service.detail(db, user.id, identity)
    finally:
        await db.rollback()


@router.post("/career-assessments/{identity}/match", response_model=AssessmentDetail)
async def match(identity: UUID, data: MatchNext, user: UserDep, db: DbDep, response: Response):
    try:
        return status(response, await service.match_next(db, user.id, identity, data))
    except GenerationError:
        raise HTTPException(409, "Career provider configuration unavailable") from None
    finally:
        await db.rollback()


@router.post("/career-assessments/{identity}/cancel", response_model=AssessmentDetail)
async def cancel(identity: UUID, user: UserDep, db: DbDep, response: Response):
    response.headers["Cache-Control"] = "no-store"
    try:
        return await service.cancel(db, user.id, identity)
    finally:
        await db.rollback()
