"""Authenticated J3 endpoints. Reads never generate, cancel or retry inference."""

from uuid import UUID

from fastapi import APIRouter, HTTPException, Response

from api.routes.career import CareerRoute
from api.routes.career_matching import DbDep, UserDep, status
from api.schemas.career_roadmap import RoadmapCreate, RoadmapDetail
from core.services import career_roadmap as service
from rag.generator.local_provider import GenerationError

router = APIRouter(tags=["career roadmap"], route_class=CareerRoute)


@router.post(
    "/career-assessments/{identity}/roadmap", response_model=RoadmapDetail, status_code=201
)
async def create(identity: UUID, data: RoadmapCreate, user: UserDep, db: DbDep, response: Response):
    try:
        return status(response, await service.create(db, user.id, identity, data))
    except GenerationError:
        raise HTTPException(409, "Career provider configuration unavailable") from None
    finally:
        await db.rollback()


@router.get("/career-assessments/{identity}/roadmap", response_model=RoadmapDetail)
async def by_assessment(identity: UUID, user: UserDep, db: DbDep, response: Response):
    response.headers["Cache-Control"] = "no-store"
    try:
        return await service.by_assessment(db, user.id, identity)
    finally:
        await db.rollback()


@router.get("/career-roadmaps/{identity}", response_model=RoadmapDetail)
async def detail(identity: UUID, user: UserDep, db: DbDep, response: Response):
    response.headers["Cache-Control"] = "no-store"
    try:
        return await service.detail(db, user.id, identity)
    finally:
        await db.rollback()


@router.post("/career-roadmaps/{identity}/cancel", response_model=RoadmapDetail)
async def cancel(identity: UUID, user: UserDep, db: DbDep, response: Response):
    response.headers["Cache-Control"] = "no-store"
    try:
        return await service.cancel(db, user.id, identity)
    finally:
        await db.rollback()
