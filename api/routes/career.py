"""Authenticated bounded J1 inputs. GET never schedules work."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Query, Response
from sqlalchemy.ext.asyncio import AsyncSession

from api.middleware.auth import get_current_user
from api.routes.search import BoundedSearchRoute
from api.schemas.career import InputCreate, InputDetail, InputList, Selection
from core.database import get_db
from core.models import User
from core.services import career_input as service


class CareerRoute(BoundedSearchRoute):
    body_label = "Career input"


router = APIRouter(prefix="/career-inputs", tags=["career input"], route_class=CareerRoute)
UserDep = Annotated[User, Depends(get_current_user)]
DbDep = Annotated[AsyncSession, Depends(get_db)]


@router.post("", status_code=201, response_model=InputDetail)
async def create(data: InputCreate, user: UserDep, db: DbDep, response: Response):
    response.headers["Cache-Control"] = "no-store"
    try:
        return await service.create(db, user.id, data)
    finally:
        await db.rollback()


@router.get("", response_model=InputList)
async def listing(
    user: UserDep,
    db: DbDep,
    response: Response,
    limit: int = Query(10, ge=1, le=20),
    offset: int = Query(0, ge=0, le=10000),
):
    response.headers["Cache-Control"] = "no-store"
    try:
        return await service.listing(db, user.id, limit, offset)
    finally:
        await db.rollback()


@router.get("/{identity}", response_model=InputDetail)
async def detail(identity: UUID, user: UserDep, db: DbDep, response: Response):
    response.headers["Cache-Control"] = "no-store"
    try:
        return await service.detail(db, user.id, identity)
    finally:
        await db.rollback()


@router.put("/{identity}/selection", response_model=InputDetail)
async def select(identity: UUID, data: Selection, user: UserDep, db: DbDep, response: Response):
    response.headers["Cache-Control"] = "no-store"
    try:
        return await service.select_repositories(db, user.id, identity, data)
    finally:
        await db.rollback()


@router.post("/{identity}/prepare", status_code=202, response_model=InputDetail)
async def prepare(identity: UUID, user: UserDep, db: DbDep, response: Response):
    response.headers["Cache-Control"] = "no-store"
    try:
        return await service.prepare(db, user.id, identity)
    finally:
        await db.rollback()
