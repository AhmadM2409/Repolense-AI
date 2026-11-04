"""Authenticated durable indexing. No semantic or hybrid search."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Response
from sqlalchemy.ext.asyncio import AsyncSession

from api.middleware.auth import get_current_user
from api.routes.search import BoundedSearchRoute
from api.schemas.retrieval import IndexCreate, IndexStatus
from core.database import get_db
from core.models import User
from core.services import retrieval_indexing as indexing


class BoundedIndexRoute(BoundedSearchRoute):
    body_label = "Indexing"


router = APIRouter(
    prefix="/retrieval-indexes", tags=["retrieval indexes"], route_class=BoundedIndexRoute
)


@router.post("", response_model=IndexStatus, status_code=202)
async def create(
    data: IndexCreate,
    response: Response,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    user_id = user.id
    ident, created = await indexing.submit(db, user_id, data.ingestion_id, data.configuration)
    result = await indexing.status(db, user_id, ident)
    if not created and result["state"] == "ready":
        response.status_code = 200
    return result


@router.get("/{index_id}", response_model=IndexStatus)
async def status(
    index_id: UUID,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    return await indexing.status(db, user.id, index_id)
