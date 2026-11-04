"""Read-only authorized catalog/history for the existing repository AI UI."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Query, Response
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.middleware.auth import get_current_user
from core.database import get_db
from core.models import Investigation, Review, User
from core.services import keyword_retrieval, repository_catalog

router = APIRouter(prefix="/repositories", tags=["repository workspace"])
UserDep = Annotated[User, Depends(get_current_user)]
DbDep = Annotated[AsyncSession, Depends(get_db)]


@router.get("")
async def repositories(
    user: UserDep,
    db: DbDep,
    response: Response,
    limit: int = Query(20, ge=1, le=25),
    offset: int = Query(0, ge=0, le=10000),
):
    response.headers["Cache-Control"] = "no-store"
    try:
        return await repository_catalog.repositories(db, str(user.id), limit, offset)
    finally:
        await db.rollback()


@router.get("/{repository_id}/revisions")
async def revisions(
    repository_id: UUID,
    user: UserDep,
    db: DbDep,
    response: Response,
    limit: int = Query(20, ge=1, le=25),
    offset: int = Query(0, ge=0, le=10000),
):
    response.headers["Cache-Control"] = "no-store"
    try:
        return await repository_catalog.revisions(db, str(user.id), repository_id, limit, offset)
    finally:
        await db.rollback()


@router.get("/{repository_id}/revisions/{revision_id}/durable-investigations")
async def history(
    repository_id: UUID,
    revision_id: UUID,
    user: UserDep,
    db: DbDep,
    response: Response,
    limit: int = Query(10, ge=1, le=20),
    offset: int = Query(0, ge=0, le=10000),
):
    response.headers["Cache-Control"] = "no-store"
    uid, repo, rev = str(user.id), str(repository_id), str(revision_id)
    try:
        await keyword_retrieval._scope(db, uid, repo, rev)
        rows = (
            (
                await db.execute(
                    select(
                        Investigation.id,
                        Investigation.request,
                        Investigation.created_at,
                        Investigation.outcome,
                        Review.status,
                        Review.completed_at,
                    )
                    .join(Review, Review.id == Investigation.id)
                    .where(
                        Investigation.owner_id == uid,
                        Investigation.repository_id == repo,
                        Investigation.revision_id == rev,
                    )
                    .order_by(Investigation.created_at.desc(), Investigation.id.desc())
                    .offset(offset)
                    .limit(limit + 1)
                )
            )
            .mappings()
            .all()
        )
        return {
            "items": [
                {
                    "id": r["id"],
                    "objective": r["request"]["objective"][:1024],
                    "status": r["status"],
                    "outcome": r["outcome"],
                    "created_at": r["created_at"],
                    "completed_at": r["completed_at"],
                }
                for r in rows[:limit]
            ],
            "next_offset": offset + limit if len(rows) > limit else None,
        }
    finally:
        await db.rollback()
