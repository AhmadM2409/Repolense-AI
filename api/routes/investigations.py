"""Authenticated synchronous investigations and separately opted-in durable executions."""

import asyncio
from contextlib import suppress
from typing import Annotated, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, Request, Response
from sqlalchemy.ext.asyncio import AsyncSession

from api.middleware.auth import get_current_user
from api.routes.questions import QuestionRoute
from core.database import get_db
from core.models import User
from core.services.investigation import InvestigationRequest, InvestigationResult, investigate


class HostedInvestigationRequest(InvestigationRequest):
    generation_provider: Literal["gemini"]


router = APIRouter(prefix="/repositories", tags=["investigations"], route_class=QuestionRoute)


@router.post(
    "/{repository_id}/revisions/{revision_id}/investigations", response_model=InvestigationResult
)
async def investigation(
    repository_id: UUID,
    revision_id: UUID,
    data: HostedInvestigationRequest,
    request: Request,
    response: Response,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    cancelled = asyncio.Event()

    async def monitor():
        while True:
            if await request.is_disconnected():
                cancelled.set()
                return
            await asyncio.sleep(0.1)

    watcher = asyncio.create_task(monitor())
    try:
        result = await investigate(
            db,
            user.id,
            repository_id,
            revision_id,
            data.model_dump(exclude={"generation_provider"}),
            cancelled=cancelled,
        )
        response.headers["Cache-Control"] = "no-store"
        if result.status == "provider_failure":
            response.status_code = 503
        elif result.status == "validation_failure":
            response.status_code = 502
        elif result.status == "scope_changed":
            response.status_code = 409
        return result
    finally:
        watcher.cancel()
        with suppress(asyncio.CancelledError):
            await watcher


# Separate durable resources preserve the existing synchronous response contract.
@router.post("/{repository_id}/revisions/{revision_id}/durable-investigations", status_code=202)
async def create_durable(
    repository_id: UUID,
    revision_id: UUID,
    data: HostedInvestigationRequest,
    response: Response,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    from core.services import durable_investigation as durable

    response.headers["Cache-Control"] = "no-store"

    identity = await durable.submit(
        db, user.id, repository_id, revision_id, data.model_dump(exclude={"generation_provider"})
    )
    return {"id": identity, "status": "pending"}


@router.get("/{repository_id}/revisions/{revision_id}/durable-investigations/{identity}")
async def get_durable(
    repository_id: UUID,
    revision_id: UUID,
    identity: UUID,
    response: Response,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    from core.services import durable_investigation as durable

    response.headers["Cache-Control"] = "no-store"

    return await durable.status(db, user.id, repository_id, revision_id, identity)


@router.post("/{repository_id}/revisions/{revision_id}/durable-investigations/{identity}/cancel")
async def cancel_durable(
    repository_id: UUID,
    revision_id: UUID,
    identity: UUID,
    response: Response,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    from core.services import durable_investigation as durable

    response.headers["Cache-Control"] = "no-store"

    return await durable.cancel(db, user.id, repository_id, revision_id, identity)
