"""Authenticated stateless repository questions; no tools or persistent investigations."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Response
from sqlalchemy.ext.asyncio import AsyncSession

from api.middleware.auth import get_current_user
from api.routes.search import BoundedSearchRoute
from api.schemas.questions import QuestionRequest, QuestionResponse
from core.database import get_db
from core.models import User
from core.services import repository_qa


class QuestionRoute(BoundedSearchRoute):
    body_label = "Question"


router = APIRouter(prefix="/repositories", tags=["repository questions"], route_class=QuestionRoute)


@router.post("/{repository_id}/revisions/{revision_id}/questions", response_model=QuestionResponse)
async def question(
    repository_id: UUID,
    revision_id: UUID,
    data: QuestionRequest,
    response: Response,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    result = await repository_qa.answer(db, str(user.id), repository_id, revision_id, data)
    if result.status == "provider_failure":
        response.status_code = 429 if result.error_category == "capacity" else 503
    elif result.status == "validation_failure":
        response.status_code = 502
    return result


@router.get("/{repository_id}/revisions/{revision_id}/question-policy")
async def question_policy(
    repository_id: UUID,
    revision_id: UUID,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    from core.config import settings
    from core.services.keyword_retrieval import _scope
    from rag.generator.hosted import CONSENT

    try:
        await _scope(db, str(user.id), str(repository_id), str(revision_id))
        return {
            "default_provider": settings.qa_provider,
            "hosted_available": settings.qa_gemini_enabled,
            "durable_investigations_available": settings.durable_investigation_enabled and settings.qa_gemini_enabled,
            "consent_version": CONSENT,
            "notice": (
                "Google Gemini receives your question and selected repository source excerpts "
                "for generation and checking. Free-tier inputs may be reviewed by humans and "
                "used to improve Google products. Do not submit sensitive, confidential or "
                "personal information. Authorize only repositories you own and may share. "
                "Consent applies to this question and revision only."
            ),
        }
    finally:
        await db.rollback()
