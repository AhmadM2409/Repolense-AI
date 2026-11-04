"""Private provider binding and truthful response identity; ordinary API is untouched."""

from types import SimpleNamespace
from typing import Literal

from fastapi import APIRouter, FastAPI

from api.routes import questions
from api.schemas.questions import QuestionResponse
from core.services import repository_qa
from tests.support.gemini.provider import IDENTITY, MODEL
from tests.support.grounding.prototype import _private_function, make_answer


class ExperimentalResponse(QuestionResponse):
    provider: Literal["gemini"] = "gemini"
    model: Literal["gemini-3.5-flash-lite"] = MODEL


def bind(factory, arm="production"):
    if arm not in {"production", "prototype"}:
        raise ValueError("Unknown contract")
    answer = (
        make_answer(factory)
        if arm == "prototype"
        else _private_function(repository_qa.answer, {"make_provider": factory})
    )
    answer.__globals__.update(QuestionResponse=ExperimentalResponse, fingerprint=lambda: IDENTITY)
    return answer


def private_app(answer):
    app = FastAPI()
    router = APIRouter(prefix="/repositories", route_class=questions.QuestionRoute)
    handler = _private_function(
        questions.question, {"repository_qa": SimpleNamespace(answer=answer)}
    )
    router.add_api_route(
        "/{repository_id}/revisions/{revision_id}/questions",
        handler,
        methods=["POST"],
        response_model=ExperimentalResponse,
    )
    app.include_router(router)
    return app
