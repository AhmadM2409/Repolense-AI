"""Authenticated explicit keyword/semantic retrieval; no jobs or findings."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from sqlalchemy.ext.asyncio import AsyncSession

from api.middleware.auth import get_current_user
from api.schemas.search import (
    HybridSearchRequest,
    HybridSearchResponse,
    KeywordSearchRequest,
    KeywordSearchResponse,
    SemanticSearchRequest,
    SemanticSearchResponse,
)
from core.database import get_db
from core.models import User
from core.services import hybrid_retrieval, keyword_retrieval, semantic_retrieval

MAX_BODY_BYTES = 16 * 1024


class BoundedSearchRoute(APIRoute):
    body_label = "Keyword"

    def get_route_handler(self):
        handler = super().get_route_handler()

        async def bounded(request: Request):
            length = request.headers.get("content-length")
            if length is not None:
                try:
                    size = int(length)
                except ValueError:
                    raise HTTPException(400, "Invalid content length") from None
                if size < 0 or size > MAX_BODY_BYTES:
                    raise HTTPException(413, f"{self.body_label} request body too large")
            received = 0

            async def receive():
                nonlocal received
                message = await request.receive()
                if message["type"] == "http.request":
                    received += len(message.get("body", b""))
                    if received > MAX_BODY_BYTES:
                        raise HTTPException(413, f"{self.body_label} request body too large")
                return message

            try:
                return await handler(Request(request.scope, receive=receive))
            except RequestValidationError as exc:
                # Standard 422 structure without echoing source/query secrets or error context.
                return JSONResponse(
                    status_code=422,
                    content={
                        "detail": [
                            {
                                "loc": [
                                    part
                                    if isinstance(part, int)
                                    or part
                                    in {
                                        "body",
                                        "path",
                                        "query",
                                        "header",
                                        "repository_id",
                                        "revision_id",
                                        "top_k",
                                        "mode",
                                        "ingestion_id",
                                        "configuration",
                                        "index_id",
                                    }
                                    else "unknown"
                                    for part in item["loc"]
                                ],
                                "type": item["type"],
                                "msg": item["msg"],
                            }
                            for item in exc.errors()
                        ]
                    },
                )

        return bounded


router = APIRouter(
    prefix="/repositories", tags=["keyword retrieval"], route_class=BoundedSearchRoute
)


@router.post(
    "/{repository_id}/revisions/{revision_id}/search",
    response_model=KeywordSearchResponse | SemanticSearchResponse | HybridSearchResponse,
)
async def keyword_search(
    repository_id: UUID,
    revision_id: UUID,
    data: KeywordSearchRequest | SemanticSearchRequest | HybridSearchRequest,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    if isinstance(data, HybridSearchRequest):
        return await hybrid_retrieval.search(db, user.id, repository_id, revision_id, data)
    if isinstance(data, SemanticSearchRequest):
        return await semantic_retrieval.search(db, user.id, repository_id, revision_id, data)
    return await keyword_retrieval.search(db, user.id, repository_id, revision_id, data)
