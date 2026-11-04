"""Hybrid composition under C1's final owner/source/generation locks; no fallback."""

import asyncio

from fastapi import HTTPException

from api.schemas.search import (
    HybridSearchRequest,
    HybridSearchResponse,
    KeywordMatch,
    SemanticSearchRequest,
)
from core.config import settings
from core.services import keyword_retrieval, semantic_retrieval
from rag.retriever.revision_rrf import fuse


def candidate_depth(top_k):
    # Explicitly saturate at the existing standalone cap; never bypass it internally.
    if type(top_k) is not int or not 1 <= top_k <= settings.keyword_search_max_top_k:
        raise HTTPException(422, "Query or top_k exceeds configured search limit")
    return min(20, settings.keyword_search_max_top_k, max(20, top_k * 4))


async def search(db, user_id, repository_id, revision_id, request):
    if db.new or db.dirty or db.deleted:
        raise RuntimeError("Hybrid search requires a clean session")
    request = HybridSearchRequest.model_validate(request)
    depth = candidate_depth(request.top_k)

    ordered = None

    async def prepare(snapshot, limits):
        nonlocal ordered
        ordered, _ = await keyword_retrieval.rank_snapshot(snapshot, request.query, limits)

    async def finalize(current, config, semantic, limits):
        # C1 has revalidated after HTTP and still holds all canonical/generation locks.
        # Initial CPU-only ranks are valid because C1 verified the identical snapshot signature.
        snapshot = current.snapshot
        canonical = {str(c.id): (c, path) for c, path in snapshot.chunks}
        keyword = []
        for position, (ident, score) in enumerate(ordered[:depth], 1):
            chunk, path = canonical[ident]
            keyword_retrieval._bound(len(chunk.text.encode()) <= limits.excerpt_bytes)
            keyword.append(
                KeywordMatch(
                    source_id=chunk.source_id,
                    chunk_id=chunk.id,
                    path=path,
                    commit_sha=snapshot.sha,
                    line_start=chunk.line_start,
                    line_end=chunk.line_end,
                    excerpt=chunk.text,
                    bm25_score=score,
                    rank=position,
                )
            )
        matches = fuse(keyword, semantic.matches, request.top_k)
        return HybridSearchResponse(
            repository_id=snapshot.repository_id,
            revision_id=snapshot.revision_id,
            commit_sha=snapshot.sha,
            provider=config.provider,
            model=config.model,
            dimensions=config.dimensions,
            index_id=current.index_id,
            configuration_fingerprint=config.fingerprint,
            requested_top_k=request.top_k,
            candidate_limit=depth,
            corpus_chunk_count=len(snapshot.chunks),
            keyword_total_match_count=len(ordered),
            keyword_candidate_count=len(keyword),
            semantic_candidate_count=len(semantic.matches),
            unique_candidate_count=len({str(m.chunk_id) for m in [*keyword, *semantic.matches]}),
            matches=matches,
        )

    semantic_request = SemanticSearchRequest(
        query=request.query,
        mode="semantic",
        top_k=depth,
        configuration=request.configuration,
        index_id=request.index_id,
    )
    try:
        async with asyncio.timeout(settings.hybrid_search_timeout_seconds):
            return await semantic_retrieval.search(
                db,
                user_id,
                repository_id,
                revision_id,
                semantic_request,
                _prepare=prepare,
                _finalize=finalize,
            )
    except TimeoutError:
        raise HTTPException(503, "Hybrid search temporarily unavailable") from None
