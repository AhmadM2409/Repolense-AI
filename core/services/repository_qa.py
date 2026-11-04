"""Stateless repository Q&A using existing owned retrieval and canonical snapshots."""

import asyncio
import json
from dataclasses import dataclass
from uuid import UUID

from fastapi import HTTPException
from pydantic import ValidationError
from sqlalchemy.exc import DBAPIError

from api.schemas.questions import (
    GeneratedAnswer,
    QuestionRequest,
    QuestionResponse,
    checker_schema,
)
from core.config import settings
from core.services import hybrid_retrieval, keyword_retrieval, semantic_retrieval
from ingestion.embeddings.approved import EmbeddingError
from rag.generator import grounded_policy, hosted
from rag.generator.completeness import requirements, validate_coverage
from rag.generator.grounding import (
    SYSTEM,
    VERIFY,
    select_evidence,
    support_payload,
    unsafe,
    validate_answer,
)
from rag.generator.local_provider import GenerationError, LocalGenerationProvider, fingerprint


@dataclass(frozen=True)
class CanonicalCandidate:
    source_id: UUID
    chunk_id: UUID
    path: str
    commit_sha: str
    line_start: int
    line_end: int
    excerpt: str


@dataclass(frozen=True)
class Scope:
    snapshot: object
    signature: str
    index_id: str | None


async def capture(db, uid, repo, rev, request, fixed_index=None):
    limits = keyword_retrieval.SearchLimits.configured()
    await keyword_retrieval._scope(db, uid, repo, rev)
    if request.retrieval_mode == "keyword":
        snap = await keyword_retrieval._snapshot(db, uid, repo, rev, limits)
        return Scope(snap, snap.digest, None)
    index = fixed_index or (str(request.index_id) if request.index_id else None)
    if index is not None:
        present = await db.scalar(
            semantic_retrieval.generations(uid, repo, rev)
            .with_only_columns(semantic_retrieval.RetrievalIndex.id)
            .where(semantic_retrieval.RetrievalIndex.id == index)
        )
        if present is None:
            raise HTTPException(404, "Retrieval index not found")
    config = semantic_retrieval.configuration("lmstudio")
    view = await semantic_retrieval.view(db, uid, repo, rev, config, limits, index)
    return Scope(view.snapshot, view.signature, view.index_id)


def make_provider():
    return LocalGenerationProvider()


async def answer(
    db,
    user_id,
    repository_id,
    revision_id,
    request,
    *,
    gemini_factory=None,
    investigation_chunk_ids=(),
    investigation_digest=None,
):
    if db.new or db.dirty or db.deleted:
        raise RuntimeError("Question answering requires a clean session")
    request = QuestionRequest.model_validate(request)
    if not request.question.strip() or unsafe(request.question):
        raise HTTPException(422, "Question input unavailable or unsafe")
    try:
        repo, rev, uid = (
            str(UUID(str(repository_id))),
            str(UUID(str(revision_id))),
            str(UUID(str(user_id))),
        )
    except (ValueError, TypeError):
        raise HTTPException(404, "Repository revision not found") from None
    if request.retrieval_mode == "keyword" and request.index_id is not None:
        raise HTTPException(422, "Index selector requires semantic or hybrid mode")
    try:
        requirements(request.question)
    except ValueError:
        raise HTTPException(422, "Question has too many required parts") from None
    metadata = []
    try:
        async with asyncio.timeout(settings.qa_timeout_seconds):
            initial = await capture(db, uid, repo, rev, request)
            chosen = request.generation_provider or settings.qa_provider
            if chosen == "gemini":
                if not settings.qa_gemini_enabled:
                    raise HTTPException(409, "Hosted Q&A is disabled")
                if request.generation_provider is None and not settings.qa_gemini_primary_accepted:
                    raise HTTPException(409, "Hosted primary provider is not accepted")
                if request.hosted_processing_consent != hosted.CONSENT:
                    raise HTTPException(
                        422,
                        "Explicit consent to send this question and repository excerpts "
                        "to Google is required",
                    )
            identity = hosted.identity if chosen == "gemini" else fingerprint
            provider_factory = (
                (gemini_factory or hosted.make_provider) if chosen == "gemini" else make_provider
            )
            policy = grounded_policy if chosen == "gemini" else None
            response_identity = (
                dict(
                    provider="gemini",
                    model="gemini-3.5-flash-lite",
                    policy_version="grounded-claims-v4",
                )
                if chosen == "gemini"
                else {}
            )
            try:
                generation_identity = identity()
            except GenerationError:
                return QuestionResponse(
                    repository_id=repo,
                    revision_id=rev,
                    commit_sha=initial.snapshot.sha,
                    question=request.question,
                    **response_identity,
                    status="provider_failure",
                    error_category="configuration",
                )
            await db.rollback()
            base = dict(
                query=request.question,
                mode=request.retrieval_mode,
                top_k=min(10, settings.keyword_search_max_top_k),
            )
            if request.retrieval_mode != "keyword":
                base.update(configuration="lmstudio", index_id=initial.index_id)
            services = {
                "keyword": keyword_retrieval,
                "semantic": semantic_retrieval,
                "hybrid": hybrid_retrieval,
            }
            retrieved = await services[request.retrieval_mode].search(db, uid, repo, rev, base)
            candidates = list(retrieved.matches)
            if investigation_chunk_ids:
                try:
                    ids = tuple(str(UUID(str(k))) for k in investigation_chunk_ids)
                except (ValueError, TypeError, AttributeError):
                    raise HTTPException(422, "Invalid investigation evidence references") from None
                if len(ids) > 5 or len(ids) != len(set(ids)):
                    raise HTTPException(422, "Invalid investigation evidence bound")
                if investigation_digest != initial.signature:
                    raise HTTPException(409, "Investigation evidence changed")
                canonical = {
                    str(chunk.id): (chunk, path) for chunk, path in initial.snapshot.chunks
                }
                if not set(ids) <= canonical.keys():
                    raise HTTPException(404, "Investigation evidence not found")
                carried = []
                for ident in ids:
                    chunk, path = canonical[ident]
                    carried.append(
                        CanonicalCandidate(
                            source_id=chunk.source_id,
                            chunk_id=chunk.id,
                            path=path,
                            commit_sha=initial.snapshot.sha,
                            line_start=chunk.line_start,
                            line_end=chunk.line_end,
                            excerpt=chunk.text,
                        )
                    )
                candidates = (
                    carried + [m for m in retrieved.matches if str(m.chunk_id) not in ids]
                )[:10]
            current = await capture(db, uid, repo, rev, request, initial.index_id)
            if current.signature != initial.signature:
                raise HTTPException(409, "Repository evidence changed")
            await db.rollback()
            selected, payload = select_evidence(
                candidates,
                request.question,
                settings.generation_max_input_bytes
                - len(json.dumps(GeneratedAnswer.model_json_schema()).encode())
                - 32
                - (max(0, len(policy.SYSTEM.encode()) - len(SYSTEM.encode())) if policy else 0),
            )

            def result(status, **extra):
                return QuestionResponse(
                    repository_id=repo,
                    revision_id=rev,
                    commit_sha=initial.snapshot.sha,
                    question=request.question,
                    status=status,
                    generation_metadata=list(metadata),
                    **response_identity,
                    **extra,
                )

            calls = 0

            async def revalidate():
                try:
                    same_generation = (
                        hosted.identity if chosen == "gemini" else fingerprint
                    )() == generation_identity
                except GenerationError:
                    same_generation = False
                if not same_generation:
                    raise HTTPException(409, "Generation configuration changed")
                current = await capture(db, uid, repo, rev, request, initial.index_id)
                if current.signature != initial.signature:
                    raise HTTPException(409, "Repository evidence changed")

            async def reserve():
                nonlocal calls
                if calls >= 2:
                    raise GenerationError("request_budget")
                await revalidate()
                await db.rollback()
                calls += 1

            await revalidate()
            await db.rollback()
            if investigation_chunk_ids and not set(ids) <= {
                str(m.chunk_id) for m in selected.values()
            }:
                return result(
                    "insufficient_evidence",
                    uncertainty="Canonical investigation evidence exceeds bounded context.",
                )
            if not selected:
                return result(
                    "insufficient_evidence", uncertainty="No safe usable evidence was retrieved."
                )
            try:
                generated = await provider_factory().generate(
                    [
                        {"role": "system", "content": policy.SYSTEM if policy else SYSTEM},
                        {"role": "user", "content": payload},
                    ],
                    reserve,
                    response_schema=GeneratedAnswer.model_json_schema(),
                )
                metadata.append(
                    {
                        "model": generated.model,
                        "protocol": generated.protocol,
                        "configuration_fingerprint": generation_identity,
                        "latency_seconds": generated.latency_seconds,
                        "usage": generated.usage,
                    }
                )
                await revalidate()
                await db.rollback()
                parsed, citations = (policy.validate_answer if policy else validate_answer)(
                    generated.content, selected, repo, rev
                )
                if parsed.status == "insufficient_evidence":
                    await revalidate()
                    return result("insufficient_evidence", uncertainty=parsed.uncertainty)
                checked = await provider_factory().generate(
                    [
                        {"role": "system", "content": policy.VERIFY if policy else VERIFY},
                        {
                            "role": "user",
                            "content": (policy.support_payload if policy else support_payload)(
                                request.question, parsed, selected
                            ),
                        },
                    ],
                    reserve,
                    response_schema=(policy.checker_schema if policy else checker_schema)(),
                )
                metadata.append(
                    {
                        "model": checked.model,
                        "protocol": checked.protocol,
                        "configuration_fingerprint": generation_identity,
                        "latency_seconds": checked.latency_seconds,
                        "usage": checked.usage,
                    }
                )
                await revalidate()
                (policy.validate_coverage if policy else validate_coverage)(
                    checked.content, request.question, parsed, selected
                )
                if policy is None:
                    # The same deterministic, fail-closed comparison-direction obligation the
                    # hosted policy applies; the local path previously relied on the model only.
                    grounded_policy.comparison_obligation(request.question, parsed, selected)
                return result(
                    "answered",
                    answer="\n".join(c.text for c in parsed.claims),
                    claims=parsed.claims,
                    citations=citations,
                    uncertainty=parsed.uncertainty,
                )
            except GenerationError as exc:
                await revalidate()
                return result("provider_failure", error_category=exc.code)
            except ValidationError:
                await revalidate()
                return result("validation_failure", error_category="invalid_schema")
            except (ValueError, KeyError, TypeError) as exc:
                await revalidate()
                categories = {
                    "Noncanonical quotation": "invalid_quote",
                    "Unknown evidence": "unknown_evidence",
                    "Unsupported literal": "unsupported_literal",
                    "Contradictory comparison": "contradictory_comparison",
                    "Contradictory literal": "contradictory_literal",
                    "Contradicted output": "contradicted_output",
                    "Insufficient support": "insufficient_support",
                    "Unsupported answer": "unsupported_answer",
                    "Unsupported output": "unsupported_output",
                    "Incomplete coverage": "incomplete_output",
                    "Unsafe output": "unsafe_output",
                    "Unsafe or unverified claim": "unsafe_claim",
                    "Invalid abstention": "invalid_abstention",
                }
                return result(
                    "validation_failure",
                    error_category=categories.get(str(exc), "invalid_or_unsupported_output"),
                )
    except EmbeddingError:
        raise HTTPException(409, "Compatible local index required") from None
    except (TimeoutError, DBAPIError):
        raise HTTPException(503, "Question answering temporarily unavailable") from None
    finally:
        await db.rollback()
