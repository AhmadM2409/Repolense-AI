"""Bounded investigation state machine; optional canonical replay journal."""

import asyncio
from collections.abc import Awaitable, Callable
from typing import Literal
from uuid import UUID, uuid4

from fastapi import HTTPException
from pydantic import Field

from api.schemas.questions import QuestionRequest, QuestionResponse
from core.contracts import Contract
from core.services import keyword_retrieval, repository_qa
from rag.generator.grounding import unsafe
from rag.generator.local_provider import GenerationError


class InvestigationRequest(Contract):
    objective: str = Field(strict=True, min_length=1, max_length=1024)
    hosted_processing_consent: Literal["gemini-source-processing-v1"] | None = None
    max_steps: int = Field(default=3, strict=True, ge=1, le=4)
    timeout_seconds: int = Field(default=90, strict=True, ge=1, le=120)


class Observation(Contract):
    source_id: UUID
    chunk_id: UUID
    commit_sha: str
    path: str
    line_start: int
    line_end: int
    excerpt: str


class Decision(Contract):
    operation: Literal["search", "answer", "stop"]
    query: str = Field(default="", strict=True, max_length=256)
    reason: str = Field(default="", strict=True, max_length=160)


class Step(Contract):
    number: int
    operation: Literal["search", "answer", "stop"]
    origin: Literal["planner", "readiness"] = "planner"
    evidence_paths: list[str] = Field(default_factory=list)
    observations: list[Observation] = Field(default_factory=list)


class InvestigationResult(Contract):
    execution_id: UUID = Field(default_factory=uuid4)
    user_id: UUID
    repository_id: UUID
    revision_id: UUID
    status: Literal[
        "completed",
        "insufficient_evidence",
        "step_limit",
        "provider_failure",
        "validation_failure",
        "invalid_decision",
        "cancelled",
        "timeout",
        "scope_changed",
    ]
    steps: list[Step]
    final: QuestionResponse | None = None
    timings: list[dict] = Field(default_factory=list)


# Internal injected decision boundary: never serialized as an HTTP permission.
Planner = Callable[[str, tuple[Step, ...]], Awaitable[Decision]]


async def investigate(
    db,
    user_id,
    repository_id,
    revision_id,
    request,
    *,
    planner: Planner | None = None,
    cancelled=None,
    journal=None,
):
    """Offline-capable orchestration. No default hosted planner or new inference allowance.

    Planner inputs contain only objective and server-created path observations. Final
    claims can originate only from the existing grounded Q&A validator. Callers must
    separately authorize planner inference. Final generation uses the existing
    consent-protected Q&A service. No planner transport is installed.
    """
    retrieve = journal.retrieve if journal is not None else keyword_retrieval.search
    request = InvestigationRequest.model_validate(request)
    if not request.objective.strip() or unsafe(request.objective):
        raise HTTPException(422, "Investigation objective unavailable or unsafe")
    if db.new or db.dirty or db.deleted:
        raise RuntimeError("Investigation requires a clean session")
    repo, rev, uid = map(lambda x: str(UUID(str(x))), (repository_id, revision_id, user_id))
    limits = keyword_retrieval.SearchLimits.configured()
    steps = []
    final = None
    signature = None
    seen = set()
    timings = []
    observed = set()
    evidence = {}
    deadline = asyncio.get_running_loop().time() + request.timeout_seconds

    async def measured(stage, operation):
        started = asyncio.get_running_loop().time()
        try:
            return await operation
        finally:
            timings.append({"stage": stage, "seconds": asyncio.get_running_loop().time() - started})

    async def scope():
        nonlocal signature
        await keyword_retrieval._scope(db, uid, repo, rev)
        snapshot = await keyword_retrieval._snapshot(db, uid, repo, rev, limits)
        if journal is not None and getattr(journal, "digest", snapshot.digest) != snapshot.digest:
            raise HTTPException(409, "Durable evidence changed")
        if signature is not None and signature != snapshot.digest:
            raise HTTPException(409, "Investigation evidence changed")
        signature = snapshot.digest
        await db.rollback()

    def result(status):
        return InvestigationResult(
            user_id=uid,
            repository_id=repo,
            revision_id=rev,
            status=status,
            steps=steps,
            timings=timings + getattr(planner, "timings", []),
            final=final if status == "completed" else None,
        )

    try:
        async with asyncio.timeout(request.timeout_seconds):
            await measured("initial_scope", scope())
            if planner is None:
                from core.config import settings
                from rag.generator.investigation_planner import Planner as GeminiPlanner

                if not settings.investigation_enabled:
                    raise HTTPException(409, "Investigations are disabled")
                if request.hosted_processing_consent is None:
                    raise HTTPException(422, "Explicit hosted processing consent required")
                planner = GeminiPlanner()
                planner.before_request = scope
                planner.deadline = deadline
            for number in range(1, request.max_steps + 1):
                if cancelled is not None and cancelled.is_set():
                    return result("cancelled")
                await measured("scope_revalidation", scope())
                if hasattr(planner, "set_readiness"):
                    from core.services.evidence_readiness import assess

                    planner.set_readiness(assess(request.objective, tuple(evidence.values())))
                try:
                    decision = Decision.model_validate(
                        await measured(
                            "planner",
                            planner(
                                request.objective, tuple(s.model_copy(deep=True) for s in steps)
                            ),
                        )
                    )
                except (ValueError, TypeError):
                    return result("invalid_decision")
                await measured("scope_revalidation", scope())
                if cancelled is not None and cancelled.is_set():
                    return result("cancelled")
                readiness_search = False
                if decision.operation == "answer":
                    from core.services.evidence_readiness import assess

                    if not evidence:
                        initial_matches = await measured(
                            "readiness_lookup",
                            retrieve(
                                db,
                                uid,
                                repo,
                                rev,
                                dict(query=request.objective, mode="keyword", top_k=5),
                            ),
                        )
                        await scope()
                        evidence.update((str(m.chunk_id), m) for m in initial_matches.matches[:5])
                    ready = assess(request.objective, tuple(evidence.values()))
                    if not ready.ready:
                        if not ready.missing:
                            return result("insufficient_evidence")
                        followup = " ".join(ready.missing)[:256]
                        if followup in seen:
                            return result("insufficient_evidence")
                        readiness_search = True
                        decision = Decision(
                            operation="search",
                            query=followup,
                            reason="Server requires missing canonical definitions",
                        )
                if decision.operation == "stop":
                    steps.append(Step(number=number, operation="stop"))
                    return result("insufficient_evidence")
                if decision.operation == "search":
                    query = decision.query.strip()
                    if not query or unsafe(query) or query in seen:
                        return result("invalid_decision")
                    seen.add(query)
                    found = await measured(
                        "search",
                        retrieve(
                            db, uid, repo, rev, dict(query=query, mode="keyword", top_k=5)
                        ),
                    )
                    await measured("scope_revalidation", scope())
                    steps.append(
                        Step(
                            number=number,
                            operation="search",
                            origin="readiness" if readiness_search else "planner",
                            evidence_paths=[m.path for m in found.matches[:2]],
                            observations=[
                                Observation(
                                    source_id=m.source_id,
                                    chunk_id=m.chunk_id,
                                    commit_sha=m.commit_sha,
                                    path=m.path,
                                    line_start=m.line_start,
                                    line_end=m.line_end,
                                    excerpt=m.excerpt[:200],
                                )
                                for m in found.matches[:2]
                                if str(m.chunk_id) not in observed
                            ],
                        )
                    )
                    evidence.update(
                        (str(m.chunk_id), m)
                        for m in found.matches
                        if len(evidence) < 5 or str(m.chunk_id) in evidence
                    )
                    observed.update(str(m.chunk_id) for m in found.matches[:2])
                    if not found.matches:
                        return result("insufficient_evidence")
                    continue
                if cancelled is not None and cancelled.is_set():
                    return result("cancelled")
                # A planner cannot replace the user's objective or publish its own claim.
                if decision.query or request.hosted_processing_consent is None:
                    return result("invalid_decision")
                final = QuestionResponse.model_validate(
                    await measured(
                        "grounded_qa",
                        repository_qa.answer(
                            db,
                            uid,
                            repo,
                            rev,
                            QuestionRequest(
                                question=request.objective,
                                retrieval_mode="keyword",
                                generation_provider="gemini",
                                hosted_processing_consent=request.hosted_processing_consent,
                            ),
                            investigation_chunk_ids=tuple(evidence),
                            investigation_digest=signature,
                            gemini_factory=(lambda: planner.provider)
                            if hasattr(planner, "provider")
                            else None,
                        ),
                    )
                )
                await measured("scope_revalidation", scope())
                if (str(final.repository_id), str(final.revision_id)) != (repo, rev):
                    return result("scope_changed")
                steps.append(Step(number=number, operation="answer"))
                if cancelled is not None and cancelled.is_set():
                    return result("cancelled")
                if final.status == "answered":
                    return result("completed")
                if final.status == "insufficient_evidence":
                    return result("insufficient_evidence")
                return result(
                    "validation_failure"
                    if final.status == "validation_failure"
                    else "provider_failure"
                )
            return result("step_limit")
    except GenerationError:
        return result("provider_failure")
    except HTTPException as exc:
        if exc.status_code == 409:
            return result("scope_changed")
        raise
    except TimeoutError:
        return result("timeout")
    finally:
        await db.rollback()
