"""Replay-safe investigation journal on the existing durable Review queue.

Committed decisions/searches replay without inference. Uncommitted external outcomes
are consumed and terminate unknown; no automatic hosted resend.
"""

import asyncio
import json
import math
from datetime import timedelta
from uuid import uuid4

from fastapi import HTTPException
from pydantic import Field
from sqlalchemy import select

from api.schemas.search import KeywordSearchResponse
from core.config import settings
from core.contracts import Contract
from core.models import Investigation, Review
from core.services import job_queue as queue
from core.services import keyword_retrieval as retrieval
from core.services.investigation import Decision, InvestigationRequest, investigate
from rag.generator.local_provider import GenerationError


class State(Contract):
    version: int = Field(default=1, strict=True, ge=1, le=1)
    events: dict = Field(default_factory=dict, max_length=10)
    attempts: list[dict] = Field(default_factory=list, max_length=6)


def validate_state(value):
    if (
        not isinstance(value, dict)
        or type(value.get("version")) is not int
        or value["version"] != 1
    ):
        raise ValueError("Unsupported checkpoint version")
    state = State.model_validate(value)
    if len(json.dumps(value).encode()) > 131072:
        raise ValueError("Checkpoint limit exceeded")
    return state


async def submit(db, uid, repo, rev, request):
    if not settings.durable_investigation_enabled:
        raise HTTPException(409, "Durable investigations disabled")
    uid, repo, rev = map(str, (uid, repo, rev))
    request = InvestigationRequest.model_validate(request)
    if request.hosted_processing_consent != "gemini-source-processing-v1":
        raise HTTPException(422, "Explicit hosted consent required")
    from rag.generator.grounding import unsafe

    if not request.objective.strip() or unsafe(request.objective):
        raise HTTPException(422, "Unsafe objective")
    if len(json.dumps(request.model_dump(mode="json")).encode()) > 8192:
        raise HTTPException(413, "Investigation request too large")
    snapshot = await retrieval._snapshot(db, uid, repo, rev, retrieval.SearchLimits.configured())
    scope = await retrieval._scope(db, uid, repo, rev)
    stamp = queue.now()
    identity = str(uuid4())
    job = Review(
        id=identity,
        profile_id=scope.profile_id,
        revision_id=str(rev),
        job_type="investigation",
        status="pending",
        requested_repository=str(repo),
    )
    record = Investigation(
        id=identity,
        owner_id=str(uid),
        profile_id=scope.profile_id,
        repository_id=str(repo),
        revision_id=str(rev),
        request=request.model_dump(mode="json"),
        state=State().model_dump(),
        snapshot_digest=snapshot.digest,
        created_at=stamp,
        updated_at=stamp,
        deadline_at=stamp + timedelta(seconds=request.timeout_seconds),
    )
    db.add(job)
    await db.flush()
    db.add(record)
    try:
        await db.commit()
    except Exception:
        await db.rollback()
        raise
    return identity


async def owned(db, uid, repo, rev, identity, *, lock=False):
    uid, repo, rev = map(str, (uid, repo, rev))
    # Do not let a transferred profile grant the new owner access to the old objective.
    await retrieval._scope(db, uid, repo, rev)
    record = await db.scalar(
        select(Investigation).where(
            Investigation.id == str(identity),
            Investigation.owner_id == str(uid),
            Investigation.repository_id == str(repo),
            Investigation.revision_id == str(rev),
        )
    )
    if record is None:
        raise HTTPException(404, "Investigation not found")
    job = await queue.owned(db, identity, uid, lock=lock)
    return record, job


async def status(db, uid, repo, rev, identity):
    uid, repo, rev = map(str, (uid, repo, rev))
    record, job = await owned(db, uid, repo, rev, identity)
    if job.status == "complete" and record.final is not None:
        snapshot = await retrieval._snapshot(
            db, uid, repo, rev, retrieval.SearchLimits.configured()
        )
        if snapshot.digest != record.snapshot_digest:
            raise HTTPException(409, "Canonical evidence changed")
    return {
        "id": record.id,
        "status": job.status,
        "outcome": record.outcome or job.error_category,
        "repository_id": record.repository_id,
        "revision_id": record.revision_id,
        "deadline_at": record.deadline_at,
        "inference_spent": len(record.state["attempts"]),
        "checkpoint_count": len(record.state["events"]),
        "cancel_requested": job.cancel_requested_at is not None,
        "result": record.final if job.status == "complete" else None,
    }


async def cancel(db, uid, repo, rev, identity):
    await owned(db, uid, repo, rev, identity)
    await db.rollback()
    await queue.cancel(db, identity, uid)
    return await status(db, uid, repo, rev, identity)


class Journal:
    def __init__(self, attempt):
        self.attempt = attempt
        self.counters = {}
        self.phase = "answer"

    async def mutate(self, update):
        async with self.attempt.factory() as db:
            job = await db.scalar(
                select(Review).where(Review.id == self.attempt.job_id).with_for_update()
            )
            queue.assert_attempt(job, self.attempt)
            record = await db.get(Investigation, job.id, populate_existing=True)
            if record is None or record.owner_id != self.attempt.user_id or record.outcome:
                raise queue.LostAttemptError()
            # Plain authorization read avoids holding a second source snapshot session.
            from core.models import Profile, User

            active = await db.scalar(
                select(User.id)
                .join(Profile)
                .where(
                    Profile.id == record.profile_id,
                    Profile.user_id == record.owner_id,
                    User.id == record.owner_id,
                    User.is_active.is_(True),
                )
            )
            if not active:
                raise HTTPException(403, "Investigation authority revoked")
            if queue.aware(record.deadline_at) <= queue.now():
                raise TimeoutError()
            state = validate_state(record.state)
            value = update(state)
            record.state = validate_state(state.model_dump()).model_dump()
            record.updated_at = queue.now()
            await db.commit()
            return value

    def key(self, kind):
        index = self.counters.get(kind, 0)
        self.counters[kind] = index + 1
        return f"{kind}/{index}"

    async def cached(self, key):
        return await self.mutate(lambda state: state.events.get(key))

    async def save(self, key, value):
        def apply(state):
            if key in state.events:
                raise ValueError("Checkpoint already committed")
            state.events[key] = value

        await self.mutate(apply)

    async def retrieve(self, db, uid, repo, rev, request):
        key = self.key("search")
        saved = await self.cached(key)
        if saved is None:
            result = await retrieval.search(db, uid, repo, rev, request)
            data = result.model_dump(mode="json")
            for match in data["matches"]:
                match.pop("excerpt")
            await self.save(key, {"query": request["query"], "response": data})
            return result
        if saved["query"] != request["query"]:
            raise ValueError("Replay diverged")
        snapshot = await retrieval._snapshot(
            db, uid, repo, rev, retrieval.SearchLimits.configured()
        )
        chunks = {str(chunk.id): (chunk, path) for chunk, path in snapshot.chunks}
        data = json.loads(json.dumps(saved["response"]))
        for match in data["matches"]:
            chunk, path = chunks[match["chunk_id"]]
            if (
                match["source_id"],
                match["path"],
                match["commit_sha"],
                match["line_start"],
                match["line_end"],
            ) != (str(chunk.source_id), path, snapshot.sha, chunk.line_start, chunk.line_end):
                raise ValueError("Canonical replay mismatch")
            match["excerpt"] = chunk.text
        await db.rollback()
        return KeywordSearchResponse.model_validate(data)


class PrepaidBudget:
    stopped = False

    def __init__(self):
        self.row = None

    def reserve(self):
        if self.stopped or self.row is None:
            raise GenerationError("request_budget")
        row, self.row = self.row, None
        return row

    def save(self):
        pass  # Provider diagnostic/raw text stays in process; never historical ledgers.


class DurableProvider:
    def __init__(self, journal, provider):
        self.journal, self.provider = journal, provider
        self.deadline = None

    async def generate(self, messages, before_request, **kwargs):
        identity = str(uuid4())

        def reserve(state):
            if len(state.attempts) >= 6:
                raise GenerationError("request_budget")
            state.attempts.append(
                {"id": identity, "phase": self.journal.phase, "state": "reserved"}
            )

        reserved = False

        async def durable_before():
            nonlocal reserved
            await before_request()
            await self.journal.mutate(reserve)  # COMMIT immediately before POST.
            reserved = True
            self.provider.budget.row = {"number": identity, "state": "reserved_before_post"}

        self.provider.deadline = self.deadline
        try:
            result = await self.provider.generate(messages, durable_before, **kwargs)
        except GenerationError as exc:
            if reserved:
                await self.finish(
                    identity,
                    "outcome_unknown"
                    if exc.code in {"transport_or_timeout", "timeout"}
                    else "failed_known",
                )
            raise
        if not reserved:
            raise RuntimeError("Provider bypassed durable reservation")
        await self.finish(identity, "response_received")
        return result

    async def finish(self, identity, status):
        def apply(state):
            item = next(a for a in state.attempts if a["id"] == identity)
            item["state"] = status

        await self.journal.mutate(apply)


class DurablePlanner:
    def __init__(self, journal, planner):
        self.journal, self.planner = journal, planner
        self.provider = planner.provider

    def set_readiness(self, value):
        self.planner.set_readiness(value)

    async def __call__(self, objective, steps):
        key = self.journal.key("plan")
        saved = await self.journal.cached(key)
        if saved is not None:
            return Decision.model_validate(saved)
        self.journal.phase = key
        decision = Decision.model_validate(await self.planner(objective, steps))
        await self.journal.save(key, decision.model_dump(mode="json"))
        self.journal.phase = "answer"
        return decision


def planner_factory(journal, deadline):
    # Reuse the existing planner prompt/decoding; bypass only its historical SQLite
    # ledger constructor, replacing accounting with this execution's PG journal.
    from rag.generator import hosted
    from rag.generator.gemini_provider import Gemini
    from rag.generator.investigation_planner import Planner

    hosted.identity()
    planner = object.__new__(Planner)
    key = hosted.credential()
    planner.provider = DurableProvider(journal, Gemini(key, PrepaidBudget()))
    planner.planning_provider = DurableProvider(
        journal,
        Gemini(key, PrepaidBudget(), thinking_level=settings.investigation_planner_thinking),
    )
    planner.deadline = deadline
    return DurablePlanner(journal, planner)


async def process(db, job_id, uid, *, attempt, make_planner=planner_factory):
    if not settings.durable_investigation_enabled:
        raise HTTPException(409, "Durable investigations disabled")
    record = await db.get(Investigation, job_id)
    if record is None or record.owner_id != uid:
        raise HTTPException(403, "Investigation authority revoked")
    request = InvestigationRequest.model_validate(record.request)
    state = validate_state(record.state)
    repo, rev, digest = record.repository_id, record.revision_id, record.snapshot_digest
    deadline_at = queue.aware(record.deadline_at)
    # These scalar fields survive rollback/ORM expiry.
    await db.rollback()
    journal = Journal(attempt)
    journal.digest = digest
    outcome, final = None, None
    try:
        snapshot = await retrieval._snapshot(
            db, uid, repo, rev, retrieval.SearchLimits.configured()
        )
        if snapshot.digest != digest:
            raise HTTPException(409, "Canonical evidence changed")
        await db.rollback()
        remaining = (deadline_at - queue.now()).total_seconds()
        if remaining <= 0:
            raise TimeoutError()
        # A committed decision proves the planner response was consumed safely.
        # Any answer attempt without atomic terminal commit remains ambiguous.
        if any(a["phase"] not in state.events for a in state.attempts):
            outcome = "unknown_provider_outcome"
        else:
            async with asyncio.timeout(remaining):
                deadline = asyncio.get_running_loop().time() + remaining
                planner = make_planner(journal, deadline)
                request.timeout_seconds = min(request.timeout_seconds, max(1, math.ceil(remaining)))
                result = await investigate(
                    db, uid, repo, rev, request, planner=planner, journal=journal
                )
                from uuid import UUID

                result.execution_id = UUID(job_id)
                outcome, final = result.status, result.model_dump(mode="json")
    except TimeoutError:
        outcome = "timeout"
    except HTTPException as exc:
        if exc.status_code not in {401, 403, 404, 409}:
            raise
        outcome = "scope_changed"
    await db.rollback()
    # Finalization and cancellation serialize on the existing fenced queue row.
    if final is not None:
        snapshot = await retrieval._snapshot(
            db, uid, repo, rev, retrieval.SearchLimits.configured()
        )
        if snapshot.digest != digest:
            outcome, final = "scope_changed", None
    job = await db.scalar(select(Review).where(Review.id == job_id).with_for_update())
    try:
        queue.assert_attempt(job, attempt)
    except queue.CancellationRequestedError:
        queue.schedule_failure(job, "cancelled", False, "Cancelled by owner")
        await db.commit()
        return
    record = await db.get(Investigation, job_id, populate_existing=True)
    if queue.aware(record.deadline_at) <= queue.now():
        outcome, final = "timeout", None
    record.outcome, record.final, record.updated_at = outcome, final, queue.now()
    queue.complete(job, attempt)
    await db.commit()
