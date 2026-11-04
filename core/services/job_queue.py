"""PostgreSQL queue, fencing and bounded retry policy on canonical Review rows."""

import threading
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import structlog
from fastapi import HTTPException
from sqlalchemy import and_, or_, select, exists
from sqlalchemy.exc import DBAPIError
from sqlalchemy.exc import TimeoutError as DatabaseTimeoutError
from sqlalchemy.ext.asyncio import async_sessionmaker

from core.config import settings
from core.models import Profile, Review, User, RetrievalIndex

log = structlog.get_logger(__name__)
COUNTERS = Counter()
TERMINAL = {"complete", "failed", "cancelled"}
TRANSIENT_GITHUB = {"network", "timeout", "rate_limited", "upstream_unavailable"}
MESSAGES = {
    "upstream": "GitHub temporarily unavailable; retry scheduled if attempts remain",
    "database": "Database connection interrupted; retry scheduled if attempts remain",
    "worker_lost": "Worker lease expired; retry scheduled if attempts remain",
    "worker_shutdown": "Worker stopped; retry scheduled if attempts remain",
    "authorization": "Job authorization is no longer valid",
    "validation": "Job input or source integrity validation failed",
    "internal": "Job failed; no partial results were saved",
}


def now():
    return datetime.now(UTC)


def aware(value):
    return value.replace(tzinfo=UTC) if value and value.tzinfo is None else value


def supported():
    return or_(
        Review.job_type == "github_ingestion",
        Review.job_type == "investigation",
        and_(
            Review.job_type == "retrieval_index",
            exists(
                select(RetrievalIndex.id).where(
                    RetrievalIndex.job_id == Review.id,
                    RetrievalIndex.profile_id == Review.profile_id,
                    RetrievalIndex.revision_id == Review.revision_id,
                )
            ),
        ),
        and_(
            Review.job_type == "review",
            Review.report_kind == "evidence",
            Review.analysis_result.is_not(None),
        ),
    )


class LostAttemptError(Exception):
    """Lease/token no longer belongs to this execution."""


class CancellationRequestedError(Exception):
    """Persisted cancellation observed at a safe checkpoint."""


@dataclass
class Attempt:
    job_id: str
    user_id: str
    worker_id: str
    token: str
    number: int
    job_type: str
    factory: object = field(repr=False)
    stopped: threading.Event = field(default_factory=threading.Event, repr=False)

    def check_sync(self):
        if self.stopped.is_set():
            raise LostAttemptError()

    async def checkpoint(self, stage=None, percent=None):
        self.check_sync()
        async with self.factory() as db:
            job = await db.scalar(select(Review).where(Review.id == self.job_id).with_for_update())
            assert_attempt(job, self)
            owner = await db.scalar(
                select(User.id)
                .join(Profile)
                .where(
                    Profile.id == job.profile_id,
                    Profile.user_id == self.user_id,
                    User.id == self.user_id,
                    User.is_active.is_(True),
                )
            )
            if owner is None:
                raise HTTPException(403, "Job authorization revoked")
            stamp = now()
            job.heartbeat_at = stamp
            job.lease_expires_at = stamp + timedelta(seconds=settings.job_lease_seconds)
            if stage is not None:
                job.progress_stage, job.progress_pct = stage, percent
            await db.commit()
            if stage is not None:
                event("progress", job)


def assert_attempt(job, attempt):
    if (
        not job
        or job.status != "processing"
        or job.execution_token != attempt.token
        or job.worker_id != attempt.worker_id
        or not job.lease_expires_at
        or aware(job.lease_expires_at) <= now()
    ):
        raise LostAttemptError()
    if job.cancel_requested_at:
        raise CancellationRequestedError()


def event(name, job, **extra):
    stamp = now()
    COUNTERS[name] += 1
    log.info(
        "job_" + name,
        counter=COUNTERS[name],
        job_id=job.id,
        job_type=job.job_type,
        attempt=job.attempt_count,
        worker_id=job.worker_id,
        execution_generation=job.attempt_count,
        stage=job.progress_stage,
        status=job.status,
        error_category=job.error_category,
        queue_seconds=max(
            0, (aware(job.started_at or stamp) - aware(job.created_at)).total_seconds()
        ),
        duration_seconds=max(0, (stamp - aware(job.started_at or stamp)).total_seconds()),
        **extra,
    )


def clear_owner(job):
    job.worker_id = job.execution_token = None
    job.lease_expires_at = None


def complete(job, attempt):
    assert_attempt(job, attempt)
    job.status = "complete"
    job.progress_stage, job.progress_pct = "complete", 100
    job.completed_at = job.updated_at = now()
    job.retry_at = None
    job.error_category = job.error_message = None
    # Retain last token/worker for diagnostics; status fences all subsequent writes.


def classify(exc):
    from ingestion.github import GitHubError
    from ingestion.embeddings.approved import EmbeddingError

    if isinstance(exc, EmbeddingError):
        return ("embedding", exc.retryable, str(exc))
    if isinstance(exc, GitHubError):
        return ("upstream", exc.code in TRANSIENT_GITHUB, exc.message)
    if isinstance(exc, DatabaseTimeoutError):
        return ("database", True, MESSAGES["database"])
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return ("upstream", True, MESSAGES["upstream"])
    if isinstance(exc, DBAPIError):
        retry = exc.connection_invalidated or getattr(exc.orig, "sqlstate", "") in {
            "08000",
            "08003",
            "08006",
            "40001",
            "40P01",
            "57P01",
            "57P03",
            "53300",
        }
        return (
            "database" if retry else "validation",
            retry,
            MESSAGES["database" if retry else "validation"],
        )
    if isinstance(exc, HTTPException):
        return (
            "authorization" if exc.status_code in {401, 403, 404} else "validation",
            False,
            MESSAGES["authorization" if exc.status_code in {401, 403, 404} else "validation"],
        )
    if isinstance(exc, (ValueError, KeyError)):
        return ("validation", False, MESSAGES["validation"])
    return ("internal", False, MESSAGES["internal"])


def schedule_failure(job, category, retry, message):
    if job.status in TERMINAL:
        raise LostAttemptError()
    if job.cancel_requested_at:
        category, message = "cancelled", "Cancelled by owner"
    stamp = now()
    job.error_category, job.error_message = category, message
    job.updated_at = stamp
    if job.cancel_requested_at:
        job.status, job.progress_stage = "cancelled", "cancelled"
        job.completed_at = stamp
        job.retry_at = None
    elif retry and job.attempt_count < job.max_attempts:
        job.status, job.progress_stage = "pending", "retry_waiting"
        delay = min(
            settings.job_retry_max_seconds,
            settings.job_retry_base_seconds * 2 ** max(0, job.attempt_count - 1),
        )
        job.retry_at = stamp + timedelta(seconds=delay)
        job.completed_at = None
    else:
        job.status, job.progress_stage = "failed", "failed"
        job.completed_at = stamp
        job.retry_at = None
    job.progress_pct = None
    clear_owner(job)


async def recover(db):
    rows = (
        await db.scalars(
            select(Review)
            .where(
                supported(),
                Review.status == "processing",
                or_(Review.lease_expires_at.is_(None), Review.lease_expires_at <= now()),
            )
            .order_by(Review.created_at)
            .with_for_update(skip_locked=True)
            .limit(50)
        )
    ).all()
    for job in rows:
        schedule_failure(job, "worker_lost", True, MESSAGES["worker_lost"])
        await sync_index(db, job)
    await db.commit()
    for job in rows:
        event("recovered", job)
        event("retried" if job.status == "pending" else job.status, job)
    return len(rows)


async def claim(db, worker_id, job_id=None, user_id=None):
    query = (
        select(Review, Profile.user_id)
        .join(Profile)
        .where(
            supported(),
            Review.status == "pending",
            Review.attempt_count < Review.max_attempts,
            or_(Review.retry_at.is_(None), Review.retry_at <= now()),
            Review.cancel_requested_at.is_(None),
        )
    )
    if job_id is not None:
        query = query.where(Review.id == str(job_id))
    if user_id is not None:
        query = query.where(Profile.user_id == str(user_id))
    row = (
        await db.execute(
            query.order_by(Review.created_at, Review.id)
            .with_for_update(of=Review, skip_locked=True)
            .limit(1)
        )
    ).first()
    if row is None:
        await db.rollback()
        return None
    job, owner = row
    stamp = now()
    job.status, job.progress_stage, job.progress_pct = "processing", "claimed", None
    job.attempt_count += 1
    job.worker_id, job.execution_token = worker_id, str(uuid4())
    job.started_at = job.heartbeat_at = job.updated_at = stamp
    job.lease_expires_at = stamp + timedelta(seconds=settings.job_lease_seconds)
    job.retry_at = job.completed_at = None
    job.error_category = job.error_message = None
    attempt = Attempt(
        job.id,
        owner,
        worker_id,
        job.execution_token,
        job.attempt_count,
        job.job_type,
        async_sessionmaker(db.bind, expire_on_commit=False),
    )
    await sync_index(db, job)
    await db.commit()
    event("claimed", job)
    return attempt


async def fail(factory, attempt, exc=None, category=None):
    attempt.stopped.set()
    if isinstance(exc, LostAttemptError):
        return
    async with factory() as db:
        job = await db.scalar(select(Review).where(Review.id == attempt.job_id).with_for_update())
        if not job or job.status != "processing" or job.execution_token != attempt.token:
            return
        if aware(job.lease_expires_at) <= now():
            return  # automatic recovery owns expired attempts
        if isinstance(exc, CancellationRequestedError):
            job.cancel_requested_at = job.cancel_requested_at or now()
        info = (category, True, MESSAGES[category]) if category else classify(exc)
        if job.job_type == "github_ingestion":
            from ingestion.github import GitHubError

            code = (
                exc.code
                if isinstance(exc, GitHubError)
                else ("timeout" if isinstance(exc, TimeoutError) else "ingestion_failed")
            )
            job.ingestion_result = {"error_code": code}
        schedule_failure(job, *info)
        await sync_index(db, job)
        await db.commit()
        event("retried" if job.status == "pending" else job.status, job)


async def direct(db, job_id, user_id, execute):
    # Compatibility for explicit service callers/tests; API never calls this path.
    from core.services.github_ingestion import _profile

    job = await db.get(Review, str(job_id))
    if job is None:
        raise HTTPException(404, "Job not found")
    await _profile(db, job.profile_id, user_id)
    await db.rollback()
    attempt = await claim(db, "direct-" + uuid4().hex, job_id, user_id)
    if attempt is None:
        return
    try:
        await execute(attempt)
    except Exception as exc:
        await db.rollback()
        await fail(attempt.factory, attempt, exc)


async def owned(db, job_id, user_id, lock=False):
    from core.services.authorization import require_active_user
    from core.services.github_ingestion import _profile

    await require_active_user(db, user_id)
    job = await db.scalar(
        select(Review)
        .join(Profile)
        .where(Review.id == str(job_id), Profile.user_id == str(user_id), supported())
    )
    if job is None:
        raise HTTPException(404, "Job not found")
    if job.job_type == "investigation":
        from core.models import Investigation
        original = await db.scalar(select(Investigation.id).where(
            Investigation.id == job.id, Investigation.owner_id == str(user_id)))
        if original is None:
            raise HTTPException(404, "Job not found")
    if lock:
        await _profile(db, job.profile_id, user_id, lock=True)
        job = await db.scalar(
            select(Review)
            .where(Review.id == job.id)
            .with_for_update()
            .execution_options(populate_existing=True)
        )
    return job


async def cancel(db, job_id, user_id):
    job = await owned(db, job_id, user_id, lock=True)
    if job.status in {"complete", "failed"}:
        raise HTTPException(409, "Terminal job cannot be cancelled")
    if job.status == "cancelled":
        return job
    job.cancel_requested_at = now()
    job.updated_at = now()
    if job.status == "pending":
        schedule_failure(job, "cancelled", False, "Cancelled by owner")
        await sync_index(db, job)
    await db.commit()
    await db.refresh(job)
    event("cancelled" if job.status == "cancelled" else "cancellation_requested", job)
    return job


async def sync_index(db, job):
    if job.job_type == "retrieval_index":
        from core.services.retrieval_indexing import sync_job_state

        await sync_job_state(db, job)
