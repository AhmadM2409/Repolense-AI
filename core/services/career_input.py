"""J1 caller-driven sequential preparation using the unchanged ingestion queue."""

from datetime import UTC, datetime
from functools import wraps

from fastapi import HTTPException
from sqlalchemy import delete, select

from api.schemas.career import InputCreate, Selection, content_hash
from core.config import settings
from core.models import CareerInput, CareerSelection, Review
from core.services import github_ingestion as ingestion
from core.services import repository_analysis
from ingestion.github_profile import Discovery, profile_account


def rollback_on_error(fn):
    @wraps(fn)
    async def wrapped(db, *args, **kwargs):
        try:
            return await fn(db, *args, **kwargs)
        except Exception:
            await db.rollback()
            raise

    return wrapped


def make_discovery():
    return Discovery(settings.github_token, pages=settings.career_discovery_pages)


async def owned(db, uid, identity, *, lock=False):
    record = await db.get(CareerInput, str(identity))
    if record is None or record.owner_id != str(uid):
        raise HTTPException(404, "Career input unavailable")
    await ingestion._profile(db, record.profile_id, uid, lock=lock)
    if lock:
        record = await db.scalar(
            select(CareerInput)
            .where(CareerInput.id == str(identity))
            .with_for_update()
            .execution_options(populate_existing=True)
        )
    if content_hash(record.job_description) != record.job_hash:
        raise HTTPException(409, "Career input integrity changed")
    return record


@rollback_on_error
async def create(db, uid, data):
    data = InputCreate.model_validate(data)
    await ingestion._profile(db, data.profile_id, uid)
    await db.rollback()  # no locks held during metadata HTTP
    account = profile_account(data.github_profile_url)
    discovery = await make_discovery().discover(account)
    await ingestion._profile(db, data.profile_id, uid, lock=True)
    record = CareerInput(
        owner_id=str(uid),
        profile_id=str(data.profile_id),
        account=account,
        job_description=data.job_description,
        job_hash=content_hash(data.job_description),
        discovery=discovery,
    )
    db.add(record)
    await db.commit()
    return await detail(db, uid, record.id)


async def selections(db, identity):
    return list(
        (
            await db.scalars(
                select(CareerSelection)
                .where(CareerSelection.input_id == str(identity))
                .order_by(CareerSelection.position)
            )
        ).all()
    )


@rollback_on_error
async def select_repositories(db, uid, identity, data):
    data = Selection.model_validate(data)
    record = await owned(db, uid, identity, lock=True)
    names = data.repositories
    if len(names) > settings.career_max_selected_repositories or len(names) != len(set(names)):
        raise HTTPException(422, "Invalid selection count or duplicates")
    allowed = {r["full_name"] for r in record.discovery["items"] if r["eligible"]}
    if not set(names) <= allowed:
        raise HTTPException(422, "Selection must use eligible discovered identities")
    current = await selections(db, identity)
    if record.started_at:
        if names == [r.full_name for r in current]:
            return await detail(db, uid, identity)
        raise HTTPException(409, "Selection frozen after preparation starts")
    await db.execute(delete(CareerSelection).where(CareerSelection.input_id == str(identity)))
    for i, name in enumerate(names):
        db.add(
            CareerSelection(
                input_id=record.id, profile_id=record.profile_id, position=i, full_name=name
            )
        )
    record.updated_at = datetime.now(UTC)
    await db.commit()
    return await detail(db, uid, identity)


@rollback_on_error
async def prepare(db, uid, identity):
    record = await owned(db, uid, identity, lock=True)
    rows = await selections(db, identity)
    if not rows:
        raise HTTPException(409, "Explicit repository selection required")
    if record.started_at is None:
        record.started_at = record.updated_at = datetime.now(UTC)
        await db.commit()  # freeze before calling existing service, which owns its commit
    for entry in rows:
        await owned(db, uid, identity, lock=True)
        if entry.revision_id:
            _, checked, _, _, _ = await repository_analysis.verified_input(db, entry.job_id, uid)
            if str(checked.id) != entry.revision_id or checked.commit_sha != entry.commit_sha:
                raise HTTPException(409, "Pinned revision integrity changed")
            continue
        if not entry.job_id:
            # Recover an interrupted association without submitting duplicate work.
            job = await db.scalar(
                select(Review)
                .where(
                    Review.profile_id == record.profile_id,
                    Review.job_type == "github_ingestion",
                    Review.requested_repository == entry.full_name,
                    Review.status.in_(["pending", "processing", "complete"]),
                )
                .order_by(Review.created_at.desc())
            )
            if job is None:
                job, _ = await ingestion.submit(db, record.profile_id, uid, entry.full_name)
            await owned(db, uid, identity, lock=True)
            entry.job_id = job.id
            await db.commit()
        job = await ingestion.get_ingestion(db, entry.job_id, uid)
        if job.requested_repository != entry.full_name:
            raise HTTPException(409, "Ingestion identity changed")
        if job.status == "complete":
            _, revision, repository, _, _ = await repository_analysis.verified_input(
                db, job.id, uid
            )
            if repository.full_name != entry.full_name:
                raise HTTPException(409, "Verified repository identity changed")
            entry.revision_id, entry.commit_sha = revision.id, revision.commit_sha
            record.updated_at = datetime.now(UTC)
            await db.commit()
            continue
        if job.status in {"pending", "processing"}:
            break
        # Terminal failed/cancelled selection stays explicit; continue the next on this POST.
    return await detail(db, uid, identity)


async def detail(db, uid, identity):
    record = await owned(db, uid, identity)
    result = []
    for entry in await selections(db, identity):
        state, error = "not_started", None
        repository_id = None
        if entry.job_id:
            job = await ingestion.get_ingestion(db, entry.job_id, uid)
            state, error = job.status, job.error_category
            if entry.revision_id:
                _, revision, repository, _, _ = await repository_analysis.verified_input(
                    db, job.id, uid
                )
                if (revision.id, revision.commit_sha, repository.full_name) != (
                    entry.revision_id,
                    entry.commit_sha,
                    entry.full_name,
                ):
                    raise HTTPException(409, "Pinned revision integrity changed")
                repository_id = repository.id
                state = "ready"
            elif state == "complete":
                state = "awaiting_pin"  # GET does not mutate the manifest or enqueue work
        result.append(
            dict(
                full_name=entry.full_name,
                position=entry.position,
                job_id=entry.job_id,
                status=state,
                error_category=error,
                repository_id=repository_id,
                revision_id=entry.revision_id,
                commit_sha=entry.commit_sha,
            )
        )
    states = [r["status"] for r in result]
    overall = (
        "input_created"
        if not states
        else "selection_saved"
        if not record.started_at
        else "ready"
        if all(s == "ready" for s in states)
        else "preparing"
        if any(s in {"pending", "processing", "not_started", "awaiting_pin"} for s in states)
        else "partial"
        if "ready" in states
        else "failed"
    )
    return dict(
        id=record.id,
        profile_id=record.profile_id,
        github_profile_url="https://github.com/" + record.account,
        job_description=record.job_description,
        job_hash=record.job_hash,
        manifest_version=1,
        discovery=record.discovery,
        selection=result,
        status=overall,
        created_at=record.created_at,
        updated_at=record.updated_at,
        future_hosted_consent="not_granted",
    )


async def listing(db, uid, limit, offset):
    from core.models import Profile
    from core.services.authorization import require_active_user

    await require_active_user(db, uid)
    rows = (
        await db.scalars(
            select(CareerInput)
            .join(Profile, Profile.id == CareerInput.profile_id)
            .where(CareerInput.owner_id == str(uid), Profile.user_id == str(uid))
            .order_by(CareerInput.created_at.desc(), CareerInput.id)
            .offset(offset)
            .limit(limit + 1)
        )
    ).all()
    return {
        "items": [
            dict(
                id=r.id,
                profile_id=r.profile_id,
                account=r.account,
                created_at=r.created_at,
                started=r.started_at is not None,
            )
            for r in rows[:limit]
        ],
        "next_offset": offset + limit if len(rows) > limit else None,
    }
