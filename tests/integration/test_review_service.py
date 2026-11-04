"""Real persisted service tests replacing the inherited mock-only review suite."""

from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest
from fastapi import HTTPException
from sqlalchemy import func, select, update

from api.schemas.profile import ProfileCreate, ProfileUpdate
from core.models import Profile, Review, User
from core.services import profile_service as profiles
from core.services import review_service as reviews
from tests.analysis_fixtures import verified_ingestion

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


async def test_create_persists_pending_review_and_uuid_arguments(database, actors):
    user = actors["users"][0]
    profile = actors["profiles"][0]
    async with database() as db:
        source=await verified_ingestion(db,profile.id,user.id)
        result = await reviews.create_review(db, UUID(profile.id), UUID(user.id),source.revision_id)
        assert result.status == "pending"
        assert result.sections is None and result.overall_score is None
        assert result.id and result.profile_id == profile.id
        review_id = result.id
    async with database() as db:
        persisted = await db.get(Review, review_id)
        assert persisted is not None and persisted.status == "pending"


@pytest.mark.parametrize("target", ["foreign", "missing"])
async def test_direct_create_cannot_bypass_owner_guard(database, actors, target):
    uid = actors["users"][0].id
    pid = actors["profiles"][1].id if target == "foreign" else str(uuid4())
    async with database() as db:
        before = await db.scalar(select(func.count()).select_from(Review))
        with pytest.raises(HTTPException) as denied:
            await reviews.create_review(db, pid, uid)
        assert denied.value.status_code == 404
        assert await db.scalar(select(func.count()).select_from(Review)) == before


async def test_review_reads_lists_counts_order_and_pagination(database, actors):
    uid = actors["users"][0].id
    pid = actors["profiles"][0].id
    async with database() as db:
        assert (await reviews.get_review(db, actors["reviews"][0].id, uid)).id
        assert await reviews.get_review(db, actors["reviews"][1].id, uid) is None
        assert await reviews.get_review(db, str(uuid4()), uid) is None
        for index in range(3):
            db.add(Review(
                profile_id=pid, status="pending",
                created_at=datetime.now(UTC) + timedelta(days=index + 1),
            ))
        await db.commit()
        all_rows, count = await reviews.list_reviews(db, uid)
        assert count == len(all_rows) == 4
        assert all(row.profile_id == pid for row in all_rows)
        assert [row.created_at for row in all_rows] == sorted(
            [row.created_at for row in all_rows], reverse=True,
        )
        first, total = await reviews.list_reviews(db, uid, page=1, page_size=2)
        second, _ = await reviews.list_reviews(db, uid, page=2, page_size=2)
        assert total == 4 and len(first) == len(second) == 2
        assert {row.id for row in first}.isdisjoint(row.id for row in second)


async def test_direct_profile_crud_enforces_ownership(database, actors):
    owner = actors["users"][0].id
    foreign = actors["profiles"][1].id
    async with database() as db:
        assert await profiles.get_profile(db, foreign, owner) is None
        assert await profiles.update_profile(
            db, foreign, owner, ProfileUpdate(github_username="stolen"),
        ) is None
        assert await profiles.delete_profile(db, foreign, owner) is False
        created = await profiles.create_profile(db, owner, ProfileCreate(github_username="owned"))
        assert created.user_id == owner
        changed = await profiles.update_profile(
            db, UUID(created.id), UUID(owner), ProfileUpdate(github_username="updated"),
        )
        assert changed.github_username == "updated"
        assert await profiles.delete_profile(db, created.id, owner) is True
        assert await profiles.get_profile(db, created.id, owner) is None
        assert (await db.get(Profile, foreign)).github_username is None


@pytest.mark.parametrize("operation", ["create_profile", "read_profile", "update_profile",
                                      "delete_profile", "create_review", "read_review",
                                      "list_reviews", "process_review"])
@pytest.mark.parametrize("principal", ["inactive", "missing", "none"])
async def test_direct_services_reject_invalid_principals(database, actors, operation, principal):
    uid = actors["users"][0].id
    pid = actors["profiles"][0].id
    rid = actors["reviews"][0].id
    async with database() as db:
        if principal == "inactive":
            await db.execute(update(User).where(User.id == uid).values(is_active=False))
            await db.commit()
        elif principal == "missing":
            uid = str(uuid4())
        else:
            uid = None
        calls = {
            "create_profile": lambda: profiles.create_profile(db, uid, ProfileCreate()),
            "read_profile": lambda: profiles.get_profile(db, pid, uid),
            "update_profile": lambda: profiles.update_profile(db, pid, uid, ProfileUpdate()),
            "delete_profile": lambda: profiles.delete_profile(db, pid, uid),
            "create_review": lambda: reviews.create_review(db, pid, uid),
            "read_review": lambda: reviews.get_review(db, rid, uid),
            "list_reviews": lambda: reviews.list_reviews(db, uid),
            "process_review": lambda: reviews.process_review(db, rid, pid, uid),
        }
        with pytest.raises(HTTPException) as denied:
            await calls[operation]()
        assert denied.value.status_code == 401
        assert (await db.get(Review, rid)).status == "pending"


@pytest.mark.parametrize("case", ["foreign_user", "wrong_foreign_profile",
                                 "wrong_owned_profile", "missing_review", "missing_profile"])
async def test_background_access_is_guarded_before_state_changes(database, actors, case):
    uid = actors["users"][0].id
    rid = actors["reviews"][0].id
    pid = actors["profiles"][0].id
    async with database() as db:
        if case == "foreign_user":
            uid = actors["users"][1].id
        elif case == "wrong_foreign_profile":
            pid = actors["profiles"][1].id
        elif case == "wrong_owned_profile":
            extra = await profiles.create_profile(db, uid, ProfileCreate())
            pid = extra.id
        elif case == "missing_review":
            rid = str(uuid4())
        else:
            pid = str(uuid4())
        with pytest.raises(HTTPException) as denied:
            await reviews.process_review(db, rid, pid, uid)
        assert denied.value.status_code == 404
        for original in actors["reviews"]:
            assert (await db.get(Review, original.id)).status == "pending"


async def test_authorized_background_processing_persists(database, actors):
    async with database() as db:
        source=await verified_ingestion(db,actors["profiles"][0].id,actors["users"][0].id)
        job=await reviews.create_review(db,actors["profiles"][0].id,actors["users"][0].id,source.revision_id)
        ident=job.id
        await reviews.process_review(
            db, ident, actors["profiles"][0].id, actors["users"][0].id,
        )
    async with database() as db:
        result = await db.get(Review, ident)
        assert result.status == "complete" and result.sections is None and result.analysis_result["summary"]
        assert (await db.get(Review, actors["reviews"][1].id)).status == "pending"


async def test_deleted_profile_does_not_leave_readable_reports(database, actors):
    async with database() as db:
        await profiles.delete_profile(db, actors["profiles"][0].id, actors["users"][0].id)
        assert await reviews.get_review(
            db, actors["reviews"][0].id, actors["users"][0].id,
        ) is None
        assert await db.get(Review, actors["reviews"][1].id) is not None


async def test_noncanonical_user_id_uses_database_identity(database, actors):
    uid = actors["users"][0].id.upper()
    async with database() as db:
        profile = await profiles.create_profile(db, uid, ProfileCreate())
        assert profile.user_id == actors["users"][0].id
        assert (await profiles.get_profile(db, profile.id, uid)).id == profile.id
        source=await verified_ingestion(db,profile.id,uid)
        review = await reviews.create_review(db, profile.id, uid,source.revision_id)
        assert (await reviews.get_review(db, review.id, uid)).id == review.id
        assert (await reviews.list_reviews(db, uid))[1] == 2


@pytest.mark.parametrize("revocation", ["inactive", "ownership_transfer"])
async def test_background_rechecks_access_before_output_persistence(database, actors, revocation):
    from sqlalchemy import text
    uid = actors["users"][0].id
    other = actors["users"][1].id
    pid = actors["profiles"][0].id
    rid = actors["reviews"][0].id
    # A real database trigger models revocation after the job's initial check.
    # IDs come from generated test UUIDs, never request input.
    if revocation == "inactive":
        action = "UPDATE users SET is_active=FALSE WHERE id='" + uid.replace("-", "") + "';"
    else:
        action = "UPDATE profiles SET user_id='" + other.replace("-", "") + "' WHERE id='" + pid.replace("-", "") + "';"
    async with database() as db:
        source=await verified_ingestion(db,pid,uid)
        job=await reviews.create_review(db,pid,uid,source.revision_id)
        rid=job.id
        from tests.database_probes import revoke_processing_job
        await revoke_processing_job(db, action)
        await db.commit()
        await reviews.process_review(db, rid, pid, uid)
    async with database() as db:
        result = await db.get(Review, rid)
        # Revocation is terminal and sanitized instead of leaving an indefinitely processing job.
        assert result.status == "failed" and result.error_category == "authorization"
        assert result.sections is None and result.overall_score is None
        from core.models import Finding, Evidence
        from sqlalchemy import func, select
        assert await db.scalar(select(func.count()).select_from(Finding)) == 0
        assert await db.scalar(select(func.count()).select_from(Evidence)) == 0
