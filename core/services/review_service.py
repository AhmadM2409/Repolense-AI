"""Compatibility review access. Repository analysis replaces the old canned pipeline."""

from uuid import UUID

from fastapi import HTTPException
from sqlalchemy import func, or_, select

from core.database import AsyncSessionLocal
from core.models import Profile, Review
from core.services.authorization import require_active_user
from core.services.profile_service import get_profile


async def create_review(db, profile_id: UUID, user_id: UUID, revision_id: UUID | None = None):
    profile = await get_profile(db, profile_id, user_id)
    if profile is None:
        raise HTTPException(404, "Profile not found")
    if revision_id is None:
        raise HTTPException(422, "Ingest a repository first and provide its revision_id")
    ingestion = await db.scalar(
        select(Review)
        .where(
            Review.profile_id == profile.id,
            Review.job_type == "github_ingestion",
            Review.status == "complete",
            Review.revision_id == str(revision_id),
        )
        .order_by(Review.created_at.desc())
    )
    if ingestion is None:
        raise HTTPException(404, "Completed repository ingestion not found")
    from core.services.repository_analysis import submit

    job, _ = await submit(db, ingestion.id, user_id)
    return job


async def get_review(db, review_id, user_id):
    user = await require_active_user(db, user_id)
    return await db.scalar(
        select(Review)
        .join(Profile)
        .where(Review.id == str(review_id), Profile.user_id == user.id, Review.job_type == "review")
    )


async def list_reviews(db, user_id, page=1, page_size=20, supported_only=False):
    user = await require_active_user(db, user_id)
    conditions = (
        Profile.user_id == user.id,
        Review.job_type == "review",
        or_(Review.status != "complete", Review.report_kind == "evidence"),
    )
    query = select(Review).join(Profile).where(*conditions)
    if supported_only:
        query = query.where(Review.analysis_result.is_not(None))
    total = await db.scalar(select(func.count()).select_from(query.subquery()))
    rows = (
        await db.scalars(
            query.order_by(Review.created_at.desc()).offset((page - 1) * page_size).limit(page_size)
        )
    ).all()
    return rows, total


async def process_review(db, review_id, profile_id, user_id):
    review = await get_review(db, review_id, user_id)
    if review is None or review.profile_id != str(profile_id):
        raise HTTPException(404, "Review not found")
    if not review.analysis_result:
        # Historical jobs/data remain stored; unsupported work cannot become a fabricated report.
        review.status = "failed"
        review.error_message = (
            "Legacy review processing is disabled; analyze a verified repository ingestion"
        )
        await db.commit()
        return
    from core.services.repository_analysis import process

    await process(db, review_id, user_id)


async def run_review_job(review_id, profile_id, user_id):
    async with AsyncSessionLocal() as db:
        try:
            await process_review(db, review_id, profile_id, user_id)
        except HTTPException:
            await db.rollback()
