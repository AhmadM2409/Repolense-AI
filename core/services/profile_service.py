from uuid import UUID
import structlog
from sqlalchemy import select

from core.services.authorization import require_active_user
from core.models.profile import Profile
from core.models.review import Review
from core.models.ingested_source import IngestedSource
from api.schemas.profile import ProfileCreate, ProfileUpdate

log = structlog.get_logger()


async def create_profile(
    db,
    user_id: UUID,
    data: ProfileCreate,
    resume_filename: str = None,
    resume_text: str = None,
) -> Profile:
    """
    Create a new profile for a user.
    """
    user = await require_active_user(db, user_id)
    profile = Profile(
        user_id=user.id,
        github_username=data.github_username,
        portfolio_url=data.portfolio_url,
        resume_filename=resume_filename,
        resume_text=resume_text,
    )
    db.add(profile)
    try:
        await db.commit()
        await db.refresh(profile)
    except Exception:
        await db.rollback()
        raise
    return profile


async def get_profile(
    db,
    profile_id: UUID,
    user_id: UUID,
) -> Profile | None:
    """
    Get a profile by ID, checking ownership.
    """
    user = await require_active_user(db, user_id)
    stmt = select(Profile).where(
        (Profile.id == str(profile_id)) & (Profile.user_id == user.id)
    )
    result = await db.execute(stmt)
    return result.scalars().first()


async def update_profile(
    db,
    profile_id: UUID,
    user_id: UUID,
    data: ProfileUpdate,
) -> Profile | None:
    """
    Update a profile, checking ownership.
    """
    profile = await get_profile(db, profile_id, user_id)
    if not profile:
        return None

    if "github_username" in data.model_fields_set:
        profile.github_username = data.github_username
    if "portfolio_url" in data.model_fields_set:
        profile.portfolio_url = data.portfolio_url

    db.add(profile)
    try:
        await db.commit()
        await db.refresh(profile)
    except Exception:
        await db.rollback()
        raise
    return profile


async def delete_profile(
    db,
    profile_id: UUID,
    user_id: UUID,
) -> bool:
    """
    Delete a profile and cascade delete reviews and ingested sources.
    Returns True if deleted, False if not found.
    """
    profile = await get_profile(db, profile_id, user_id)
    if not profile:
        return False

    try:
        # Delete related reviews
        stmt = select(Review).where(Review.profile_id == str(profile_id))
        result = await db.execute(stmt)
        reviews = result.scalars().all()
        for review in reviews:
            await db.delete(review)

        # Delete related ingested sources
        stmt = select(IngestedSource).where(IngestedSource.profile_id == str(profile_id))
        result = await db.execute(stmt)
        sources = result.scalars().all()
        for source in sources:
            await db.delete(source)

        # Delete profile
        await db.delete(profile)
        await db.commit()

        log.info("profile_deleted_cascade", profile_id=str(profile_id))
        return True

    except Exception as exc:
        log.error("profile_cascade_delete_failed", profile_id=str(profile_id), error=str(exc))
        await db.rollback()
        raise
