"""Bounded owner catalog projections over existing repository models."""

from fastapi import HTTPException
from sqlalchemy import select

from core.models import Profile, Repository, RepositoryRevision, User
from core.services.authorization import require_active_user


async def repositories(db, uid, limit, offset):
    await require_active_user(db, uid)
    rows = (
        (
            await db.execute(
                select(Repository.id, Repository.full_name, Repository.provider)
                .join(Profile, Profile.id == Repository.profile_id)
                .join(User, User.id == Profile.user_id)
                .where(User.id == uid, User.is_active.is_(True))
                .order_by(Repository.id)
                .offset(offset)
                .limit(limit + 1)
                .with_for_update(of=(User, Profile, Repository))
            )
        )
        .mappings()
        .all()
    )
    return {
        "items": [dict(r) for r in rows[:limit]],
        "next_offset": offset + limit if len(rows) > limit else None,
    }


async def revisions(db, uid, repo, limit, offset):
    await require_active_user(db, uid)
    owned = await db.scalar(
        select(Repository.id)
        .join(Profile, Profile.id == Repository.profile_id)
        .where(Repository.id == str(repo), Profile.user_id == uid)
        .with_for_update(of=(Profile, Repository))
    )
    if owned is None:
        raise HTTPException(404, "Repository not found")
    rows = (
        (
            await db.execute(
                select(
                    RepositoryRevision.id,
                    RepositoryRevision.repository_id,
                    RepositoryRevision.commit_sha,
                )
                .where(RepositoryRevision.repository_id == str(repo))
                .order_by(RepositoryRevision.id)
                .offset(offset)
                .limit(limit + 1)
            )
        )
        .mappings()
        .all()
    )
    return {
        "items": [dict(r) for r in rows[:limit]],
        "next_offset": offset + limit if len(rows) > limit else None,
    }
