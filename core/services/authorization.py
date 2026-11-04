"""Active-principal checks shared by dependencies and direct service calls."""
from uuid import UUID

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.models.user import User


async def require_active_user(db: AsyncSession, user_id: str | UUID | None) -> User:
    """Resolve a trusted principal ID against current database activity state."""
    denied = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid authentication credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        normalized_id = str(UUID(str(user_id)))
    except (ValueError, TypeError, AttributeError):
        raise denied from None
    result = await db.execute(
        select(User)
        .where(User.id == normalized_id, User.is_active.is_(True))
        .execution_options(populate_existing=True)
    )
    user = result.scalar_one_or_none()
    if user is None:
        raise denied
    return user
