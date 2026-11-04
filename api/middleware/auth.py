"""JWT authentication backed by current user state."""
from typing import Annotated
from uuid import UUID

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db
from core.models.user import User
from core.security import decode_access_token
from core.services.authorization import require_active_user

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=False)


async def get_current_user(
    token: Annotated[str | None, Depends(oauth2_scheme)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> User:
    """Reject missing/invalid tokens, invalid subjects, and inactive/deleted users."""
    payload = decode_access_token(token) if token else None
    try:
        user_id = UUID(payload["sub"]) if payload else None
    except (KeyError, ValueError, TypeError, AttributeError):
        user_id = None
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return await require_active_user(db, user_id)
