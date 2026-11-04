"""Private process credential; existing RepoLens JWT verification only."""

from uuid import UUID

from fastapi import HTTPException

from core.security import decode_access_token
from core.services.authorization import require_active_user


async def principal(db, token, expected=None):
    payload = decode_access_token(token) if token else None
    try:
        uid = str(UUID(payload["sub"])) if payload else None
    except (ValueError, TypeError, KeyError, AttributeError):
        uid = None
    if uid is None or (expected is not None and uid != expected):
        raise HTTPException(401, "Authentication required")
    await require_active_user(db, uid)
    return uid
