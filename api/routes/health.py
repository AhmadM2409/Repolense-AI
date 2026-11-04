"""Readiness of the dependencies actually required by the running application."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

from alembic.config import Config
from alembic.script import ScriptDirectory
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db

router = APIRouter(prefix="/health", tags=["health"])
_config = Config(str(Path(__file__).resolve().parents[2] / "alembic.ini"))
_config.set_main_option("script_location", str(Path(__file__).resolve().parents[2] / "alembic"))
_HEAD = ScriptDirectory.from_config(_config).get_current_head()


@router.get("")
async def health_check(db: Annotated[AsyncSession, Depends(get_db)]):
    result = {
        "status": "healthy",
        "dependencies": {
            "postgres": "healthy",
            "redis": "not_required",
            "vector_db": "not_required",
        },
        "safety_events_last_hour": None,
        "timestamp": datetime.now(UTC).isoformat(),
    }
    try:
        versions = (
            (await db.execute(text("SELECT version_num FROM alembic_version"))).scalars().all()
        )
        if versions != [_HEAD]:
            raise RuntimeError("Schema not current")
    except Exception:
        result["status"] = "unhealthy"
        result["dependencies"]["postgres"] = "unhealthy"
        raise HTTPException(503, detail=result) from None
    return result
