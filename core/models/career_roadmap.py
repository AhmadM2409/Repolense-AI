"""A versioned, fenced interpretation of a completed immutable J2 snapshot."""

from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import (
    JSON,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from core.database import Base


class CareerRoadmap(Base):
    __tablename__ = "career_roadmaps"
    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4())
    )
    assessment_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("career_assessments.id", ondelete="CASCADE")
    )
    version: Mapped[int] = mapped_column(Integer, default=1)
    source_signature: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(16), default="generating")
    error_category: Mapped[str | None] = mapped_column(String(40))
    result: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    operation_token: Mapped[str | None] = mapped_column(String(36))
    lease_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    __table_args__ = (
        UniqueConstraint("assessment_id", name="uq_roadmap_assessment"),
        CheckConstraint(
            "version=1 AND status IN ('generating','complete','failed','cancelled') AND octet_length(result::text)<=65536",
            name="ck_roadmap_bounds",
        ),
    )
