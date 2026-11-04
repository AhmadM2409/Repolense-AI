"""J2 results and separate durable aggregate reservations; never historical ledgers."""

from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from core.database import Base


class CareerAssessment(Base):
    __tablename__ = "career_assessments"
    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4())
    )
    owner_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("users.id", ondelete="CASCADE")
    )
    input_id: Mapped[str] = mapped_column(UUID(as_uuid=False))
    profile_id: Mapped[str] = mapped_column(UUID(as_uuid=False))
    request_id: Mapped[str] = mapped_column(UUID(as_uuid=False))
    snapshot: Mapped[dict] = mapped_column(JSON)
    signature: Mapped[str] = mapped_column(String(64))
    provider_signature: Mapped[str] = mapped_column(String(64))
    mode: Mapped[str] = mapped_column(String(12))
    consent: Mapped[str] = mapped_column(String(64))
    requirements: Mapped[list] = mapped_column(JSON, default=list)
    matches: Mapped[list] = mapped_column(JSON, default=list)
    calls: Mapped[list] = mapped_column(JSON, default=list)
    status: Mapped[str] = mapped_column(String(16), default="extracting")
    error_category: Mapped[str | None] = mapped_column(String(40))
    operation_token: Mapped[str | None] = mapped_column(String(36))
    lease_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    __table_args__ = (
        ForeignKeyConstraint(
            ["input_id", "profile_id"],
            ["career_inputs.id", "career_inputs.profile_id"],
            ondelete="CASCADE",
        ),
        UniqueConstraint("owner_id", "request_id", name="uq_assessment_request"),
        CheckConstraint(
            "status IN ('extracting','review','matching','complete','failed','cancelled')",
            name="ck_assessment_status",
        ),
        CheckConstraint("mode IN ('keyword','semantic','hybrid')", name="ck_assessment_mode"),
        CheckConstraint(
            "json_array_length(calls)<=12 AND json_array_length(requirements)<=4 AND json_array_length(matches)<=4 AND octet_length(snapshot::text)<=32768 AND octet_length(requirements::text)<=16384 AND octet_length(matches::text)<=65536 AND octet_length(calls::text)<=16384",
            name="ck_assessment_bounds",
        ),
    )


class CareerAIBudget(Base):
    __tablename__ = "career_ai_budget"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    spent: Mapped[int] = mapped_column(Integer, default=0)
    stopped: Mapped[bool] = mapped_column(Boolean, default=False)
    recent: Mapped[list] = mapped_column(JSON, default=list)
    owners: Mapped[dict] = mapped_column(JSON, default=dict)
    __table_args__ = (
        CheckConstraint(
            "id=1 AND spent BETWEEN 0 AND 100 AND json_array_length(recent)<=10",
            name="ck_career_budget",
        ),
    )
