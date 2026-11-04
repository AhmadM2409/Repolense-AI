"""Persistent investigations reuse Review queue ownership/fencing."""

from datetime import datetime

from sqlalchemy import (
    JSON,
    CheckConstraint,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    String,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from core.database import Base


class Investigation(Base):
    __tablename__ = "investigations"
    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("reviews.id", ondelete="CASCADE"), primary_key=True
    )
    owner_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    profile_id: Mapped[str] = mapped_column(UUID(as_uuid=False), nullable=False)
    repository_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("repositories.id", ondelete="CASCADE"), nullable=False
    )
    revision_id: Mapped[str] = mapped_column(UUID(as_uuid=False), nullable=False)
    request: Mapped[dict] = mapped_column(JSON, nullable=False)
    state: Mapped[dict] = mapped_column(JSON, nullable=False)
    snapshot_digest: Mapped[str] = mapped_column(String(64), nullable=False)
    deadline_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    outcome: Mapped[str | None] = mapped_column(String(40), nullable=True)
    final: Mapped[dict | None] = mapped_column(JSON(none_as_null=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    __table_args__ = (
        CheckConstraint(
            "json_typeof(state) = 'object' AND state->'version' IS NOT NULL AND json_typeof(state->'version') = 'number' AND state->>'version' = '1' AND state->'events' IS NOT NULL AND json_typeof(state->'events') = 'object' AND state->'attempts' IS NOT NULL AND json_typeof(state->'attempts') = 'array' AND json_array_length(state->'attempts') <= 6 AND octet_length(state::text) <= 131072",
            name="ck_investigation_state",
        ),
        CheckConstraint(
            "octet_length(request::text) <= 8192 AND (final IS NULL OR octet_length(final::text) <= 131072)",
            name="ck_investigation_sizes",
        ),
        CheckConstraint("deadline_at > created_at", name="ck_investigation_deadline"),
        CheckConstraint(
            "outcome IS NULL OR outcome IN ('completed','insufficient_evidence','step_limit','provider_failure','validation_failure','invalid_decision','cancelled','timeout','scope_changed','unknown_provider_outcome')",
            name="ck_investigation_outcome",
        ),
        ForeignKeyConstraint(
            ["id", "revision_id", "profile_id"],
            ["reviews.id", "reviews.revision_id", "reviews.profile_id"],
            ondelete="CASCADE",
        ),
        Index("ix_investigation_owner_created", "owner_id", "created_at"),
        Index("ix_investigation_revision", "revision_id"),
    )
