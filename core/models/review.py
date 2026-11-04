"""Review model for storing portfolio review results."""

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import uuid4

from sqlalchemy import (
    text,
    Integer,
    JSON,
    ForeignKeyConstraint,
    UniqueConstraint,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from core.database import Base

if TYPE_CHECKING:
    from core.models.profile import Profile


class Review(Base):
    """Model for storing portfolio review results and analysis."""

    __tablename__ = "reviews"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4())
    )
    profile_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("profiles.id", ondelete="CASCADE"),
        nullable=False,
    )
    status: Mapped[str] = mapped_column(
        String(50), nullable=False, default="pending", server_default="pending"
    )  # "pending", "processing", "complete", "failed"
    sections: Mapped[list[dict] | None] = mapped_column(
        JSON(none_as_null=True), nullable=True
    )  # Structured review output
    overall_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC), onupdate=lambda: datetime.now(UTC)
    )

    revision_id: Mapped[str | None] = mapped_column(UUID(as_uuid=False), nullable=True)
    report_kind: Mapped[str] = mapped_column(
        String(30), nullable=False, default="legacy_unverified", server_default="legacy_unverified"
    )

    job_type: Mapped[str] = mapped_column(
        String(30), nullable=False, default="review", server_default="review"
    )
    requested_repository: Mapped[str | None] = mapped_column(String(255), nullable=True)
    ingestion_result: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    analysis_result: Mapped[dict | None] = mapped_column(JSON(none_as_null=True), nullable=True)

    attempt_count: Mapped[int] = mapped_column(Integer, default=0, server_default="0", nullable=False)
    max_attempts: Mapped[int] = mapped_column(Integer, default=3, server_default="3", nullable=False)
    worker_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    execution_token: Mapped[str | None] = mapped_column(String(36), nullable=True)
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    retry_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    cancel_requested_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    error_category: Mapped[str | None] = mapped_column(String(64), nullable=True)
    progress_stage: Mapped[str] = mapped_column(String(64), default="queued", server_default="queued", nullable=False)
    progress_pct: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Relationships
    profile: Mapped["Profile"] = relationship("Profile", back_populates="reviews")

    __table_args__ = (
        CheckConstraint("attempt_count >= 0 AND max_attempts BETWEEN 1 AND 10 AND attempt_count <= max_attempts", name="ck_job_attempts"),
        CheckConstraint("progress_pct IS NULL OR progress_pct BETWEEN 0 AND 100", name="ck_job_progress"),
        CheckConstraint("(worker_id IS NULL) = (execution_token IS NULL)", name="ck_job_worker_token"),
        Index("ix_job_claimable", "status", "retry_at", "created_at"),
        Index("ix_job_expired", "status", "lease_expires_at"),
        Index(
            "uq_active_analysis_profile",
            "profile_id",
            unique=True,
            postgresql_where=text(
                "job_type='review' AND report_kind='evidence' AND status IN ('pending','processing')"
            ),
            sqlite_where=text(
                "job_type='review' AND report_kind='evidence' AND status IN ('pending','processing')"
            ),
        ),
        CheckConstraint("job_type IN ('review','github_ingestion','retrieval_index','investigation')", name="ck_review_job_type"),
        CheckConstraint(
            "job_type = 'review' OR (requested_repository IS NOT NULL AND sections IS NULL AND overall_score IS NULL)",
            name="ck_ingestion_job",
        ),
        Index(
            "uq_active_ingestion_profile",
            "profile_id",
            unique=True,
            postgresql_where=text(
                "job_type = 'github_ingestion' AND status IN ('pending','processing')"
            ),
            sqlite_where=text(
                "job_type = 'github_ingestion' AND status IN ('pending','processing')"
            ),
        ),
        UniqueConstraint("id", "profile_id", name="uq_review_scope"),
        UniqueConstraint("id", "revision_id", "profile_id", name="uq_review_revision_scope"),
        ForeignKeyConstraint(
            ["revision_id", "profile_id"],
            ["repository_revisions.id", "repository_revisions.profile_id"],
            ondelete="CASCADE",
            name="fk_review_revision_scope",
        ),
        CheckConstraint(
            "status IN ('pending', 'processing', 'complete', 'failed', 'cancelled')", name="ck_review_status"
        ),
        CheckConstraint(
            "report_kind IN ('legacy_unverified', 'demo', 'evidence')", name="ck_review_kind"
        ),
        Index("ix_reviews_profile_id", "profile_id"),
        Index("ix_reviews_status", "status"),
        Index("ix_reviews_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<Review(id={self.id}, profile_id={self.profile_id}, status={self.status})>"
