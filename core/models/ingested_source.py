"""Ingested source model for tracking imported content."""

from datetime import datetime
from typing import TYPE_CHECKING
from uuid import uuid4

from sqlalchemy import JSON, Text, DateTime, ForeignKey, ForeignKeyConstraint, UniqueConstraint, CheckConstraint, Index, Integer, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from core.database import Base

if TYPE_CHECKING:
    from core.models.profile import Profile


class IngestedSource(Base):
    """Model for tracking ingested portfolio sources (resumes, READMEs, repos, web pages)."""

    __tablename__ = "ingested_sources"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4())
    )
    profile_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("profiles.id", ondelete="CASCADE"),
        nullable=False,
    )
    source_type: Mapped[str] = mapped_column(
        String(50), nullable=False
    )  # "resume", "readme", "repo", "web"
    source_url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    filename: Mapped[str | None] = mapped_column(String(255), nullable=True)
    content_hash: Mapped[str | None] = mapped_column(
        String(64), nullable=True
    )  # SHA256 hash for deduplication
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow
    )

    raw_data: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    text: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_metadata: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    revision_id: Mapped[str | None] = mapped_column(UUID(as_uuid=False), nullable=True)
    path: Mapped[str | None] = mapped_column(String(1024), nullable=True)

    # Relationships
    profile: Mapped["Profile"] = relationship("Profile", back_populates="ingested_sources")

    __table_args__ = (
        UniqueConstraint("id", "profile_id", name="uq_source_scope"),
        UniqueConstraint("profile_id", "revision_id", "path", name="uq_source_revision_path"),
        UniqueConstraint("id", "revision_id", "profile_id", name="uq_source_revision_scope"),
        ForeignKeyConstraint(["revision_id", "profile_id"], ["repository_revisions.id", "repository_revisions.profile_id"], ondelete="CASCADE", name="fk_source_revision_scope"),
        CheckConstraint("(revision_id IS NULL AND path IS NULL) OR (revision_id IS NOT NULL AND path IS NOT NULL)", name="ck_source_revision_path"),
        CheckConstraint("chunk_count >= 0", name="ck_source_chunk_count"),
        Index("ix_ingested_sources_profile_id", "profile_id"),
        Index("ix_ingested_sources_content_hash", "content_hash"),
        Index("ix_ingested_sources_source_type", "source_type"),
    )

    def __repr__(self) -> str:
        return f"<IngestedSource(id={self.id}, profile_id={self.profile_id}, source_type={self.source_type})>"
