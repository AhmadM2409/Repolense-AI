"""Persisted chunks, findings and evidence; composite keys prevent cross-profile links."""

from uuid import uuid4
from sqlalchemy import (
    Text,
    String,
    Integer,
    ForeignKeyConstraint,
    UniqueConstraint,
    CheckConstraint,
    Index,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column
from core.database import Base


class SourceChunk(Base):
    __tablename__ = "source_chunks"
    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4())
    )
    profile_id: Mapped[str] = mapped_column(UUID(as_uuid=False), nullable=False)
    source_id: Mapped[str] = mapped_column(UUID(as_uuid=False), nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    line_start: Mapped[int] = mapped_column(Integer, nullable=False)
    line_end: Mapped[int] = mapped_column(Integer, nullable=False)
    __table_args__ = (
        ForeignKeyConstraint(
            ["source_id", "profile_id"],
            ["ingested_sources.id", "ingested_sources.profile_id"],
            ondelete="CASCADE",
            name="fk_chunk_source_scope",
        ),
        UniqueConstraint("source_id", "chunk_index", name="uq_chunk_position"),
        UniqueConstraint("id", "source_id", "profile_id", name="uq_chunk_vector_scope"),
        CheckConstraint(
            "chunk_index >= 0 AND line_start >= 1 AND line_end >= line_start", name="ck_chunk_range"
        ),
    )


class Finding(Base):
    __tablename__ = "findings"
    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4())
    )
    profile_id: Mapped[str] = mapped_column(UUID(as_uuid=False), nullable=False)
    review_id: Mapped[str] = mapped_column(UUID(as_uuid=False), nullable=False)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    detail: Mapped[str] = mapped_column(Text, nullable=False)
    rule_id: Mapped[str] = mapped_column(
        String(64), nullable=False, default="legacy", server_default="legacy"
    )
    severity: Mapped[str] = mapped_column(
        String(16), nullable=False, default="info", server_default="info"
    )
    category: Mapped[str] = mapped_column(
        String(32), nullable=False, default="maintainability", server_default="maintainability"
    )
    certainty: Mapped[str] = mapped_column(
        String(16), nullable=False, default="observation", server_default="observation"
    )
    fingerprint: Mapped[str | None] = mapped_column(String(64), nullable=True)
    __table_args__ = (
        CheckConstraint(
            "severity IN ('info','low','medium','high','critical')", name="ck_finding_severity"
        ),
        CheckConstraint(
            "category IN ('security','maintainability','configuration')", name="ck_finding_category"
        ),
        CheckConstraint("certainty IN ('observation','potential')", name="ck_finding_certainty"),
        Index("uq_finding_fingerprint", "review_id", "fingerprint", unique=True),
        ForeignKeyConstraint(
            ["review_id", "profile_id"],
            ["reviews.id", "reviews.profile_id"],
            ondelete="CASCADE",
            name="fk_finding_review_scope",
        ),
        UniqueConstraint("id", "profile_id", name="uq_finding_scope"),
        UniqueConstraint("id", "review_id", "profile_id", name="uq_finding_review_scope"),
    )


class Evidence(Base):
    __tablename__ = "evidence_spans"
    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4())
    )
    profile_id: Mapped[str] = mapped_column(UUID(as_uuid=False), nullable=False)
    finding_id: Mapped[str] = mapped_column(UUID(as_uuid=False), nullable=False)
    source_id: Mapped[str] = mapped_column(UUID(as_uuid=False), nullable=False)
    review_id: Mapped[str] = mapped_column(UUID(as_uuid=False), nullable=False)
    revision_id: Mapped[str] = mapped_column(UUID(as_uuid=False), nullable=False)
    line_start: Mapped[int] = mapped_column(Integer, nullable=False)
    line_end: Mapped[int] = mapped_column(Integer, nullable=False)
    quote: Mapped[str] = mapped_column(Text, nullable=False)
    __table_args__ = (
        ForeignKeyConstraint(
            ["finding_id", "review_id", "profile_id"],
            ["findings.id", "findings.review_id", "findings.profile_id"],
            ondelete="CASCADE",
            name="fk_evidence_finding_scope",
        ),
        ForeignKeyConstraint(
            ["source_id", "revision_id", "profile_id"],
            ["ingested_sources.id", "ingested_sources.revision_id", "ingested_sources.profile_id"],
            ondelete="CASCADE",
            name="fk_evidence_source_scope",
        ),
        ForeignKeyConstraint(
            ["review_id", "revision_id", "profile_id"],
            ["reviews.id", "reviews.revision_id", "reviews.profile_id"],
            ondelete="CASCADE",
            name="fk_evidence_review_revision",
        ),
        CheckConstraint("line_start >= 1 AND line_end >= line_start", name="ck_evidence_range"),
    )
