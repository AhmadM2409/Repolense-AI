"""Versioned PostgreSQL-only retrieval generations and canonical chunk vectors."""

import json
from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import (
    JSON,
    CheckConstraint,
    DateTime,
    ForeignKeyConstraint,
    Index,
    Integer,
    String,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.dialects.postgresql.base import ischema_names
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import UserDefinedType

from core.database import Base


class Vector(UserDefinedType):
    cache_ok = True

    def get_col_spec(self, **kw):
        return "public.vector"

    def bind_processor(self, dialect):
        return lambda value: (
            None if value is None else json.dumps(value, allow_nan=False, separators=(",", ":"))
        )

    def result_processor(self, dialect, coltype):
        return lambda value: (
            None if value is None else json.loads(value) if isinstance(value, str) else value
        )


ischema_names["vector"] = Vector
ischema_names["public.vector"] = Vector


class RetrievalIndex(Base):
    __tablename__ = "retrieval_indexes"
    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4())
    )
    profile_id: Mapped[str] = mapped_column(UUID(as_uuid=False), nullable=False)
    revision_id: Mapped[str] = mapped_column(UUID(as_uuid=False), nullable=False)
    ingestion_id: Mapped[str] = mapped_column(UUID(as_uuid=False), nullable=False)
    job_id: Mapped[str] = mapped_column(UUID(as_uuid=False), nullable=False, unique=True)
    provider: Mapped[str] = mapped_column(String(20), nullable=False)
    model: Mapped[str] = mapped_column(String(128), nullable=False)
    dimensions: Mapped[int] = mapped_column(Integer, nullable=False)
    metric: Mapped[str] = mapped_column(String(20), nullable=False, server_default="cosine")
    config_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    config: Mapped[dict] = mapped_column(JSON, nullable=False)
    manifest_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    chunk_manifest: Mapped[list] = mapped_column(JSON, nullable=False)
    expected_chunks: Mapped[int] = mapped_column(Integer, nullable=False)
    state: Mapped[str] = mapped_column(String(20), nullable=False, server_default="pending")
    attempt_token: Mapped[str | None] = mapped_column(String(36))
    reserved_requests: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    published_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    __table_args__ = (
        ForeignKeyConstraint(
            ["revision_id", "profile_id"],
            ["repository_revisions.id", "repository_revisions.profile_id"],
            ondelete="CASCADE",
            name="fk_index_revision",
        ),
        ForeignKeyConstraint(
            ["job_id", "revision_id", "profile_id"],
            ["reviews.id", "reviews.revision_id", "reviews.profile_id"],
            ondelete="CASCADE",
            name="fk_index_job",
        ),
        ForeignKeyConstraint(
            ["ingestion_id", "revision_id", "profile_id"],
            ["reviews.id", "reviews.revision_id", "reviews.profile_id"],
            ondelete="CASCADE",
            name="fk_index_ingestion",
        ),
        UniqueConstraint(
            "id",
            "revision_id",
            "profile_id",
            "config_fingerprint",
            "dimensions",
            name="uq_index_vector_scope",
        ),
        CheckConstraint(
            "state IN ('pending','building','ready','failed','cancelled')", name="ck_index_state"
        ),
        CheckConstraint("metric = 'cosine'", name="ck_index_metric"),
        CheckConstraint(
            "(provider='lmstudio' AND model='text-embedding-nomic-embed-text-v1.5' AND dimensions=768) OR (provider='mistral' AND model='mistral-embed' AND dimensions=1024)",
            name="ck_index_model",
        ),
        CheckConstraint(
            "expected_chunks BETWEEN 0 AND 2000 AND published_count BETWEEN 0 AND expected_chunks AND reserved_requests BETWEEN 0 AND 512",
            name="ck_index_counts",
        ),
        CheckConstraint(
            "length(config_fingerprint)=64 AND length(manifest_fingerprint)=64",
            name="ck_index_digests",
        ),
        CheckConstraint(
            "(state='ready') = (published_at IS NOT NULL) AND (state<>'ready' OR published_count=expected_chunks)",
            name="ck_index_published",
        ),
        CheckConstraint(
            "json_array_length(chunk_manifest)=expected_chunks AND octet_length(chunk_manifest::text)<=1048576",
            name="ck_index_manifest",
        ),
        Index("ix_index_revision_config", "profile_id", "revision_id", "config_fingerprint"),
        Index(
            "uq_index_active",
            "profile_id",
            "revision_id",
            "config_fingerprint",
            unique=True,
            postgresql_where=text("state IN ('pending','building','ready')"),
        ),
    )


class ChunkEmbedding(Base):
    __tablename__ = "chunk_embeddings"
    index_id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True)
    chunk_id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True)
    source_id: Mapped[str] = mapped_column(UUID(as_uuid=False), nullable=False)
    revision_id: Mapped[str] = mapped_column(UUID(as_uuid=False), nullable=False)
    profile_id: Mapped[str] = mapped_column(UUID(as_uuid=False), nullable=False)
    config_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    dimensions: Mapped[int] = mapped_column(Integer, nullable=False)
    attempt_token: Mapped[str] = mapped_column(String(36), nullable=False)
    input_digest: Mapped[str] = mapped_column(String(64), nullable=False)
    embedding: Mapped[list] = mapped_column(Vector(), nullable=False)
    __table_args__ = (
        ForeignKeyConstraint(
            ["index_id", "revision_id", "profile_id", "config_fingerprint", "dimensions"],
            [
                "retrieval_indexes.id",
                "retrieval_indexes.revision_id",
                "retrieval_indexes.profile_id",
                "retrieval_indexes.config_fingerprint",
                "retrieval_indexes.dimensions",
            ],
            ondelete="CASCADE",
            name="fk_vector_index",
        ),
        ForeignKeyConstraint(
            ["chunk_id", "source_id", "profile_id"],
            ["source_chunks.id", "source_chunks.source_id", "source_chunks.profile_id"],
            ondelete="CASCADE",
            name="fk_vector_chunk",
        ),
        ForeignKeyConstraint(
            ["source_id", "revision_id", "profile_id"],
            ["ingested_sources.id", "ingested_sources.revision_id", "ingested_sources.profile_id"],
            ondelete="CASCADE",
            name="fk_vector_source_revision",
        ),
        CheckConstraint(
            "public.vector_dims(embedding)=dimensions AND public.vector_norm(embedding)>0",
            name="ck_vector_values",
        ),
        CheckConstraint("length(input_digest)=64", name="ck_vector_digest"),
        Index("ix_vector_revision", "profile_id", "revision_id", "index_id"),
    )
