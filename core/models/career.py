"""J1 immutable input snapshots and ordered canonical ingestion references."""

from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import (
    JSON,
    CheckConstraint,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Integer,
    String,
    Text,
    UniqueConstraint,
    event,
    inspect,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from core.database import Base


class CareerInput(Base):
    __tablename__ = "career_inputs"
    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4())
    )
    owner_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("users.id", ondelete="CASCADE")
    )
    profile_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("profiles.id", ondelete="CASCADE")
    )
    account: Mapped[str] = mapped_column(String(39))
    job_description: Mapped[str] = mapped_column(Text)
    job_hash: Mapped[str] = mapped_column(String(64))
    discovery: Mapped[dict] = mapped_column(JSON)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    __table_args__ = (
        UniqueConstraint("id", "profile_id", name="uq_career_scope"),
        CheckConstraint(
            "length(job_hash)=64 AND octet_length(job_description)<=65536 AND length(job_description) BETWEEN 20 AND 20000",
            name="ck_career_job",
        ),
        CheckConstraint("octet_length(discovery::text)<=262144", name="ck_career_discovery"),
    )


class CareerSelection(Base):
    __tablename__ = "career_selections"
    input_id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True)
    position: Mapped[int] = mapped_column(Integer, primary_key=True)
    profile_id: Mapped[str] = mapped_column(UUID(as_uuid=False))
    full_name: Mapped[str] = mapped_column(String(255))
    job_id: Mapped[str | None] = mapped_column(UUID(as_uuid=False))
    revision_id: Mapped[str | None] = mapped_column(UUID(as_uuid=False))
    commit_sha: Mapped[str | None] = mapped_column(String(40))
    __table_args__ = (
        ForeignKeyConstraint(
            ["input_id", "profile_id"],
            ["career_inputs.id", "career_inputs.profile_id"],
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(["job_id", "profile_id"], ["reviews.id", "reviews.profile_id"]),
        ForeignKeyConstraint(
            ["revision_id", "profile_id"],
            ["repository_revisions.id", "repository_revisions.profile_id"],
        ),
        ForeignKeyConstraint(
            ["job_id", "revision_id", "profile_id"],
            ["reviews.id", "reviews.revision_id", "reviews.profile_id"],
        ),
        UniqueConstraint("input_id", "full_name", name="uq_career_selection"),
        CheckConstraint("position BETWEEN 0 AND 9", name="ck_career_position"),
        CheckConstraint(
            "(revision_id IS NULL) = (commit_sha IS NULL) AND (revision_id IS NULL OR job_id IS NOT NULL)",
            name="ck_career_pin",
        ),
    )


@event.listens_for(CareerInput, "before_update")
def immutable_input(mapper, connection, target):
    if any(
        inspect(target).attrs[k].history.has_changes()
        for k in ("owner_id", "profile_id", "account", "job_description", "job_hash", "discovery")
    ):
        raise ValueError("Career input snapshots are immutable; create a new input")


@event.listens_for(CareerSelection, "before_update")
def immutable_pin(mapper, connection, target):
    state = inspect(target)
    for key in ("job_id", "revision_id", "commit_sha"):
        history = state.attrs[key].history
        if history.has_changes() and history.deleted and history.deleted[0] is not None:
            raise ValueError("Selected ingestion and pinned revision cannot be replaced")
