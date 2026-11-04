"""Repository identities and immutable revisions owned through existing profiles."""

from uuid import uuid4
from sqlalchemy import (
    String,
    ForeignKey,
    ForeignKeyConstraint,
    UniqueConstraint,
    CheckConstraint,
    event,
    inspect,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column
from core.database import Base


class Repository(Base):
    __tablename__ = "repositories"
    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4())
    )
    profile_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("profiles.id", ondelete="CASCADE"), nullable=False
    )
    provider: Mapped[str] = mapped_column(
        String(20), nullable=False, default="github", server_default="github"
    )
    full_name: Mapped[str] = mapped_column(String(255), nullable=False)
    __table_args__ = (
        UniqueConstraint("id", "profile_id", name="uq_repositories_scope"),
        UniqueConstraint("profile_id", "provider", "full_name", name="uq_repositories_identity"),
        CheckConstraint("provider = 'github'", name="ck_repository_provider"),
    )


class RepositoryRevision(Base):
    __tablename__ = "repository_revisions"
    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4())
    )
    profile_id: Mapped[str] = mapped_column(UUID(as_uuid=False), nullable=False)
    repository_id: Mapped[str] = mapped_column(UUID(as_uuid=False), nullable=False)
    commit_sha: Mapped[str] = mapped_column(String(40), nullable=False)
    __table_args__ = (
        ForeignKeyConstraint(
            ["repository_id", "profile_id"],
            ["repositories.id", "repositories.profile_id"],
            ondelete="CASCADE",
            name="fk_revision_repository_scope",
        ),
        UniqueConstraint("id", "profile_id", name="uq_revision_scope"),
        UniqueConstraint("repository_id", "commit_sha", name="uq_revision_sha"),
        CheckConstraint(
            "length(commit_sha) = 40 AND commit_sha = lower(commit_sha) AND length(replace(replace(replace(replace(replace(replace(replace(replace(replace(replace(replace(replace(replace(replace(replace(replace(commit_sha, '0', ''), '1', ''), '2', ''), '3', ''), '4', ''), '5', ''), '6', ''), '7', ''), '8', ''), '9', ''), 'a', ''), 'b', ''), 'c', ''), 'd', ''), 'e', ''), 'f', '')) = 0",
            name="ck_revision_sha",
        ),
    )


@event.listens_for(RepositoryRevision, "before_update")
def immutable_revision(mapper, connection, target):
    if any(
        inspect(target).attrs[key].history.has_changes()
        for key in ("id", "profile_id", "repository_id", "commit_sha")
    ):
        raise ValueError("Repository revisions are immutable")
