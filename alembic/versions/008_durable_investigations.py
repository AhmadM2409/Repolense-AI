"""Durable investigation records on the existing queue."""

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

from alembic import op

revision = "008"
down_revision = "007"
branch_labels = depends_on = None


def upgrade():
    op.drop_constraint("ck_review_job_type", "reviews", type_="check")
    op.create_check_constraint(
        "ck_review_job_type",
        "reviews",
        "job_type IN ('review','github_ingestion','retrieval_index','investigation')",
    )
    op.create_table(
        "investigations",
        sa.Column(
            "id",
            UUID(as_uuid=False),
            sa.ForeignKey("reviews.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column(
            "owner_id",
            UUID(as_uuid=False),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("profile_id", UUID(as_uuid=False), nullable=False),
        sa.Column(
            "repository_id",
            UUID(as_uuid=False),
            sa.ForeignKey("repositories.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("revision_id", UUID(as_uuid=False), nullable=False),
        sa.Column("request", sa.JSON(), nullable=False),
        sa.Column("state", sa.JSON(), nullable=False),
        sa.Column("snapshot_digest", sa.String(64), nullable=False),
        sa.Column("deadline_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("outcome", sa.String(40), nullable=True),
        sa.Column("final", sa.JSON(none_as_null=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "json_typeof(state) = 'object' AND state->'version' IS NOT NULL AND json_typeof(state->'version') = 'number' AND state->>'version' = '1' AND state->'events' IS NOT NULL AND json_typeof(state->'events') = 'object' AND state->'attempts' IS NOT NULL AND json_typeof(state->'attempts') = 'array' AND json_array_length(state->'attempts') <= 6 AND octet_length(state::text) <= 131072",
            name="ck_investigation_state",
        ),
        sa.CheckConstraint(
            "octet_length(request::text) <= 8192 AND (final IS NULL OR octet_length(final::text) <= 131072)",
            name="ck_investigation_sizes",
        ),
        sa.CheckConstraint("deadline_at > created_at", name="ck_investigation_deadline"),
        sa.CheckConstraint(
            "outcome IS NULL OR outcome IN ('completed','insufficient_evidence','step_limit','provider_failure','validation_failure','invalid_decision','cancelled','timeout','scope_changed','unknown_provider_outcome')",
            name="ck_investigation_outcome",
        ),
        sa.ForeignKeyConstraint(
            ["id", "revision_id", "profile_id"],
            ["reviews.id", "reviews.revision_id", "reviews.profile_id"],
            ondelete="CASCADE",
        ),
    )
    op.create_index("ix_investigation_owner_created", "investigations", ["owner_id", "created_at"])
    op.create_index("ix_investigation_revision", "investigations", ["revision_id"])


def downgrade():
    if (
        op.get_bind()
        .execute(sa.text("SELECT count(*) FROM reviews WHERE job_type='investigation'"))
        .scalar()
    ):
        raise RuntimeError("Investigation records exist; downgrade would delete history")
    op.drop_table("investigations")
    op.drop_constraint("ck_review_job_type", "reviews", type_="check")
    op.create_check_constraint(
        "ck_review_job_type",
        "reviews",
        "job_type IN ('review','github_ingestion','retrieval_index')",
    )
