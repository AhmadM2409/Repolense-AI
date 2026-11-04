"""Deterministic analysis metadata on existing reviews/findings; no historical relabelling."""

import sqlalchemy as sa

from alembic import op

revision = "005"
down_revision = "004"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("reviews", sa.Column("analysis_result", sa.JSON(), nullable=True))
    op.add_column(
        "findings", sa.Column("rule_id", sa.String(64), nullable=False, server_default="legacy")
    )
    op.add_column(
        "findings",
        sa.Column(
            "severity",
            sa.String(16),
            sa.CheckConstraint(
                "severity IN ('info','low','medium','high','critical')", name="ck_finding_severity"
            ),
            nullable=False,
            server_default="info",
        ),
    )
    op.add_column(
        "findings",
        sa.Column(
            "category",
            sa.String(32),
            sa.CheckConstraint(
                "category IN ('security','maintainability','configuration')",
                name="ck_finding_category",
            ),
            nullable=False,
            server_default="maintainability",
        ),
    )
    op.add_column(
        "findings",
        sa.Column(
            "certainty",
            sa.String(16),
            sa.CheckConstraint(
                "certainty IN ('observation','potential')", name="ck_finding_certainty"
            ),
            nullable=False,
            server_default="observation",
        ),
    )
    op.add_column("findings", sa.Column("fingerprint", sa.String(64), nullable=True))
    op.create_index("uq_finding_fingerprint", "findings", ["review_id", "fingerprint"], unique=True)
    op.create_index(
        "uq_active_analysis_profile",
        "reviews",
        ["profile_id"],
        unique=True,
        postgresql_where=sa.text(
            "job_type='review' AND report_kind='evidence' AND status IN ('pending','processing')"
        ),
        sqlite_where=sa.text(
            "job_type='review' AND report_kind='evidence' AND status IN ('pending','processing')"
        ),
    )


def downgrade():
    op.drop_index("uq_active_analysis_profile", table_name="reviews")
    op.drop_index("uq_finding_fingerprint", table_name="findings")
    for name in ("fingerprint", "certainty", "category", "severity", "rule_id"):
        op.drop_column("findings", name)
    op.drop_column("reviews", "analysis_result")
