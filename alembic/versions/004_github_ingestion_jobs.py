"""Extend existing review jobs for bounded GitHub ingestion."""

from alembic import op
import sqlalchemy as sa

revision = "004"
down_revision = "003"
branch_labels = None
depends_on = None


def upgrade():
    # Inline ADD COLUMN checks avoid SQLite table recreation/cascading deletion of evidence.
    op.add_column(
        "reviews",
        sa.Column(
            "job_type",
            sa.String(30),
            sa.CheckConstraint(
                "job_type IN ('review','github_ingestion')", name="ck_review_job_type"
            ),
            nullable=False,
            server_default="review",
        ),
    )
    op.add_column(
        "reviews",
        sa.Column(
            "requested_repository",
            sa.String(255),
            sa.CheckConstraint(
                "job_type = 'review' OR (requested_repository IS NOT NULL AND sections IS NULL AND overall_score IS NULL)",
                name="ck_ingestion_job",
            ),
            nullable=True,
        ),
    )
    op.add_column("reviews", sa.Column("ingestion_result", sa.JSON(), nullable=True))
    op.create_index(
        "uq_active_ingestion_profile",
        "reviews",
        ["profile_id"],
        unique=True,
        postgresql_where=sa.text(
            "job_type = 'github_ingestion' AND status IN ('pending','processing')"
        ),
        sqlite_where=sa.text(
            "job_type = 'github_ingestion' AND status IN ('pending','processing')"
        ),
    )


def downgrade():
    if op.get_bind().dialect.name == "sqlite":
        # 006's safe parent recreation reflects original inline checks as table checks.
        # Preserve dependent rows while explicitly dropping those checks/columns.
        for table in ("findings", "evidence_spans"):
            op.execute(f"CREATE TEMP TABLE phase004_{table} AS SELECT * FROM {table}")
        with op.batch_alter_table("reviews") as batch:
            batch.drop_index("uq_active_ingestion_profile")
            batch.drop_constraint("ck_ingestion_job", type_="check")
            batch.drop_constraint("ck_review_job_type", type_="check")
            for name in ("ingestion_result", "requested_repository", "job_type"):
                batch.drop_column(name)
        for table in ("findings", "evidence_spans"):
            op.execute(f"INSERT INTO {table} SELECT * FROM phase004_{table}")
            op.execute(f"DROP TABLE phase004_{table}")
    else:
        op.drop_index("uq_active_ingestion_profile", table_name="reviews")
        op.drop_column("reviews", "ingestion_result")
        op.drop_column("reviews", "requested_repository")
        op.drop_column("reviews", "job_type")
