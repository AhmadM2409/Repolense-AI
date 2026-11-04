"""Durable scheduling on existing Review rows, preserving historical results."""

import sqlalchemy as sa

from alembic import op

revision = "006"
down_revision = "005"
branch_labels = None
depends_on = None

COLUMNS = [
    sa.Column("attempt_count", sa.Integer(), nullable=False, server_default="0"),
    sa.Column("max_attempts", sa.Integer(), nullable=False, server_default="3"),
    sa.Column("worker_id", sa.String(128)),
    sa.Column("execution_token", sa.String(36)),
    *[
        sa.Column(name, sa.DateTime(timezone=True))
        for name in (
            "lease_expires_at",
            "heartbeat_at",
            "retry_at",
            "started_at",
            "completed_at",
            "cancel_requested_at",
        )
    ],
    sa.Column("error_category", sa.String(64)),
    sa.Column("progress_stage", sa.String(64), nullable=False, server_default="queued"),
    sa.Column("progress_pct", sa.Integer()),
]
CHECKS = {
    "ck_job_attempts": "attempt_count >= 0 AND max_attempts BETWEEN 1 AND 10 AND attempt_count <= max_attempts",
    "ck_job_progress": "progress_pct IS NULL OR progress_pct BETWEEN 0 AND 100",
    "ck_job_worker_token": "(worker_id IS NULL) = (execution_token IS NULL)",
}


def backup():
    # SQLite batch parent-table recreation cascades to children. Preserve them transactionally.
    if op.get_bind().dialect.name == "sqlite":
        for table in ("findings", "evidence_spans"):
            op.execute(f"CREATE TEMP TABLE phase006_{table} AS SELECT * FROM {table}")


def restore():
    if op.get_bind().dialect.name == "sqlite":
        for table in ("findings", "evidence_spans"):
            op.execute(f"INSERT INTO {table} SELECT * FROM phase006_{table}")
            op.execute(f"DROP TABLE phase006_{table}")


def upgrade():
    backup()
    with op.batch_alter_table("reviews") as batch:
        for column in COLUMNS:
            batch.add_column(column.copy())
        batch.drop_constraint("ck_review_status", type_="check")
        batch.create_check_constraint(
            "ck_review_status", "status IN ('pending','processing','complete','failed','cancelled')"
        )
        for name, condition in CHECKS.items():
            batch.create_check_constraint(name, condition)
        batch.create_index("ix_job_claimable", ["status", "retry_at", "created_at"])
        batch.create_index("ix_job_expired", ["status", "lease_expires_at"])
    restore()
    # Existing supported processing jobs have no live durable owner. Worker recovery will claim them.
    op.execute("UPDATE reviews SET lease_expires_at=CURRENT_TIMESTAMP WHERE status='processing'")
    op.execute(
        "UPDATE reviews SET progress_stage=status, progress_pct=CASE WHEN status='complete' THEN 100 ELSE NULL END WHERE status IN ('complete','failed')"
    )


def downgrade():
    bind = op.get_bind()
    if not op.get_context().as_sql and bind.scalar(
        sa.text(
            "SELECT count(*) FROM reviews WHERE status='cancelled' OR (status='processing' AND worker_id IS NOT NULL)"
        )
    ):
        raise RuntimeError(
            "Downgrade requires stopped workers and no cancelled jobs; preserve records instead of coercing statuses"
        )
    backup()
    with op.batch_alter_table("reviews") as batch:
        batch.drop_index("ix_job_expired")
        batch.drop_index("ix_job_claimable")
        for name in CHECKS:
            batch.drop_constraint(name, type_="check")
        batch.drop_constraint("ck_review_status", type_="check")
        batch.create_check_constraint(
            "ck_review_status", "status IN ('pending','processing','complete','failed')"
        )
        for column in reversed(COLUMNS):
            batch.drop_column(column.name)
    restore()
