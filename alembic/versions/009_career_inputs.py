"""J1 input snapshots and selected canonical ingestions."""

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

from alembic import op

revision = "009"
down_revision = "008"
branch_labels = depends_on = None


def upgrade():
    op.create_table(
        "career_inputs",
        sa.Column("id", UUID(as_uuid=False), primary_key=True),
        sa.Column(
            "owner_id",
            UUID(as_uuid=False),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "profile_id",
            UUID(as_uuid=False),
            sa.ForeignKey("profiles.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("account", sa.String(39), nullable=False),
        sa.Column("job_description", sa.Text(), nullable=False),
        sa.Column("job_hash", sa.String(64), nullable=False),
        sa.Column("discovery", sa.JSON(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("id", "profile_id", name="uq_career_scope"),
        sa.CheckConstraint(
            "length(job_hash)=64 AND octet_length(job_description)<=65536 AND length(job_description) BETWEEN 20 AND 20000",
            name="ck_career_job",
        ),
        sa.CheckConstraint("octet_length(discovery::text)<=262144", name="ck_career_discovery"),
    )
    op.create_table(
        "career_selections",
        sa.Column("input_id", UUID(as_uuid=False), primary_key=True),
        sa.Column("position", sa.Integer(), primary_key=True),
        sa.Column("profile_id", UUID(as_uuid=False), nullable=False),
        sa.Column("full_name", sa.String(255), nullable=False),
        sa.Column("job_id", UUID(as_uuid=False)),
        sa.Column("revision_id", UUID(as_uuid=False)),
        sa.Column("commit_sha", sa.String(40)),
        sa.ForeignKeyConstraint(
            ["input_id", "profile_id"],
            ["career_inputs.id", "career_inputs.profile_id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(["job_id", "profile_id"], ["reviews.id", "reviews.profile_id"]),
        sa.ForeignKeyConstraint(
            ["revision_id", "profile_id"],
            ["repository_revisions.id", "repository_revisions.profile_id"],
        ),
        sa.ForeignKeyConstraint(
            ["job_id", "revision_id", "profile_id"],
            ["reviews.id", "reviews.revision_id", "reviews.profile_id"],
        ),
        sa.UniqueConstraint("input_id", "full_name", name="uq_career_selection"),
        sa.CheckConstraint("position BETWEEN 0 AND 9", name="ck_career_position"),
        sa.CheckConstraint(
            "(revision_id IS NULL) = (commit_sha IS NULL) AND (revision_id IS NULL OR job_id IS NOT NULL)",
            name="ck_career_pin",
        ),
    )

    op.execute("""CREATE FUNCTION career_input_immutable() RETURNS trigger LANGUAGE plpgsql AS $$
    BEGIN
      IF ROW(NEW.owner_id,NEW.profile_id,NEW.account,NEW.job_description,NEW.job_hash,NEW.discovery::text)
         IS DISTINCT FROM ROW(OLD.owner_id,OLD.profile_id,OLD.account,OLD.job_description,OLD.job_hash,OLD.discovery::text) THEN
        RAISE EXCEPTION 'Career input snapshot is immutable';
      END IF;
      RETURN NEW;
    END $$""")
    op.execute(
        "CREATE TRIGGER career_snapshot_immutable BEFORE UPDATE ON career_inputs FOR EACH ROW EXECUTE FUNCTION career_input_immutable()"
    )


def downgrade():
    if op.get_bind().execute(sa.text("SELECT count(*) FROM career_inputs")).scalar():
        raise RuntimeError("Career input history exists; refusing destructive downgrade")
    op.drop_table("career_selections")
    op.drop_table("career_inputs")
    op.execute("DROP FUNCTION career_input_immutable()")
