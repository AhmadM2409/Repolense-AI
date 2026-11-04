"""Bounded J2 assessment snapshots and durable aggregate inference budget."""

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

from alembic import op

revision = "010"
down_revision = "009"
branch_labels = depends_on = None


def upgrade():
    op.create_table(
        "career_assessments",
        sa.Column("id", UUID(as_uuid=False), primary_key=True),
        sa.Column(
            "owner_id",
            UUID(as_uuid=False),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("input_id", UUID(as_uuid=False), nullable=False),
        sa.Column("profile_id", UUID(as_uuid=False), nullable=False),
        sa.Column("request_id", UUID(as_uuid=False), nullable=False),
        sa.Column("snapshot", sa.JSON(), nullable=False),
        sa.Column("signature", sa.String(64), nullable=False),
        sa.Column("provider_signature", sa.String(64), nullable=False),
        sa.Column("mode", sa.String(12), nullable=False),
        sa.Column("consent", sa.String(64), nullable=False),
        sa.Column("requirements", sa.JSON(), nullable=False),
        sa.Column("matches", sa.JSON(), nullable=False),
        sa.Column("calls", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("error_category", sa.String(40)),
        sa.Column("operation_token", sa.String(36)),
        sa.Column("lease_until", sa.DateTime(timezone=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["input_id", "profile_id"],
            ["career_inputs.id", "career_inputs.profile_id"],
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint("owner_id", "request_id", name="uq_assessment_request"),
        sa.CheckConstraint(
            "status IN ('extracting','review','matching','complete','failed','cancelled')",
            name="ck_assessment_status",
        ),
        sa.CheckConstraint("mode IN ('keyword','semantic','hybrid')", name="ck_assessment_mode"),
        sa.CheckConstraint(
            "json_array_length(calls)<=10 AND json_array_length(requirements)<=4 AND json_array_length(matches)<=4 AND octet_length(snapshot::text)<=32768 AND octet_length(requirements::text)<=16384 AND octet_length(matches::text)<=65536 AND octet_length(calls::text)<=16384",
            name="ck_assessment_bounds",
        ),
    )
    op.create_table(
        "career_ai_budget",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("spent", sa.Integer(), nullable=False),
        sa.Column("stopped", sa.Boolean(), nullable=False),
        sa.Column("recent", sa.JSON(), nullable=False),
        sa.Column("owners", sa.JSON(), nullable=False),
        sa.CheckConstraint(
            "id=1 AND spent BETWEEN 0 AND 100 AND json_array_length(recent)<=10",
            name="ck_career_budget",
        ),
    )
    op.execute(
        "INSERT INTO career_ai_budget (id,spent,stopped,recent,owners) VALUES (1,0,false,'[]'::json,'{}'::json)"
    )
    op.execute("""CREATE FUNCTION career_assessment_immutable() RETURNS trigger LANGUAGE plpgsql AS $$
    BEGIN
      IF ROW(NEW.owner_id,NEW.input_id,NEW.profile_id,NEW.request_id,NEW.snapshot::text,NEW.signature,NEW.provider_signature,NEW.mode,NEW.consent)
         IS DISTINCT FROM ROW(OLD.owner_id,OLD.input_id,OLD.profile_id,OLD.request_id,OLD.snapshot::text,OLD.signature,OLD.provider_signature,OLD.mode,OLD.consent) THEN
        RAISE EXCEPTION 'Career assessment scope is immutable';
      END IF;
      IF json_array_length(OLD.requirements)>0 AND NEW.requirements::text IS DISTINCT FROM OLD.requirements::text THEN
        RAISE EXCEPTION 'Extracted requirement snapshot is immutable';
      END IF;
      IF json_array_length(NEW.calls)<json_array_length(OLD.calls) OR json_array_length(NEW.matches)<json_array_length(OLD.matches) THEN
        RAISE EXCEPTION 'Assessment history cannot be truncated';
      END IF;
      RETURN NEW;
    END $$""")
    op.execute(
        "CREATE TRIGGER career_assessment_scope BEFORE UPDATE ON career_assessments FOR EACH ROW EXECUTE FUNCTION career_assessment_immutable()"
    )


def downgrade():
    if (
        op.get_bind().execute(sa.text("SELECT count(*) FROM career_assessments")).scalar()
        or op.get_bind().execute(sa.text("SELECT spent FROM career_ai_budget WHERE id=1")).scalar()
    ):
        raise RuntimeError(
            "Career assessment history/budget exists; refusing destructive downgrade"
        )
    op.drop_table("career_assessments")
    op.execute("DROP FUNCTION career_assessment_immutable()")
    op.drop_table("career_ai_budget")
