"""Versioned J3 roadmaps; existing J2 inference budget is reused unchanged."""

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

from alembic import op

revision = "011"
down_revision = "010"
branch_labels = depends_on = None


def upgrade():
    op.create_table(
        "career_roadmaps",
        sa.Column("id", UUID(as_uuid=False), primary_key=True),
        sa.Column(
            "assessment_id",
            UUID(as_uuid=False),
            sa.ForeignKey("career_assessments.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("source_signature", sa.String(64), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("error_category", sa.String(40)),
        sa.Column("result", sa.JSON(), nullable=True),
        sa.Column("operation_token", sa.String(36)),
        sa.Column("lease_until", sa.DateTime(timezone=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("assessment_id", name="uq_roadmap_assessment"),
        sa.CheckConstraint(
            "version=1 AND status IN ('generating','complete','failed','cancelled') AND octet_length(result::text)<=65536",
            name="ck_roadmap_bounds",
        ),
    )
    op.execute("""CREATE FUNCTION career_roadmap_immutable() RETURNS trigger LANGUAGE plpgsql AS $$
    BEGIN
      IF ROW(NEW.assessment_id,NEW.version,NEW.source_signature,NEW.created_at) IS DISTINCT FROM ROW(OLD.assessment_id,OLD.version,OLD.source_signature,OLD.created_at) THEN
        RAISE EXCEPTION 'Roadmap provenance is immutable';
      END IF;
      IF OLD.status IN ('complete','failed','cancelled') AND ROW(NEW.status,NEW.result::text,NEW.error_category,NEW.operation_token,NEW.lease_until) IS DISTINCT FROM ROW(OLD.status,OLD.result::text,OLD.error_category,OLD.operation_token,OLD.lease_until) THEN
        RAISE EXCEPTION 'Terminal roadmap is immutable';
      END IF;
      RETURN NEW;
    END $$""")
    op.execute(
        "CREATE TRIGGER career_roadmap_scope BEFORE UPDATE ON career_roadmaps FOR EACH ROW EXECUTE FUNCTION career_roadmap_immutable()"
    )


def downgrade():
    if op.get_bind().execute(sa.text("SELECT count(*) FROM career_roadmaps")).scalar():
        raise RuntimeError("Roadmap history exists; refusing destructive downgrade")
    op.drop_table("career_roadmaps")
    op.execute("DROP FUNCTION career_roadmap_immutable()")
