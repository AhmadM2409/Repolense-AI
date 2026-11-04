"""Allow 12 recorded attempts per assessment (10 matching + 2 roadmap); no data change.

Only the per-row bound on `career_assessments.calls` is widened from 10 to 12. Owner (40) and
aggregate (100) limits, the per-minute window, existing rows, receipts and triggers are unchanged.
"""

import sqlalchemy as sa

from alembic import op

revision = "012"
down_revision = "011"
branch_labels = depends_on = None

_OTHER_BOUNDS = (
    "json_array_length(requirements)<=4 AND json_array_length(matches)<=4 "
    "AND octet_length(snapshot::text)<=32768 AND octet_length(requirements::text)<=16384 "
    "AND octet_length(matches::text)<=65536 AND octet_length(calls::text)<=16384"
)


def _bounds(calls):
    return f"json_array_length(calls)<={calls} AND {_OTHER_BOUNDS}"


def upgrade():
    op.drop_constraint("ck_assessment_bounds", "career_assessments", type_="check")
    op.create_check_constraint("ck_assessment_bounds", "career_assessments", _bounds(12))


def downgrade():
    over = (
        op.get_bind()
        .execute(
            sa.text("SELECT count(*) FROM career_assessments WHERE json_array_length(calls) > 10")
        )
        .scalar()
    )
    if over:
        raise RuntimeError("Assessments with more than 10 attempts exist; refusing downgrade")
    op.drop_constraint("ck_assessment_bounds", "career_assessments", type_="check")
    op.create_check_constraint("ck_assessment_bounds", "career_assessments", _bounds(10))
