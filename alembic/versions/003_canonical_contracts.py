"""Canonical repository, revision, source and evidence contracts.

Existing profile/review IDs and historical output are retained.
"""

from alembic import op
import sqlalchemy as sa

revision = "003"
down_revision = "002"
branch_labels = None
depends_on = None


def upgrade():
    # Frozen schema operations; no import of runtime model metadata.
    op.create_table(
        "repositories",
        sa.Column("id", sa.UUID(as_uuid=False), nullable=False),
        sa.Column("profile_id", sa.UUID(as_uuid=False), nullable=False),
        sa.Column("provider", sa.String(length=20), server_default="github", nullable=False),
        sa.Column("full_name", sa.String(length=255), nullable=False),
        sa.CheckConstraint("provider = 'github'", name="ck_repository_provider"),
        sa.ForeignKeyConstraint(["profile_id"], ["profiles.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("id", "profile_id", name="uq_repositories_scope"),
        sa.UniqueConstraint("profile_id", "provider", "full_name", name="uq_repositories_identity"),
    )
    op.create_table(
        "repository_revisions",
        sa.Column("id", sa.UUID(as_uuid=False), nullable=False),
        sa.Column("profile_id", sa.UUID(as_uuid=False), nullable=False),
        sa.Column("repository_id", sa.UUID(as_uuid=False), nullable=False),
        sa.Column("commit_sha", sa.String(length=40), nullable=False),
        sa.CheckConstraint(
            "length(commit_sha) = 40 AND commit_sha = lower(commit_sha) AND length(replace(replace(replace(replace(replace(replace(replace(replace(replace(replace(replace(replace(replace(replace(replace(replace(commit_sha, '0', ''), '1', ''), '2', ''), '3', ''), '4', ''), '5', ''), '6', ''), '7', ''), '8', ''), '9', ''), 'a', ''), 'b', ''), 'c', ''), 'd', ''), 'e', ''), 'f', '')) = 0",
            name="ck_revision_sha",
        ),
        sa.ForeignKeyConstraint(
            ["repository_id", "profile_id"],
            ["repositories.id", "repositories.profile_id"],
            name="fk_revision_repository_scope",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("id", "profile_id", name="uq_revision_scope"),
        sa.UniqueConstraint("repository_id", "commit_sha", name="uq_revision_sha"),
    )
    with op.batch_alter_table("ingested_sources", schema=None) as batch_op:
        batch_op.add_column(sa.Column("raw_data", sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column("text", sa.Text(), nullable=True))
        batch_op.add_column(sa.Column("source_metadata", sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column("revision_id", sa.UUID(as_uuid=False), nullable=True))
        batch_op.add_column(sa.Column("path", sa.String(length=1024), nullable=True))
        batch_op.create_check_constraint(
            "ck_source_revision_path",
            "(revision_id IS NULL AND path IS NULL) OR (revision_id IS NOT NULL AND path IS NOT NULL)",
        )
        batch_op.create_check_constraint("ck_source_chunk_count", "chunk_count >= 0")
        batch_op.create_unique_constraint("uq_source_scope", ["id", "profile_id"])
        batch_op.create_unique_constraint("uq_source_revision_path", ["profile_id", "revision_id", "path"])
        batch_op.create_unique_constraint(
            "uq_source_revision_scope", ["id", "revision_id", "profile_id"]
        )
        batch_op.create_foreign_key(
            "fk_source_revision_scope",
            "repository_revisions",
            ["revision_id", "profile_id"],
            ["id", "profile_id"],
            ondelete="CASCADE",
        )

    with op.batch_alter_table("reviews", schema=None) as batch_op:
        batch_op.add_column(sa.Column("revision_id", sa.UUID(as_uuid=False), nullable=True))
        batch_op.add_column(
            sa.Column(
                "report_kind",
                sa.String(length=30),
                server_default="legacy_unverified",
                nullable=False,
            )
        )
        batch_op.create_check_constraint(
            "ck_review_status", "status IN ('pending', 'processing', 'complete', 'failed')"
        )
        batch_op.create_check_constraint(
            "ck_review_kind", "report_kind IN ('legacy_unverified', 'demo', 'evidence')"
        )
        batch_op.create_unique_constraint("uq_review_scope", ["id", "profile_id"])
        batch_op.create_unique_constraint(
            "uq_review_revision_scope", ["id", "revision_id", "profile_id"]
        )
        batch_op.create_foreign_key(
            "fk_review_revision_scope",
            "repository_revisions",
            ["revision_id", "profile_id"],
            ["id", "profile_id"],
            ondelete="CASCADE",
        )

    op.create_table(
        "findings",
        sa.Column("id", sa.UUID(as_uuid=False), nullable=False),
        sa.Column("profile_id", sa.UUID(as_uuid=False), nullable=False),
        sa.Column("review_id", sa.UUID(as_uuid=False), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("detail", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(
            ["review_id", "profile_id"],
            ["reviews.id", "reviews.profile_id"],
            name="fk_finding_review_scope",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("id", "profile_id", name="uq_finding_scope"),
        sa.UniqueConstraint("id", "review_id", "profile_id", name="uq_finding_review_scope"),
    )
    op.create_table(
        "source_chunks",
        sa.Column("id", sa.UUID(as_uuid=False), nullable=False),
        sa.Column("profile_id", sa.UUID(as_uuid=False), nullable=False),
        sa.Column("source_id", sa.UUID(as_uuid=False), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("line_start", sa.Integer(), nullable=False),
        sa.Column("line_end", sa.Integer(), nullable=False),
        sa.CheckConstraint(
            "chunk_index >= 0 AND line_start >= 1 AND line_end >= line_start", name="ck_chunk_range"
        ),
        sa.ForeignKeyConstraint(
            ["source_id", "profile_id"],
            ["ingested_sources.id", "ingested_sources.profile_id"],
            name="fk_chunk_source_scope",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("source_id", "chunk_index", name="uq_chunk_position"),
    )
    op.create_table(
        "evidence_spans",
        sa.Column("id", sa.UUID(as_uuid=False), nullable=False),
        sa.Column("profile_id", sa.UUID(as_uuid=False), nullable=False),
        sa.Column("finding_id", sa.UUID(as_uuid=False), nullable=False),
        sa.Column("source_id", sa.UUID(as_uuid=False), nullable=False),
        sa.Column("review_id", sa.UUID(as_uuid=False), nullable=False),
        sa.Column("revision_id", sa.UUID(as_uuid=False), nullable=False),
        sa.Column("line_start", sa.Integer(), nullable=False),
        sa.Column("line_end", sa.Integer(), nullable=False),
        sa.Column("quote", sa.Text(), nullable=False),
        sa.CheckConstraint("line_start >= 1 AND line_end >= line_start", name="ck_evidence_range"),
        sa.ForeignKeyConstraint(
            ["finding_id", "review_id", "profile_id"],
            ["findings.id", "findings.review_id", "findings.profile_id"],
            name="fk_evidence_finding_scope",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["source_id", "revision_id", "profile_id"],
            ["ingested_sources.id", "ingested_sources.revision_id", "ingested_sources.profile_id"],
            name="fk_evidence_source_scope",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["review_id", "revision_id", "profile_id"],
            ["reviews.id", "reviews.revision_id", "reviews.profile_id"],
            name="fk_evidence_review_revision",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    _install_immutability()
    # ### end Alembic commands ###


def downgrade():
    _remove_immutability()
    op.drop_table("evidence_spans")
    op.drop_table("source_chunks")
    op.drop_table("findings")
    # Frozen schema operations; no import of runtime model metadata.
    with op.batch_alter_table("reviews", schema=None) as batch_op:
        batch_op.drop_constraint("ck_review_status", type_="check")
        batch_op.drop_constraint("ck_review_kind", type_="check")
        batch_op.drop_constraint("fk_review_revision_scope", type_="foreignkey")
        batch_op.drop_constraint("uq_review_revision_scope", type_="unique")
        batch_op.drop_constraint("uq_review_scope", type_="unique")
        batch_op.drop_column("report_kind")
        batch_op.drop_column("revision_id")

    with op.batch_alter_table("ingested_sources", schema=None) as batch_op:
        batch_op.drop_constraint("ck_source_revision_path", type_="check")
        batch_op.drop_constraint("ck_source_chunk_count", type_="check")
        batch_op.drop_constraint("fk_source_revision_scope", type_="foreignkey")
        batch_op.drop_constraint("uq_source_revision_scope", type_="unique")
        batch_op.drop_constraint("uq_source_scope", type_="unique")
        batch_op.drop_constraint("uq_source_revision_path", type_="unique")
        batch_op.drop_column("path")
        batch_op.drop_column("revision_id")
        batch_op.drop_column("source_metadata")
        batch_op.drop_column("text")
        batch_op.drop_column("raw_data")

    op.drop_table("repository_revisions")
    op.drop_table("repositories")
    # ### end Alembic commands ###


def _install_immutability():
    dialect = op.get_bind().dialect.name
    for table, fields, condition in (
        ("repositories", ("id", "profile_id", "provider", "full_name"), "TRUE"),
        ("repository_revisions", ("id", "profile_id", "repository_id", "commit_sha"), "TRUE"),
        (
            "ingested_sources",
            ("id", "profile_id", "revision_id", "path", "text", "content_hash"),
            "OLD.revision_id IS NOT NULL",
        ),
    ):
        if dialect == "sqlite":
            changed = " OR ".join("OLD." + field + " IS NOT NEW." + field for field in fields)
            op.execute(
                "CREATE TRIGGER immutable_"
                + table
                + " BEFORE UPDATE ON "
                + table
                + " WHEN ("
                + condition
                + ") AND ("
                + changed
                + ") BEGIN SELECT RAISE(ABORT, 'immutable source/revision'); END"
            )
        elif dialect == "postgresql":
            old = ", ".join("OLD." + field for field in fields)
            new = ", ".join("NEW." + field for field in fields)
            op.execute(
                "CREATE FUNCTION immutable_" + table + "() RETURNS trigger LANGUAGE plpgsql AS $$ "
                "BEGIN IF ("
                + condition
                + ") AND ROW("
                + old
                + ") IS DISTINCT FROM ROW("
                + new
                + ") THEN RAISE EXCEPTION 'immutable source/revision'; END IF; RETURN NEW; END; $$"
            )
            op.execute(
                "CREATE TRIGGER immutable_"
                + table
                + " BEFORE UPDATE ON "
                + table
                + " FOR EACH ROW EXECUTE FUNCTION immutable_"
                + table
                + "()"
            )


def _remove_immutability():
    dialect = op.get_bind().dialect.name
    for table in ("ingested_sources", "repository_revisions", "repositories"):
        if dialect == "postgresql":
            op.execute("DROP TRIGGER immutable_" + table + " ON " + table)
            op.execute("DROP FUNCTION immutable_" + table + "()")
        elif dialect == "sqlite":
            op.execute("DROP TRIGGER immutable_" + table)
