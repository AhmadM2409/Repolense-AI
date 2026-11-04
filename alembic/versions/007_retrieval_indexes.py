"""Versioned, fenced PostgreSQL pgvector indexing; existing history is unchanged."""

import re

import sqlalchemy as sa
from sqlalchemy.types import UserDefinedType

from alembic import op

revision = "007"
down_revision = "006"
branch_labels = None
depends_on = None


class FrozenVector(UserDefinedType):
    def get_col_spec(self, **kw):
        return "public.vector"


def _ddl(script):
    # asyncpg prepares one SQL statement at a time; never split inside PL/pgSQL bodies.
    for statement in re.split(r"(?m)(?=^    CREATE (?:FUNCTION|(?:CONSTRAINT )?TRIGGER))", script):
        if statement.strip():
            op.execute(statement.strip())


def upgrade():
    if op.get_bind().dialect.name != "postgresql":
        raise RuntimeError(
            "Migration 007 requires PostgreSQL with pgvector; no fallback vector backend"
        )
    op.execute("CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public")
    op.create_unique_constraint(
        "uq_chunk_vector_scope", "source_chunks", ["id", "source_id", "profile_id"]
    )
    op.drop_constraint("ck_review_job_type", "reviews", type_="check")
    op.create_check_constraint(
        "ck_review_job_type",
        "reviews",
        "job_type IN ('review','github_ingestion','retrieval_index')",
    )
    uuid = sa.dialects.postgresql.UUID(as_uuid=False)
    op.create_table(
        "retrieval_indexes",
        sa.Column("id", uuid, primary_key=True),
        sa.Column("profile_id", uuid, nullable=False),
        sa.Column("revision_id", uuid, nullable=False),
        sa.Column("ingestion_id", uuid, nullable=False),
        sa.Column("job_id", uuid, nullable=False, unique=True),
        sa.Column("provider", sa.String(20), nullable=False),
        sa.Column("model", sa.String(128), nullable=False),
        sa.Column("dimensions", sa.Integer(), nullable=False),
        sa.Column("metric", sa.String(20), nullable=False, server_default="cosine"),
        sa.Column("config_fingerprint", sa.String(64), nullable=False),
        sa.Column("config", sa.JSON(), nullable=False),
        sa.Column("manifest_fingerprint", sa.String(64), nullable=False),
        sa.Column("chunk_manifest", sa.JSON(), nullable=False),
        sa.Column("expected_chunks", sa.Integer(), nullable=False),
        sa.Column("state", sa.String(20), nullable=False, server_default="pending"),
        sa.Column("attempt_token", sa.String(36)),
        sa.Column("reserved_requests", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("published_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("published_at", sa.DateTime(timezone=True)),
        sa.ForeignKeyConstraint(
            ["revision_id", "profile_id"],
            ["repository_revisions.id", "repository_revisions.profile_id"],
            ondelete="CASCADE",
            name="fk_index_revision",
        ),
        sa.ForeignKeyConstraint(
            ["job_id", "revision_id", "profile_id"],
            ["reviews.id", "reviews.revision_id", "reviews.profile_id"],
            ondelete="CASCADE",
            name="fk_index_job",
        ),
        sa.ForeignKeyConstraint(
            ["ingestion_id", "revision_id", "profile_id"],
            ["reviews.id", "reviews.revision_id", "reviews.profile_id"],
            ondelete="CASCADE",
            name="fk_index_ingestion",
        ),
        sa.UniqueConstraint(
            "id",
            "revision_id",
            "profile_id",
            "config_fingerprint",
            "dimensions",
            name="uq_index_vector_scope",
        ),
        sa.CheckConstraint(
            "state IN ('pending','building','ready','failed','cancelled')", name="ck_index_state"
        ),
        sa.CheckConstraint("metric = 'cosine'", name="ck_index_metric"),
        sa.CheckConstraint(
            "(provider='lmstudio' AND model='text-embedding-nomic-embed-text-v1.5' AND dimensions=768) OR (provider='mistral' AND model='mistral-embed' AND dimensions=1024)",
            name="ck_index_model",
        ),
        sa.CheckConstraint(
            "expected_chunks BETWEEN 0 AND 2000 AND published_count BETWEEN 0 AND expected_chunks AND reserved_requests BETWEEN 0 AND 512",
            name="ck_index_counts",
        ),
        sa.CheckConstraint(
            "length(config_fingerprint)=64 AND length(manifest_fingerprint)=64",
            name="ck_index_digests",
        ),
        sa.CheckConstraint(
            "(state='ready') = (published_at IS NOT NULL) AND (state<>'ready' OR published_count=expected_chunks)",
            name="ck_index_published",
        ),
        sa.CheckConstraint(
            "json_array_length(chunk_manifest)=expected_chunks AND octet_length(chunk_manifest::text)<=1048576",
            name="ck_index_manifest",
        ),
    )
    op.create_index(
        "ix_index_revision_config",
        "retrieval_indexes",
        ["profile_id", "revision_id", "config_fingerprint"],
    )
    op.create_index(
        "uq_index_active",
        "retrieval_indexes",
        ["profile_id", "revision_id", "config_fingerprint"],
        unique=True,
        postgresql_where=sa.text("state IN ('pending','building','ready')"),
    )
    op.create_table(
        "chunk_embeddings",
        sa.Column("index_id", uuid, primary_key=True),
        sa.Column("chunk_id", uuid, primary_key=True),
        sa.Column("source_id", uuid, nullable=False),
        sa.Column("revision_id", uuid, nullable=False),
        sa.Column("profile_id", uuid, nullable=False),
        sa.Column("config_fingerprint", sa.String(64), nullable=False),
        sa.Column("dimensions", sa.Integer(), nullable=False),
        sa.Column("attempt_token", sa.String(36), nullable=False),
        sa.Column("input_digest", sa.String(64), nullable=False),
        sa.Column("embedding", FrozenVector(), nullable=False),
        sa.ForeignKeyConstraint(
            ["index_id", "revision_id", "profile_id", "config_fingerprint", "dimensions"],
            [
                "retrieval_indexes.id",
                "retrieval_indexes.revision_id",
                "retrieval_indexes.profile_id",
                "retrieval_indexes.config_fingerprint",
                "retrieval_indexes.dimensions",
            ],
            ondelete="CASCADE",
            name="fk_vector_index",
        ),
        sa.ForeignKeyConstraint(
            ["chunk_id", "source_id", "profile_id"],
            ["source_chunks.id", "source_chunks.source_id", "source_chunks.profile_id"],
            ondelete="CASCADE",
            name="fk_vector_chunk",
        ),
        sa.ForeignKeyConstraint(
            ["source_id", "revision_id", "profile_id"],
            ["ingested_sources.id", "ingested_sources.revision_id", "ingested_sources.profile_id"],
            ondelete="CASCADE",
            name="fk_vector_source_revision",
        ),
        sa.CheckConstraint(
            "public.vector_dims(embedding)=dimensions AND public.vector_norm(embedding)>0",
            name="ck_vector_values",
        ),
        sa.CheckConstraint("length(input_digest)=64", name="ck_vector_digest"),
    )
    op.create_index(
        "ix_vector_revision", "chunk_embeddings", ["profile_id", "revision_id", "index_id"]
    )
    _ddl(r"""
    CREATE FUNCTION retrieval_identity_guard() RETURNS trigger LANGUAGE plpgsql AS $$
    BEGIN
      IF ROW(OLD.profile_id,OLD.revision_id,OLD.ingestion_id,OLD.job_id,OLD.provider,OLD.model,
             OLD.dimensions,OLD.metric,OLD.config_fingerprint,OLD.config::text,
             OLD.manifest_fingerprint,OLD.chunk_manifest::text,OLD.expected_chunks)
         IS DISTINCT FROM
         ROW(NEW.profile_id,NEW.revision_id,NEW.ingestion_id,NEW.job_id,NEW.provider,NEW.model,
             NEW.dimensions,NEW.metric,NEW.config_fingerprint,NEW.config::text,
             NEW.manifest_fingerprint,NEW.chunk_manifest::text,NEW.expected_chunks)
      THEN RAISE EXCEPTION 'immutable retrieval identity'; END IF;
      RETURN NEW;
    END; $$;
    CREATE TRIGGER retrieval_identity BEFORE UPDATE ON retrieval_indexes
      FOR EACH ROW EXECUTE FUNCTION retrieval_identity_guard();
    """)
    _ddl(r"""
    CREATE FUNCTION retrieval_ready_guard() RETURNS trigger LANGUAGE plpgsql AS $$
    DECLARE g retrieval_indexes%ROWTYPE; n bigint; j reviews%ROWTYPE; s reviews%ROWTYPE;
    BEGIN
      SELECT * INTO g FROM retrieval_indexes WHERE id=NEW.id;
      IF NOT FOUND OR g.state<>'ready' THEN RETURN NULL; END IF;
      SELECT * INTO j FROM reviews WHERE id=g.job_id;
      SELECT * INTO s FROM reviews WHERE id=g.ingestion_id;
      IF j.status<>'complete' OR j.job_type<>'retrieval_index'
          OR j.execution_token IS DISTINCT FROM g.attempt_token
          OR g.attempt_token IS NULL OR j.lease_expires_at IS NULL
          OR j.lease_expires_at<=clock_timestamp() OR j.cancel_requested_at IS NOT NULL
          OR NOT EXISTS (SELECT 1 FROM profiles p JOIN users u ON u.id=p.user_id
                         WHERE p.id=g.profile_id AND u.is_active)
          OR (s.ingestion_result->>'commit_sha') IS DISTINCT FROM (
              SELECT commit_sha FROM repository_revisions WHERE id=g.revision_id)
          OR s.status<>'complete' OR s.job_type<>'github_ingestion'
          OR (s.ingestion_result->>'verified') IS DISTINCT FROM 'true'
          OR (s.ingestion_result->>'policy') IS DISTINCT FROM 'github-text-v1'
      THEN RAISE EXCEPTION 'retrieval publication job or provenance invalid'; END IF;
      SELECT count(*) INTO n FROM chunk_embeddings WHERE index_id=g.id;
      IF n<>g.expected_chunks THEN RAISE EXCEPTION 'retrieval coverage incomplete'; END IF;
      SELECT count(*) INTO n FROM source_chunks c JOIN ingested_sources d
        ON d.id=c.source_id AND d.profile_id=c.profile_id
        WHERE d.revision_id=g.revision_id AND d.profile_id=g.profile_id;
      IF n<>g.expected_chunks THEN RAISE EXCEPTION 'retrieval canonical coverage changed'; END IF;
      SELECT count(DISTINCT m->>'chunk_id') INTO n FROM json_array_elements(g.chunk_manifest) m;
      IF n<>g.expected_chunks THEN RAISE EXCEPTION 'retrieval manifest duplicate or missing chunk'; END IF;
      SELECT count(*) INTO n FROM json_array_elements(g.chunk_manifest) m
        JOIN chunk_embeddings v ON v.index_id=g.id AND v.chunk_id::text=m->>'chunk_id'
          AND v.source_id::text=m->>'source_id' AND v.input_digest=m->>'input_digest'
        JOIN source_chunks c ON c.id=v.chunk_id AND c.source_id=v.source_id AND c.profile_id=v.profile_id
        WHERE v.attempt_token=g.attempt_token AND v.input_digest =
          encode(sha256(convert_to(g.config_fingerprint || E'\n' ||
            CASE WHEN g.provider='lmstudio' THEN 'search_document: ' ELSE '' END || c.text,'UTF8')),'hex');
      IF n<>g.expected_chunks THEN RAISE EXCEPTION 'retrieval input manifest or fencing invalid'; END IF;
      RETURN NULL;
    END; $$;
    CREATE CONSTRAINT TRIGGER retrieval_ready AFTER INSERT OR UPDATE ON retrieval_indexes
      DEFERRABLE INITIALLY DEFERRED FOR EACH ROW EXECUTE FUNCTION retrieval_ready_guard();
    """)
    _ddl(r"""
    CREATE FUNCTION retrieval_vectors_changed() RETURNS trigger LANGUAGE plpgsql AS $$
    BEGIN
      UPDATE retrieval_indexes SET state='failed',published_at=NULL,published_count=0
        WHERE id=CASE WHEN TG_OP='DELETE' THEN OLD.index_id ELSE NEW.index_id END AND state='ready';
      RETURN NULL;
    END; $$;
    CREATE TRIGGER retrieval_vector_changed AFTER INSERT OR UPDATE OR DELETE ON chunk_embeddings
      FOR EACH ROW EXECUTE FUNCTION retrieval_vectors_changed();

    CREATE FUNCTION retrieval_chunk_changed() RETURNS trigger LANGUAGE plpgsql AS $$
    BEGIN
      IF TG_OP='INSERT' THEN
        UPDATE retrieval_indexes SET state='failed',published_at=NULL,published_count=0
          WHERE state='ready' AND profile_id=NEW.profile_id AND revision_id=(
            SELECT revision_id FROM ingested_sources WHERE id=NEW.source_id);
        RETURN NEW;
      END IF;
      IF TG_OP='UPDATE' AND to_jsonb(NEW) IS NOT DISTINCT FROM to_jsonb(OLD) THEN RETURN NEW; END IF;
      UPDATE retrieval_indexes g SET state='failed',published_at=NULL,published_count=0
        WHERE g.state='ready' AND EXISTS (
          SELECT 1 FROM chunk_embeddings v WHERE v.index_id=g.id AND v.chunk_id=OLD.id);
      IF TG_OP='DELETE' THEN RETURN OLD; END IF;
      RETURN NEW;
    END; $$;
    CREATE TRIGGER retrieval_source_chunk_changed BEFORE INSERT OR UPDATE OR DELETE ON source_chunks
      FOR EACH ROW EXECUTE FUNCTION retrieval_chunk_changed();

    CREATE FUNCTION retrieval_source_changed() RETURNS trigger LANGUAGE plpgsql AS $$
    BEGIN
      IF TG_OP='UPDATE' AND to_jsonb(NEW) IS NOT DISTINCT FROM to_jsonb(OLD) THEN RETURN NEW; END IF;
      IF TG_OP='DELETE' THEN
        UPDATE retrieval_indexes SET state='failed',published_at=NULL,published_count=0
          WHERE state='ready' AND revision_id=OLD.revision_id AND profile_id=OLD.profile_id;
        RETURN OLD;
      END IF;
      UPDATE retrieval_indexes SET state='failed',published_at=NULL,published_count=0
        WHERE state='ready' AND revision_id=NEW.revision_id AND profile_id=NEW.profile_id;
      RETURN NEW;
    END; $$;
    CREATE TRIGGER retrieval_source_changed BEFORE INSERT OR UPDATE OR DELETE ON ingested_sources
      FOR EACH ROW EXECUTE FUNCTION retrieval_source_changed();

    CREATE FUNCTION retrieval_proof_changed() RETURNS trigger LANGUAGE plpgsql AS $$
    BEGIN
      IF NEW.status IS NOT DISTINCT FROM OLD.status
        AND NEW.ingestion_result::text IS NOT DISTINCT FROM OLD.ingestion_result::text
      THEN RETURN NULL; END IF;
      UPDATE retrieval_indexes SET state='failed',published_at=NULL,published_count=0
        WHERE ingestion_id=OLD.id AND state='ready';
      RETURN NULL;
    END; $$;
    CREATE TRIGGER retrieval_ingestion_proof_changed AFTER UPDATE OF ingestion_result,status ON reviews
      FOR EACH ROW WHEN (OLD.job_type='github_ingestion') EXECUTE FUNCTION retrieval_proof_changed();
    """)


def downgrade():
    if not op.get_context().as_sql and op.get_bind().scalar(
        sa.text("SELECT count(*) FROM reviews WHERE job_type='retrieval_index'")
    ):
        raise RuntimeError("Downgrade would destroy indexing history; preserve records instead")
    for trigger, table in [
        ("retrieval_ingestion_proof_changed", "reviews"),
        ("retrieval_source_changed", "ingested_sources"),
        ("retrieval_source_chunk_changed", "source_chunks"),
        ("retrieval_vector_changed", "chunk_embeddings"),
        ("retrieval_ready", "retrieval_indexes"),
        ("retrieval_identity", "retrieval_indexes"),
    ]:
        op.execute(f"DROP TRIGGER {trigger} ON {table}")
    for name in [
        "retrieval_proof_changed",
        "retrieval_source_changed",
        "retrieval_chunk_changed",
        "retrieval_vectors_changed",
        "retrieval_ready_guard",
        "retrieval_identity_guard",
    ]:
        op.execute(f"DROP FUNCTION {name}()")
    op.drop_table("chunk_embeddings")
    op.drop_table("retrieval_indexes")
    op.drop_constraint("uq_chunk_vector_scope", "source_chunks", type_="unique")
    op.drop_constraint("ck_review_job_type", "reviews", type_="check")
    op.create_check_constraint(
        "ck_review_job_type", "reviews", "job_type IN ('review','github_ingestion')"
    )
    # Extension is database-wide and may serve other schemas: deliberately retain it.
