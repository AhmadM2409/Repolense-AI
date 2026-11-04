"""Migration 004 must not rebuild reviews and cascade-delete existing evidence."""

from uuid import uuid4
import hashlib
import pytest
from sqlalchemy import text
from alembic import command
from alembic.config import Config
from tests.auth_fixtures import isolated_engine


@pytest.mark.integration
@pytest.mark.asyncio
async def test_004_preserves_existing_revision_sources_and_evidence():
    async with isolated_engine() as engine:
        async with engine.begin() as connection:

            def exercise(conn):
                cfg = Config("alembic.ini")
                cfg.attributes["connection"] = conn
                command.upgrade(cfg, "003")
                uid, pid, repo, rev, source, review, finding, evidence, chunk = [
                    uuid4().hex for _ in range(9)
                ]
                digest = hashlib.sha256(b"x").hexdigest()
                statements = [
                    f"INSERT INTO users(id,email,hashed_password,created_at,updated_at,is_active) VALUES('{uid}','migration-fixture@example.com','fixture',CURRENT_TIMESTAMP,CURRENT_TIMESTAMP,TRUE)",
                    f"INSERT INTO profiles(id,user_id,created_at,updated_at) VALUES('{pid}','{uid}',CURRENT_TIMESTAMP,CURRENT_TIMESTAMP)",
                    f"INSERT INTO repositories(id,profile_id,provider,full_name) VALUES('{repo}','{pid}','github','fixture/repo')",
                    f"INSERT INTO repository_revisions(id,profile_id,repository_id,commit_sha) VALUES('{rev}','{pid}','{repo}','{'a' * 40}')",
                    f"INSERT INTO ingested_sources(id,profile_id,source_type,revision_id,path,text,content_hash,chunk_count,ingested_at) VALUES('{source}','{pid}','repo','{rev}','x.py','x','{digest}',1,CURRENT_TIMESTAMP)",
                    f"INSERT INTO reviews(id,profile_id,revision_id,status,created_at,updated_at) VALUES('{review}','{pid}','{rev}','pending',CURRENT_TIMESTAMP,CURRENT_TIMESTAMP)",
                    f"INSERT INTO source_chunks(id,profile_id,source_id,chunk_index,text,line_start,line_end) VALUES('{chunk}','{pid}','{source}',0,'x',1,1)",
                    f"INSERT INTO findings(id,profile_id,review_id,title,detail) VALUES('{finding}','{pid}','{review}','Fixture','Kept')",
                    f"INSERT INTO evidence_spans(id,profile_id,finding_id,source_id,review_id,revision_id,line_start,line_end,quote) VALUES('{evidence}','{pid}','{finding}','{source}','{review}','{rev}',1,1,'x')",
                ]
                for statement in statements:
                    conn.execute(text(statement))
                command.upgrade(cfg, "004")
                assert conn.scalar(text("SELECT job_type FROM reviews")) == "review"
                for table in ("reviews", "source_chunks", "findings", "evidence_spans"):
                    assert conn.scalar(text("SELECT count(*) FROM " + table)) == 1
                command.downgrade(cfg, "003")
                for table in ("reviews", "source_chunks", "findings", "evidence_spans"):
                    assert conn.scalar(text("SELECT count(*) FROM " + table)) == 1
                command.upgrade(cfg, "head")
                assert conn.scalar(text("SELECT quote FROM evidence_spans")) == "x"

            await connection.run_sync(exercise)
