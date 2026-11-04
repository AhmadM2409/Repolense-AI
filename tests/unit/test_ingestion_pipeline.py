"""Real DB replacements for the old fake-session sync ingestion tests."""

import hashlib
from uuid import UUID
from unittest.mock import Mock
import pytest
from fastapi import HTTPException
from sqlalchemy import select, func, text
from sqlalchemy.exc import SQLAlchemyError
from core.models import IngestedSource
from core.models.evidence import SourceChunk
from ingestion.pipeline import IngestionPipeline
from ingestion.chunking.base import Chunk
from ingestion.embeddings.batch_processor import BatchEmbeddingProcessor
from rag.retriever.vector_store import VectorStore

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


class ExternalCollection:
    """External Chroma test double; ORM, parsers and chunk logic stay real."""

    def __init__(self):
        self.rows = {}
        self.calls = 0

    def add(self, ids, embeddings, documents, metadatas):
        self.calls += 1
        for ident, embedding, doc, meta in zip(ids, embeddings, documents, metadatas):
            assert all(isinstance(v, (str, int, float, bool)) for v in meta.values())
            self.rows[ident] = (embedding, doc, meta)

    upsert = add

    def delete(self, where):
        self.rows = {
            key: row for key, row in self.rows.items() if row[2]["source_id"] != where["source_id"]
        }


def make(db):
    collection = ExternalCollection()
    provider = Mock()
    provider.embed.side_effect = lambda texts: [[0.1, 0.2] for _ in texts]
    return IngestionPipeline(collection, db, provider), collection, provider


async def test_first_source_persists_full_hash_and_real_chunks(database, actors):
    async with database() as db:
        pipeline, collection, provider = make(db)
        result = await pipeline.ingest_readme(
            actors["profiles"][0].id,
            "example/project",
            "# Title\nActual content",
            user_id=actors["users"][0].id,
        )
        source = await db.get(IngestedSource, result.source_id)
        assert UUID(result.source_id) and not result.skipped and result.chunk_count == 1
        assert source.content_hash == hashlib.sha256(source.text.encode()).hexdigest()
        assert (
            source.source_metadata["raw_content_hash"]
            == hashlib.sha256(b"# Title\nActual content").hexdigest()
        )
        chunk = await db.scalar(select(SourceChunk).where(SourceChunk.source_id == source.id))
        assert chunk.text == source.text and chunk.line_start == 1
        assert set(collection.rows) == {source.id + "_chunk_0"}


async def test_dedup_changed_content_and_repository_scope(database, actors):
    async with database() as db:
        pipeline, collection, provider = make(db)
        uid, pid = actors["users"][0].id, actors["profiles"][0].id
        first = await pipeline.ingest_readme(pid, "repo", "# Title\nA", user_id=uid)
        same = await pipeline.ingest_readme(pid, "repo", b"# Title\nA", user_id=uid)
        changed = await pipeline.ingest_readme(pid, "repo", "# Title\nB", user_id=uid)
        other = await pipeline.ingest_readme(pid, "other", "# Title\nA", user_id=uid)
        assert same.skipped and same.source_id == first.source_id and same.chunk_count == 0
        assert len({first.source_id, changed.source_id, other.source_id}) == 3
        assert provider.embed.call_count == 3
        assert await db.scalar(select(func.count()).select_from(IngestedSource)) == 3


async def test_profile_scoped_dedup_and_cross_user_rejection(database, actors):
    async with database() as db:
        pipeline, collection, provider = make(db)
        result = []
        for i in (0, 1):
            result.append(
                await pipeline.ingest_readme(
                    actors["profiles"][i].id, "repo", "same text", user_id=actors["users"][i].id
                )
            )
        assert result[0].source_id != result[1].source_id
        with pytest.raises(HTTPException) as denied:
            await pipeline.ingest_readme(
                actors["profiles"][1].id, "repo", "x", user_id=actors["users"][0].id
            )
        assert denied.value.status_code == 404
        assert provider.embed.call_count == 2


async def test_lookup_database_error_fails_closed(database, actors):
    async with database() as db:
        pipeline, collection, provider = make(db)
        cascade = " CASCADE" if db.bind.dialect.name == "postgresql" else ""
        await db.execute(text("DROP TABLE source_chunks" + cascade))
        await db.execute(text("DROP TABLE evidence_spans" + cascade))
        await db.execute(text("DROP TABLE ingested_sources" + cascade))
        await db.commit()
        with pytest.raises(SQLAlchemyError):
            await pipeline.ingest_readme(
                actors["profiles"][0].id, "repo", "x", user_id=actors["users"][0].id
            )
        provider.embed.assert_not_called()
        assert not collection.rows


async def test_database_write_failure_rolls_back_and_compensates_vectors(database, actors):
    async with database() as db:
        pipeline, collection, provider = make(db)
        from tests.database_probes import reject_insert
        await reject_insert(db, "reject_source", "ingested_sources", "fixture failure")
        await db.commit()
        with pytest.raises(SQLAlchemyError):
            await pipeline.ingest_readme(
                actors["profiles"][0].id, "repo", "real content", user_id=actors["users"][0].id
            )
        assert await db.scalar(select(func.count()).select_from(IngestedSource)) == 0
        assert await db.scalar(select(func.count()).select_from(SourceChunk)) == 0
        assert not collection.rows


async def test_external_embedding_failure_has_no_sql_rows(database, actors):
    async with database() as db:
        pipeline, collection, provider = make(db)
        provider.embed.side_effect = RuntimeError("external failure")
        with pytest.raises(RuntimeError):
            await pipeline.ingest_readme(
                actors["profiles"][0].id, "repo", "content", user_id=actors["users"][0].id
            )
        assert await db.scalar(select(func.count()).select_from(IngestedSource)) == 0
        assert not collection.rows


async def test_resume_markdown_bytes_and_repo_metadata(database, actors):
    async with database() as db:
        pipeline, collection, provider = make(db)
        uid, pid = actors["users"][0].id, actors["profiles"][0].id
        result = await pipeline.ingest_resume(pid, b"# Skills\nPython", "resume.md", user_id=uid)
        assert (await db.get(IngestedSource, result.source_id)).filename == "resume.md"
        a = await pipeline.ingest_repo_metadata(
            pid, {"name": "project", "language": "Python", "stargazers_count": 5}, user_id=uid
        )
        b = await pipeline.ingest_repo_metadata(
            pid, {"stargazers_count": 5, "language": "Python", "name": "project"}, user_id=uid
        )
        assert b.skipped and a.source_id == b.source_id
        assert "Stars: 5" in (await db.get(IngestedSource, a.source_id)).text


async def test_vector_store_accepts_actual_ingestion_chunk():
    collection = ExternalCollection()
    client = Mock()
    client.get_collection.return_value = collection
    store = VectorStore.__new__(VectorStore)
    store.client = client
    chunk = Chunk(
        "line",
        {
            "source_id": "source",
            "chunk_index": 0,
            "heading_path": "Title",
            "tags": ["x"],
            "absent": None,
            "line_start": 1,
            "line_end": 1,
        },
    )
    store.add_chunks([(chunk, [0.1, 0.2])], "profile_test")
    batch = BatchEmbeddingProcessor(Mock(), collection)
    batch._store_embedding(chunk, [0.1, 0.2])
    assert set(collection.rows) == {"source_chunk_0"}
    assert collection.rows["source_chunk_0"][2]["tags"] == '["x"]'
    assert "absent" not in collection.rows["source_chunk_0"][2]


async def test_embedding_cardinality_mismatch_rejected():
    collection = ExternalCollection()
    provider = Mock()
    provider.embed.return_value = []
    processor = BatchEmbeddingProcessor(provider, collection)
    with pytest.raises(ValueError, match="number of vectors"):
        processor.process([Chunk("x", {"source_id": "s", "chunk_index": 0})])
    assert not collection.rows
