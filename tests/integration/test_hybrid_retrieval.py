"""Real PostgreSQL hybrid composition; deterministic external embedding doubles only."""

import asyncio
import copy
from fractions import Fraction

import pytest
from fastapi import HTTPException
from pydantic import SecretStr
from sqlalchemy import delete, select, update
from sqlalchemy.exc import DBAPIError

from core.config import settings
from core.models import IngestedSource, Profile, Review, SourceChunk, User
from core.models.retrieval import ChunkEmbedding, RetrievalIndex
from core.services import hybrid_retrieval, keyword_retrieval, retrieval_indexing
from ingestion.embeddings.approved import EmbeddingError
from tests.embedding_fixtures import submitted
from tests.integration import test_semantic_retrieval as base

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


@pytest.fixture
def provider(monkeypatch):
    base.RankProvider.calls = []
    monkeypatch.setattr(retrieval_indexing, "make_provider", base.RankProvider)
    return base.RankProvider


async def query(client, actors, source, index=0, **data):
    return await client.post(
        base.endpoint(source),
        headers=base.auth(actors, index),
        json={"query": "POS", "mode": "hybrid", **data},
    )


async def test_canonical_fusion_counts_depth_and_unchanged_modes(
    client, database, actors, provider
):
    ident, source = await base.ready(database, actors, provider)
    before = await base.counts(database)
    result = await query(client, actors, source, top_k=1)
    assert result.status_code == 200, result.text
    body = result.json()
    assert body["candidate_limit"] == 20 and body["requested_top_k"] == 1
    assert (
        body["corpus_chunk_count"]
        == body["semantic_candidate_count"]
        == body["unique_candidate_count"]
        == 4
    )
    assert body["keyword_candidate_count"] == body["keyword_total_match_count"] == 2
    assert len(body["matches"]) == 1 and body["index_id"] == ident and body["rrf_constant"] == 60
    assert body["matches"][0]["retrieval_source"] == "both"
    all_rows = (await query(client, actors, source, top_k=20)).json()
    assert len(all_rows["matches"]) == 4
    assert all_rows == (await query(client, actors, source, top_k=20)).json()
    async with database() as db:
        for row in all_rows["matches"]:
            chunk = await db.get(SourceChunk, row["chunk_id"])
            doc = await db.get(IngestedSource, row["source_id"])
            assert chunk.source_id == doc.id and doc.revision_id == source.revision_id
            assert (
                row["excerpt"]
                == chunk.text
                == "\n".join(doc.text.splitlines()[row["line_start"] - 1 : row["line_end"]])
            )
            assert (
                row["path"] == doc.path
                and row["commit_sha"] == source.ingestion_result["commit_sha"]
            )
            terms = [
                Fraction(1, 60 + row[k])
                for k in ("keyword_rank", "semantic_rank")
                if row[k] is not None
            ]
            assert row["rrf_score"] == float(sum(terms)) and row["provenance_verified"] is True
    for mode in ("keyword", "semantic"):
        standalone = await client.post(
            base.endpoint(source), headers=base.auth(actors), json={"query": "POS", "mode": mode}
        )
        assert standalone.status_code == 200 and standalone.json()["mode"] == mode
    assert await base.counts(database) == before


async def test_users_revisions_foreign_ids_and_concurrent_isolation(
    client, database, actors, provider
):
    a, source = await base.ready(database, actors, provider)
    b, foreign = await base.ready(database, actors, provider, 1)
    c, other = await base.ready(database, actors, provider, files={"x.py": b"POS another revision"})
    for src, index, index_id in [(source, 1, a), (foreign, 0, b), (source, 0, b), (source, 0, c)]:
        result = await query(client, actors, src, index=index, index_id=index_id)
        assert result.status_code == 404, result.text
    assert provider.calls == []
    unauth = await client.post(base.endpoint(source), json={"query": "x", "mode": "hybrid"})
    assert unauth.status_code == 401 and provider.calls == []
    one, two = await asyncio.gather(
        query(client, actors, source), query(client, actors, foreign, index=1)
    )
    # Exercise two independent owners within the existing two-slot capacity.
    three = await query(client, actors, other)
    for response, index_id in [(one, a), (two, b), (three, c)]:
        assert response.status_code == 200, response.text
        assert response.json()["index_id"] == index_id
    async with database() as db:
        with pytest.raises(HTTPException) as exc:
            await hybrid_retrieval.search(
                db,
                actors["users"][1].id,
                source.ingestion_result["repository_id"],
                source.revision_id,
                {"mode": "hybrid", "query": "POS"},
            )
        assert exc.value.status_code == 404


@pytest.mark.parametrize(
    "fault",
    ["missing", "inactive", "failed", "version", "dimension", "proof", "source", "missing_vector"],
)
async def test_preflight_denial_before_either_branch(
    client, database, actors, provider, monkeypatch, fault
):
    if fault == "missing":
        ident, source = await submitted(database, actors, files=base.FILES)
    else:
        ident, source = await base.ready(database, actors, provider)
    called = []
    original = keyword_retrieval.rank_snapshot

    async def ranked(*args):
        called.append(1)
        return await original(*args)

    monkeypatch.setattr(keyword_retrieval, "rank_snapshot", ranked)
    if fault in ("version", "dimension"):
        monkeypatch.setattr(
            settings,
            "embedding_lmstudio_version" if fault == "version" else "embedding_lmstudio_dimensions",
            "changed" if fault == "version" else 1024,
        )
    elif fault != "missing":
        async with database() as db:
            if fault == "inactive":
                await db.execute(
                    update(User).where(User.id == actors["users"][0].id).values(is_active=False)
                )
            elif fault == "failed":
                await db.execute(
                    update(RetrievalIndex)
                    .where(RetrievalIndex.id == ident)
                    .values(state="failed", published_at=None, published_count=0)
                )
            elif fault == "proof":
                proof = copy.deepcopy(source.ingestion_result)
                proof["verified"] = False
                await db.execute(
                    update(Review).where(Review.id == source.id).values(ingestion_result=proof)
                )
            elif fault == "source":
                row = await db.scalar(
                    select(SourceChunk)
                    .where(SourceChunk.profile_id == actors["profiles"][0].id)
                    .limit(1)
                )
                row.text = "corrupt"
            else:
                await db.execute(delete(ChunkEmbedding).where(ChunkEmbedding.index_id == ident))
            await db.commit()
    response = await query(client, actors, source, index_id=ident)
    assert response.status_code == (401 if fault == "inactive" else 409), response.text
    assert provider.calls == [] and called == []


@pytest.mark.parametrize(
    "fault", ["inactive", "owner", "chunk", "metadata", "index", "configuration", "vector"]
)
async def test_changes_between_branches_fail_closed_without_held_http_locks(
    client, database, actors, provider, monkeypatch, fault
):
    ident, source = await base.ready(database, actors, provider)
    entered, release = asyncio.Event(), asyncio.Event()

    class Paused(base.RankProvider):
        async def embed_query(self, text, reserve):
            await reserve()
            entered.set()
            await release.wait()
            return self.vector("POS")

    monkeypatch.setattr(retrieval_indexing, "make_provider", Paused)
    pending = asyncio.create_task(query(client, actors, source))
    try:
        await asyncio.wait_for(entered.wait(), 10)
        async with database() as db:
            if fault == "inactive":
                await db.execute(
                    update(User).where(User.id == actors["users"][0].id).values(is_active=False)
                )
            elif fault == "owner":
                await db.execute(
                    update(Profile)
                    .where(Profile.id == actors["profiles"][0].id)
                    .values(user_id=actors["users"][1].id)
                )
            elif fault == "index":
                await db.execute(
                    update(RetrievalIndex)
                    .where(RetrievalIndex.id == ident)
                    .values(state="cancelled", published_at=None, published_count=0)
                )
            elif fault == "configuration":
                monkeypatch.setattr(settings, "embedding_lmstudio_version", "changed")
            elif fault == "vector":
                row = await db.scalar(
                    select(ChunkEmbedding).where(ChunkEmbedding.index_id == ident).limit(1)
                )
                row.embedding = [-value for value in row.embedding]
            elif fault == "chunk":
                row = await db.scalar(
                    select(SourceChunk)
                    .where(SourceChunk.profile_id == actors["profiles"][0].id)
                    .limit(1)
                )
                row.text = "corrupt"
            else:
                row = await db.scalar(
                    select(IngestedSource)
                    .where(IngestedSource.revision_id == source.revision_id)
                    .limit(1)
                )
                row.source_metadata = {**row.source_metadata, "changed": True}
            await asyncio.wait_for(db.commit(), 5)
    finally:
        release.set()
    response = await pending
    assert response.status_code in (401, 404, 409), response.text
    assert "matches" not in response.json()


async def test_hosted_permission_no_fallback_and_dimension_separation(
    client, database, actors, provider, monkeypatch
):
    monkeypatch.setattr(settings, "embedding_hosted_source_allowed", True)
    monkeypatch.setattr(settings, "embedding_mistral_api_key", SecretStr("not-real"))
    monkeypatch.setattr(settings, "embedding_mistral_max_requests", 10)
    ident, source = await base.ready(database, actors, provider, configuration="mistral")
    assert (await query(client, actors, source, configuration="mistral")).status_code == 409
    assert provider.calls == []
    monkeypatch.setattr(settings, "semantic_search_hosted_query_allowed", True)
    response = await query(client, actors, source, configuration="mistral", index_id=ident)
    assert response.status_code == 200 and response.json()["dimensions"] == 1024, response.text
    provider.calls = []
    assert (
        await query(client, actors, source, configuration="lmstudio", index_id=ident)
    ).status_code == 409
    assert provider.calls == []


async def test_provider_failure_does_not_return_keyword_fallback(
    client, database, actors, provider, monkeypatch
):
    _, source = await base.ready(database, actors, provider)

    class Failed(base.RankProvider):
        async def embed_query(self, text, reserve):
            await reserve()
            raise EmbeddingError("network_failure")

    monkeypatch.setattr(retrieval_indexing, "make_provider", Failed)
    result = await query(client, actors, source)
    assert result.status_code == 503 and "matches" not in result.json()


@pytest.mark.parametrize("empty", [True, False])
async def test_legitimate_empty_branches(client, database, actors, provider, empty):
    _, source = await base.ready(
        database, actors, provider, files={"README.md": b"" if empty else b"POS"}
    )
    result = await query(client, actors, source, query="no_lexical_match")
    assert result.status_code == 200, result.text
    data = result.json()
    assert data["keyword_candidate_count"] == 0
    assert data["semantic_candidate_count"] == (0 if empty else 1)
    assert all(
        row["retrieval_source"] == "semantic" and row["keyword_rank"] is None
        for row in data["matches"]
    )
    assert len(provider.calls) == (0 if empty else 1)


async def test_token_budget_before_provider_and_dirty_session(
    client, database, actors, provider, monkeypatch
):
    _, source = await base.ready(database, actors, provider)
    monkeypatch.setattr(settings, "keyword_search_max_tokens", 1)
    result = await query(client, actors, source)
    assert result.status_code == 413 and provider.calls == [], result.text
    async with database() as db:
        user = await db.get(User, actors["users"][0].id)
        user.is_active = False
        with pytest.raises(RuntimeError, match="clean session"):
            await hybrid_retrieval.search(
                db,
                user.id,
                source.ingestion_result["repository_id"],
                source.revision_id,
                {"mode": "hybrid", "query": "POS"},
            )
        assert db.dirty


async def test_database_rejects_cross_revision_contamination(database, actors, provider):
    ident, source = await base.ready(database, actors, provider)
    _, other = await base.ready(database, actors, provider, files={"x.py": b"POS other"})
    async with database() as db:
        row = await db.scalar(
            select(ChunkEmbedding).where(ChunkEmbedding.index_id == ident).limit(1)
        )
        row.revision_id = other.revision_id
        with pytest.raises(DBAPIError):
            await db.commit()
        await db.rollback()


async def test_hybrid_timeout_and_log_redaction(
    client, database, actors, provider, monkeypatch, caplog, capsys
):
    _, source = await base.ready(
        database, actors, provider, files={"a.py": b"PRIVATE_SOURCE_CONTENT"}
    )

    class Slow(base.RankProvider):
        async def embed_query(self, text, reserve):
            await reserve()
            await asyncio.sleep(1)
            return self.vector("POS")

    monkeypatch.setattr(retrieval_indexing, "make_provider", Slow)
    monkeypatch.setattr(settings, "hybrid_search_timeout_seconds", 0.1)
    response = await query(client, actors, source, query="PRIVATE_QUERY_SECRET")
    assert response.status_code == 503 and "matches" not in response.json()
    monkeypatch.setattr(retrieval_indexing, "make_provider", base.RankProvider)
    monkeypatch.setattr(settings, "hybrid_search_timeout_seconds", 55)
    assert (await query(client, actors, source)).status_code == 200
    logs = caplog.text + capsys.readouterr().out
    assert "PRIVATE_QUERY_SECRET" not in logs and "PRIVATE_SOURCE_CONTENT" not in logs


async def test_real_candidate_cap_and_keyword_only_union(client, database, actors, provider):
    files = {f"f{n:02}.py": f"POS file {n}".encode() for n in range(24)}
    files["z.py"] = b"NEG lexical target"
    _, source = await base.ready(database, actors, provider, files=files)
    response = await query(client, actors, source, query="NEG", top_k=5)
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["candidate_limit"] == data["semantic_candidate_count"] == 20
    assert data["keyword_candidate_count"] == 1 and data["corpus_chunk_count"] == 25
    assert data["unique_candidate_count"] == 21
    lexical = [m for m in data["matches"] if m["retrieval_source"] == "keyword"]
    assert len(lexical) == 1 and lexical[0]["path"] == "z.py"
    assert lexical[0]["semantic_rank"] is None and lexical[0]["cosine_distance"] is None
