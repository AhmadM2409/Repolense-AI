"""Real Nomic/PG evaluation of fixed synthetic, non-executed, pre-labeled source."""

import hashlib
import json
import math
import statistics
import time
from pathlib import Path

import pytest

from core.models import IngestedSource, SourceChunk
from tests.integration import test_semantic_retrieval as base
from tests.retrieval_quality import metrics

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]
ROOT = Path(__file__).resolve().parents[2]


async def test_live_local_retrieval_quality(client, database, actors):
    dataset = json.loads((ROOT / "tests/fixtures/retrieval_quality.json").read_text())
    assert len(dataset["queries"]) == 20 and len(dataset["documents"]) == 16
    assert all(set(q["relevant"]) <= dataset["documents"].keys() for q in dataset["queries"])
    started = time.perf_counter()
    ident, source = await base.ready(
        database, actors, None, files={p: t.encode() for p, t in dataset["documents"].items()}
    )
    indexing_seconds = time.perf_counter() - started
    before = await base.counts(database)
    output = {
        "fixture_version": dataset["version"],
        "fixture_sha256": hashlib.sha256(
            (ROOT / "tests/fixtures/retrieval_quality.json").read_bytes()
        ).hexdigest(),
        "documents": 16,
        "queries": 20,
        "provider": "lmstudio",
        "dimensions": 768,
        "indexing_seconds": indexing_seconds,
        "modes": {},
    }
    for mode in ("keyword", "semantic", "hybrid"):
        rows = []
        latencies = []
        for q in dataset["queries"]:
            started = time.perf_counter()
            response = await client.post(
                base.endpoint(source),
                headers=base.auth(actors),
                json={"query": q["query"], "mode": mode, "top_k": 10},
            )
            elapsed = time.perf_counter() - started
            assert response.status_code == 200, (mode, q["id"], response.status_code)
            body = response.json()
            paths = [m["path"] for m in body["matches"]]
            assert len(paths) == len(set(paths))
            async with database() as db:
                for m in body["matches"]:
                    chunk = await db.get(SourceChunk, m["chunk_id"])
                    doc = await db.get(IngestedSource, m["source_id"])
                    assert doc.revision_id == source.revision_id and doc.path == m["path"]
                    assert (
                        m["excerpt"]
                        == chunk.text
                        == "\n".join(doc.text.splitlines()[m["line_start"] - 1 : m["line_end"]])
                    )
                    assert m["commit_sha"] == source.ingestion_result["commit_sha"]
            latencies.append(elapsed)
            rows.append(
                {
                    "query_id": q["id"],
                    "paths": paths,
                    "metrics": metrics(paths, q["relevant"]),
                    "no_answer_result_count": len(paths) if not q["relevant"] else None,
                    "latency_seconds": elapsed,
                }
            )
        positive = [row["metrics"] for row in rows if row["metrics"] is not None]
        output["modes"][mode] = {
            "metrics": {key: statistics.mean(m[key] for m in positive) for key in positive[0]},
            "answerable_queries": len(positive),
            "no_answer_queries": 20 - len(positive),
            "no_answer_queries_with_results": sum(
                row["no_answer_result_count"] > 0 for row in rows if row["metrics"] is None
            ),
            "median_seconds": statistics.median(latencies),
            "p95_nearest_rank_seconds": sorted(latencies)[math.ceil(0.95 * len(latencies)) - 1],
            "first_request_seconds": latencies[0],
            "model_state": "already loaded and used for indexing; not a cold model benchmark",
            "per_query": rows,
        }
    assert await base.counts(database) == before
    (ROOT / ".test-tmp/phase17c3-quality.json").write_text(json.dumps(output, indent=2))
    # No quality threshold is tuned to force hybrid to win. Gold metrics are observations.


async def test_all_mode_contracts_keyword_independent_of_provider(
    client, database, actors, monkeypatch
):
    from core.services import retrieval_indexing
    from tests.embedding_fixtures import submitted

    _, source = await submitted(database, actors, files={"a.py": b"POS canonical source"})
    before = await base.counts(database)

    def forbidden(config):
        raise AssertionError("Provider must not be invoked")

    monkeypatch.setattr(retrieval_indexing, "make_provider", forbidden)
    for body in ({"query": "POS"}, {"query": "POS", "mode": "keyword"}):
        response = await client.post(base.endpoint(source), headers=base.auth(actors), json=body)
        assert response.status_code == 200 and response.json()["mode"] == "keyword"
    for mode in ("semantic", "hybrid"):
        response = await client.post(
            base.endpoint(source), headers=base.auth(actors), json={"query": "POS", "mode": mode}
        )
        assert response.status_code == 409 and "matches" not in response.json()
    for body in ({"query": "POS", "mode": "chat"}, {"query": "POS", "owner_id": "foreign"}):
        response = await client.post(base.endpoint(source), headers=base.auth(actors), json=body)
        assert response.status_code == 422
    assert await base.counts(database) == before
