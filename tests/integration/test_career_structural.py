"""Real PostgreSQL normal J2 path; GitHub HTTP doubled, no inference or remote requests."""

import json
from pathlib import Path

import httpx
import pytest
from fastapi import HTTPException
from sqlalchemy.exc import DBAPIError

from core.models import IngestedSource, SourceChunk
from core.services import career_assessment as service
from core.services import github_ingestion
from ingestion.github import GitHubClient
from tests.github_responses import GitHubResponses
from tests.unit.test_career_structural import FLASK, GO

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]

H1 = json.loads(Path("tests/fixtures/career_matching_holdout_v1.json").read_text())["cases"][0]


async def ingest(db, uid, pid, name, files):
    fixture = GitHubResponses(name=name, files={p: t.encode() for p, t in files.items()})
    job, _ = await github_ingestion.submit(db, pid, uid, name)
    async with GitHubClient(transport=httpx.MockTransport(fixture)) as http:
        await github_ingestion.process(db, job.id, uid, http)
    await db.refresh(job)
    assert job.status == "complete"
    return {
        "repository_id": job.ingestion_result["repository_id"],
        "revision_id": job.revision_id,
        "full_name": name,
    }


@pytest.mark.parametrize(
    "path,text,job",
    [
        ("orders.py", FLASK, "Reject incomplete submissions at a web endpoint."),
        ("edge.go", GO, "Check inbound messages before an API endpoint accepts them."),
        ("handler.py", H1["repositories"]["demo/web"]["handler.py"], H1["job"]),
    ],
)
async def test_normal_pipeline_structural_context_exact_scope_and_integrity(
    database, actors, path, text, job
):
    uid, pid = actors["users"][0].id, actors["profiles"][0].id
    req = {"skill": "input validation", "aliases": [], "quote": job, "depth": job}
    async with database() as db:
        manifest = [
            await ingest(
                db,
                uid,
                pid,
                "independent/edge",
                {path: text, "style.css": ".web { display: block; }"},
            )
        ]
        selected, scopes, context, coverage = await service.gather(
            db, uid, manifest, "keyword", req
        )
        assert (
            selected
            and coverage["structural_hints"] >= 1
            and coverage["structural_seed_candidates"] >= 1
        )
        assert all(v.path == path for v in selected.values())
        assert (
            len(selected) <= 5
            and len(json.dumps({"requirement": req, "evidence": context}).encode()) <= 1800
        )
        assert any("422" in v.excerpt or "empty payload" in v.excerpt for v in selected.values())
        assert any("post" in v.excerpt or "net/http" in v.excerpt for v in selected.values())
        for key, item in selected.items():
            source = await db.get(IngestedSource, str(item.source_id))
            chunk = await db.get(SourceChunk, str(item.chunk_id))
            assert str(source.revision_id) == scopes[key][1] and chunk.source_id == source.id
            assert item.excerpt == "\n".join(
                source.text.splitlines()[item.line_start - 1 : item.line_end]
            )
            assert chunk.line_start <= item.line_start <= item.line_end <= chunk.line_end
        with pytest.raises(HTTPException) as denied:
            await service.gather(db, actors["users"][1].id, manifest, "keyword", req)
        assert denied.value.status_code == 404
        missing = {
            "skill": "kubernetes",
            "aliases": [],
            "quote": "Configure Kubernetes manifests.",
            "depth": "",
        }
        assert not (await service.gather(db, uid, manifest, "keyword", missing))[0]
        source = await db.get(IngestedSource, str(next(iter(selected.values())).source_id))
        original_text = source.text
        source.text += "\n# unverified edit"
        with pytest.raises(DBAPIError, match="immutable source/revision"):
            await db.commit()
        await db.rollback()
        reread, _, _, _ = await service.gather(db, uid, manifest, "keyword", req)
        assert reread
        current = await db.get(IngestedSource, str(next(iter(reread.values())).source_id))
        assert current.text == original_text


async def test_multi_repository_groups_do_not_mix_source_scope(database, actors):
    uid, pid = actors["users"][0].id, actors["profiles"][0].id
    req = {
        "skill": "input validation",
        "aliases": [],
        "quote": "Reject invalid inbound submissions at an API endpoint.",
        "depth": "",
    }
    async with database() as db:
        manifest = [
            await ingest(db, uid, pid, "independent/forms", {"orders.py": FLASK}),
            await ingest(db, uid, pid, "independent/transport", {"edge.go": GO}),
        ]
        selected, scopes, _, coverage = await service.gather(db, uid, manifest, "keyword", req)
        assert {scope[0] for scope in scopes.values()} == {m["repository_id"] for m in manifest}
        for key, item in selected.items():
            assert item.required_evidence
            assert all(scopes[member] == scopes[key] for member in item.required_evidence)
        assert coverage["searched_revisions"] == 2 and coverage["structural_seed_candidates"] == 2
