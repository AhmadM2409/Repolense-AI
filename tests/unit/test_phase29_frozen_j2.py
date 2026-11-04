"""Evaluation infrastructure only; no live model or authoritative-budget writes."""

import hashlib
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import HTTPException

from core.services import career_context, career_retrieval
from core.services import keyword_retrieval as keyword
from tests.support import frozen_career as harness

OWNER = str(uuid4())


async def authorized():
    pass


@pytest.fixture
def db():
    return harness.FixtureDB(harness.cases()[0], OWNER, authorized)


@pytest.mark.asyncio
async def test_all_six_interfaces_and_truthful_snapshots():
    for case in harness.cases():
        db = harness.FixtureDB(case, OWNER, authorized)
        for row in db.manifest:
            snap = await db.snapshot(
                OWNER, row["repository_id"], row["revision_id"], keyword.SearchLimits.configured()
            )
            assert snap.chunks and all(
                d.source_metadata["remote_verified"] is False
                for d in db.documents[row["repository_id"]]
            )
        # Probe interfaces only; these are not model extractions or quality results.
        req = {"skill": "python", "aliases": [], "quote": case["job"], "depth": case["job"]}
        await harness.gather(db, req)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault", ["owner", "revision", "text", "chunk", "id", "marker", "metadata", "missing"]
)
async def test_scope_and_source_tampering_rejected(db, fault):
    row = db.manifest[0]
    uid = OWNER
    rev = row["revision_id"]
    repo = row["repository_id"]
    if fault == "owner":
        uid = str(uuid4())
    if fault == "revision":
        rev = str(uuid4())
    if fault == "text":
        db.documents[repo][0] = db.documents[repo][0].model_copy(update={"text": "tampered"})
    if fault == "chunk":
        db.chunks[repo][0] = (
            db.chunks[repo][0][0].model_copy(update={"line_end": 99}),
            "handler.py",
        )
    if fault == "id":
        db.documents[repo][0] = db.documents[repo][0].model_copy(update={"id": uuid4()})
    if fault == "marker":
        row["commit_sha"] = "0" * 40
    if fault == "metadata":
        db.documents[repo][0].source_metadata["remote_verified"] = True
    if fault == "missing":
        db.chunks[repo] = []
    with pytest.raises((ValueError, HTTPException)):
        await db.snapshot(uid, repo, rev, keyword.SearchLimits.configured())


@pytest.mark.asyncio
async def test_real_session_loader_is_not_replaced(monkeypatch):
    calls = []

    async def original(*args):
        calls.append(args)
        return "actual SQL loader"

    monkeypatch.setattr(keyword, "_snapshot", original)
    with harness.adapted_reads():
        assert await keyword._snapshot(object(), "u", "r", "v", None) == "actual SQL loader"
    assert len(calls) == 1 and keyword._snapshot is original


@pytest.mark.asyncio
async def test_canonical_citation_rejection(db):
    row = db.manifest[0]
    chunk, path = db.chunks[row["repository_id"]][0]
    cite = dict(
        source_id=str(chunk.source_id),
        chunk_id=str(chunk.id),
        repository_id=row["repository_id"],
        revision_id=row["revision_id"],
        commit_sha=row["commit_sha"],
        path=path,
        line_start=chunk.line_start,
        line_end=chunk.line_end,
        quote=chunk.text,
    )
    await harness.audit_citations(db, {"citations": [cite]})
    with pytest.raises(ValueError):
        await harness.audit_citations(db, {"citations": [{**cite, "quote": "fabricated"}]})


@pytest.mark.asyncio
async def test_positive_negative_group_controls(db):
    req = {
        "skill": "payload",
        "aliases": [],
        "quote": "Accept payload.",
        "depth": "Accept payload.",
    }
    selected, scopes, _, _ = await harness.gather(db, req)
    assert selected
    # The adapter must pass production's completeness signal through unchanged, never forge it.
    # (Until 2026-11-04 production flagged builtin ValueError and the framework decorator as
    # unresolved local calls; it now treats them as non-local code. This is a post-exposure
    # regression change, not held-out accuracy evidence for h1.)
    row = db.manifest[0]
    snap = await db.snapshot(
        db.owner, row["repository_id"], row["revision_id"], keyword.SearchLimits.configured()
    )
    production, _ = career_context.groups(
        snap,
        [SimpleNamespace(chunk_id=c.id) for c, _ in snap.chunks],
        career_retrieval.query(req),
    )
    expected = {
        (w.path, w.line_start, w.line_end): w.implementation_complete
        for g in production
        for w in g.windows
    }
    assert all(
        expected[(v.path, v.line_start, v.line_end)] is v.implementation_complete
        for v in selected.values()
    )
    negative = {
        "skill": "kubernetes",
        "aliases": [],
        "quote": "Configure Kubernetes manifests.",
        "depth": "",
    }
    assert not (await harness.gather(db, negative))[0]
    assert all(v.commit_sha == db.manifest[0]["commit_sha"] for v in selected.values())


@pytest.fixture
def journal(tmp_path):
    plan = tmp_path / "plan.json"
    grant = tmp_path / "grant.json"
    plan.write_text(json.dumps({"owner_id": OWNER, "input_id": "i", "frozen_files": {}}))
    grant.write_text(
        json.dumps(
            dict(
                approved=True,
                billing_and_quota_confirmed=True,
                approval_reference="OFFLINE_DOUBLE_ONLY",
                allowed_arm="frozen-j2-only",
                maximum_posts=22,
                owner_id=OWNER,
                input_id="i",
                plan_sha256=hashlib.sha256(plan.read_bytes()).hexdigest(),
                expires_at=(datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
            )
        )
    )
    return harness.SuiteJournal(tmp_path / "journal.sqlite3", grant, plan)


def test_order_requires_passed_predecessor(journal):
    with pytest.raises(ValueError):
        journal.begin("h2", 4, 22, 82)
    journal.begin("h1", 4, 22, 82)
    with pytest.raises(ValueError):
        journal.begin("h2", 4, 22, 82)
    with pytest.raises(ValueError):
        journal.finish("h1", True)


def test_no_j3_retry_or_unknown_replay(journal):
    journal.begin("h1", 4, 22, 82)
    with pytest.raises(ValueError):
        journal.reserve("h1", "roadmap", "frozen-j2-only")
    ident = journal.reserve("h1", "extract", "frozen-j2-only")
    with pytest.raises(ValueError):
        journal.reserve("h1", "extract", "frozen-j2-only")
    with pytest.raises(ValueError):
        journal.reserve("h1", "verify_extraction", "frozen-j2-only")
    journal.mark(ident, "failed_no_retry")
    with pytest.raises(ValueError):
        journal.reserve("h1", "extract", "frozen-j2-only")


@pytest.mark.parametrize("balance", [0, 3])
def test_full_case_admission(journal, balance):
    with pytest.raises(ValueError):
        journal.begin("h1", 4, balance, 82)
    assert journal.count() == 0


def test_exact_22_cap_durable_and_stopped(journal):
    for case in harness.ORDER:
        cost = 2 if case == "h4" else 4
        journal.begin(case, cost, 22 - journal.count(), 82 - journal.count())
        for phase in harness.PHASES[:cost]:
            journal.mark(journal.reserve(case, phase, "frozen-j2-only"), "provider_validated")
        journal.finish(case, True)
    assert journal.count() == 22
    fresh = harness.SuiteJournal(journal.path, journal.grant, journal.plan)
    with pytest.raises(ValueError):
        fresh.reserve("h6", "verify_match", "frozen-j2-only")
    fresh.halt()
    with pytest.raises(ValueError):
        fresh.begin("h1", 4, 22, 82)


@pytest.mark.asyncio
async def test_runner_only_j2_and_no_evidence_negative_stays_conservative(db):
    db = harness.FixtureDB(harness.cases()[3], OWNER, authorized)
    phases = []

    async def call(system, payload, schema, phase):
        phases.append(phase)
        if phase == "extract":
            return json.dumps(
                {
                    "requirements": [
                        dict(
                            quote=db.case["job"],
                            skill="http validation",
                            aliases=[],
                            priority="required",
                            depth=db.case["job"],
                        )
                    ]
                }
            )
        if phase == "verify_extraction":
            return json.dumps(
                {
                    "complete": True,
                    "items": [
                        {"requirement_id": payload["requirements"][0]["id"], "supported": True}
                    ],
                }
            )
        raise AssertionError("No evidence must not generate a model match or J3")

    result = await harness.evaluate(db.case, db, call)
    assert phases == ["extract", "verify_extraction"]
    assert result["status"] == "DETERMINISTIC_GATES_PASS_PENDING_SEMANTIC_REVIEW"
    assert result["match"]["classification"] == "not evidenced"
    assert result["match"]["citations"] == [] and result["semantic_review"] is None


@pytest.mark.asyncio
async def test_complete_direct_body_group_is_preserved():
    db = harness.FixtureDB(harness.cases()[4], OWNER, authorized)
    selected, _, _, _ = await harness.gather(
        db, {"skill": "mass", "aliases": [], "quote": "Admit mass.", "depth": ""}
    )
    assert selected and any(v.implementation_complete is True for v in selected.values())


@pytest.mark.asyncio
async def test_live_authority_callback_is_mandatory():
    async def denied():
        raise HTTPException(403, "Owner authority revoked")

    db = harness.FixtureDB(harness.cases()[0], OWNER, denied)
    row = db.manifest[0]
    with pytest.raises(HTTPException):
        await db.snapshot(
            OWNER, row["repository_id"], row["revision_id"], keyword.SearchLimits.configured()
        )


def test_invalid_grant_cannot_reserve(journal):
    grant = json.loads(journal.grant.read_text())
    grant["maximum_posts"] = 23
    journal.grant.write_text(json.dumps(grant))
    with pytest.raises(ValueError):
        journal.begin("h1", 4, 22, 82)
    assert journal.count() == 0
