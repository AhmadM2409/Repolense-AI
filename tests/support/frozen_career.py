"""Evaluation-only frozen fixture boundary. Never register with application routes.

Fixture revision markers are content addresses, NOT GitHub commit attestations.
Production SQL snapshots are delegated unchanged; only an explicit FixtureDB is adapted.
"""

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from uuid import NAMESPACE_URL, uuid5

from fastapi import HTTPException

from api.schemas.career_matching import Extraction, ExtractionCheck, MatchCheck, MatchDraft
from api.schemas.search import KeywordMatch, KeywordSearchRequest
from core.contracts import SourceDocument, TextChunk
from core.services import career_assessment as service
from core.services import career_evidence as policy
from core.services import keyword_retrieval as keyword
from ingestion.chunking.base import source_line_chunks

FIXTURE = Path("tests/fixtures/career_matching_holdout_v1.json")
FROZEN_SHA = "7c850131b955638801759f2967ae5a5cfd442ca76922bf8d4f8fc4a35fd13405"
ORDER = ("h1", "h2", "h3", "h4", "h5", "h6")
PHASES = ("extract", "verify_extraction", "match", "verify_match")
PROVENANCE = "synthetic_fixture_content_address_not_remote_GitHub"


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def identity(value):
    return str(uuid5(NAMESPACE_URL, "phase29-fixture-v2:" + value))


def cases():
    if hashlib.sha256(FIXTURE.read_bytes()).hexdigest() != FROZEN_SHA:
        raise ValueError("Frozen specification changed")
    result = json.loads(FIXTURE.read_text())["cases"]
    if tuple(c["id"] for c in result) != ORDER:
        raise ValueError("Frozen order changed")
    return result


class FixtureDB:
    def __init__(self, case, owner, authorize):
        original = next(c for c in cases() if c["id"] == case["id"])
        if original != case:
            raise ValueError("Fixture substitution")
        self.case, self.owner, self.authorize = case, str(owner), authorize
        self.profile = identity(self.owner + ":fixture-profile")
        self.manifest, self.documents, self.chunks = [], {}, {}
        for name, files in case["repositories"].items():
            rid = identity(case["id"] + ":" + name)
            marker = sha(json.dumps(files, sort_keys=True))[:40]
            rev = identity(rid + ":" + marker)
            self.manifest.append(
                dict(
                    repository_id=rid,
                    revision_id=rev,
                    commit_sha=marker,
                    full_name="synthetic fixture: " + name,
                )
            )
            documents, chunks = [], []
            for path, text in files.items():
                sid = identity(rev + ":" + path)
                doc = SourceDocument(
                    id=sid,
                    profile_id=self.profile,
                    source_type="synthetic_fixture",
                    revision_id=rev,
                    path=path,
                    text=text,
                    content_hash=sha(text),
                    source_metadata={
                        "provenance": PROVENANCE,
                        "remote_verified": False,
                        "frozen_spec_sha256": FROZEN_SHA,
                    },
                )
                documents.append(doc)
                for part in source_line_chunks(text, {}):
                    index = part.metadata["chunk_index"]
                    chunks.append(
                        (
                            TextChunk(
                                id=identity(sid + ":" + str(index)),
                                source_id=sid,
                                profile_id=self.profile,
                                chunk_index=index,
                                text=part.text,
                                line_start=part.metadata["line_start"],
                                line_end=part.metadata["line_end"],
                            ),
                            path,
                        )
                    )
            self.documents[rid], self.chunks[rid] = documents, chunks

    async def rollback(self):
        pass  # No SQL/source writes in this immutable fixture boundary.

    async def snapshot(self, uid, repo, rev, limits):
        original = next(c for c in cases() if c["id"] == self.case["id"])
        if self.case != original:
            raise ValueError("Frozen fixture mutated")
        if str(uid) != self.owner:
            raise HTTPException(404, "Fixture scope unavailable")
        await self.authorize()  # Live callback authenticates original real owner/input in SQL.
        row = next(
            (
                m
                for m in self.manifest
                if m["repository_id"] == str(repo) and m["revision_id"] == str(rev)
            ),
            None,
        )
        if row is None:
            raise HTTPException(404, "Fixture revision unavailable")
        documents, chunks = self.documents[str(repo)], self.chunks[str(repo)]
        expected = next(
            files
            for name, files in self.case["repositories"].items()
            if identity(self.case["id"] + ":" + name) == str(repo)
        )
        if (
            len(documents) != len(expected)
            or len(chunks) > limits.chunks
            or len(documents) > limits.sources
        ):
            raise ValueError("Fixture counts/resource bounds")
        if sum(len(d.text.encode()) for d in documents) > limits.corpus_bytes:
            raise ValueError("Fixture corpus bound")
        marker = sha(json.dumps(expected, sort_keys=True))[:40]
        if row["commit_sha"] != marker or str(rev) != identity(str(repo) + ":" + marker):
            raise ValueError("Fixture revision content address")
        if len({str(d.id) for d in documents}) != len(documents) or len(
            {str(c.id) for c, _ in chunks}
        ) != len(chunks):
            raise ValueError("Duplicate fixture identity")
        if {str(c.source_id) for c, _ in chunks} != {str(d.id) for d in documents}:
            raise ValueError("Orphan/missing source chunks")
        if (
            sum(len(json.dumps(d.source_metadata).encode()) for d in documents)
            > limits.metadata_bytes
        ):
            raise ValueError("Fixture metadata bound")
        for doc in documents:
            SourceDocument.model_validate(doc.model_dump())
            if (
                doc.path not in expected
                or doc.text != expected[doc.path]
                or str(doc.id) != identity(str(rev) + ":" + doc.path)
                or doc.content_hash != sha(expected[doc.path])
                or str(doc.profile_id) != self.profile
                or str(doc.revision_id) != str(rev)
                or doc.source_type != "synthetic_fixture"
                or doc.source_metadata
                != {
                    "provenance": PROVENANCE,
                    "remote_verified": False,
                    "frozen_spec_sha256": FROZEN_SHA,
                }
            ):
                raise ValueError("Fixture source integrity")
            members = sorted(
                (c for c, p in chunks if str(c.source_id) == str(doc.id)),
                key=lambda c: c.chunk_index,
            )
            canonical = source_line_chunks(doc.text, {})
            if len(members) != len(canonical):
                raise ValueError("Missing fixture chunks")
            for chunk, part in zip(members, canonical, strict=True):
                TextChunk.model_validate(chunk.model_dump())
                if (
                    str(chunk.id) != identity(str(doc.id) + ":" + str(part.metadata["chunk_index"]))
                    or str(chunk.profile_id) != self.profile
                    or chunk.chunk_index != part.metadata["chunk_index"]
                    or chunk.line_start != part.metadata["line_start"]
                    or chunk.line_end != part.metadata["line_end"]
                    or chunk.text != part.text
                    or (chunk, doc.path) not in chunks
                ):
                    raise ValueError("Fixture chunk integrity")
        if sum(len(c.text.encode()) for c, _ in chunks) > limits.corpus_bytes:
            raise ValueError("Fixture chunk bound")
        fingerprint = json.dumps(
            [
                row,
                [d.model_dump(mode="json") for d in documents],
                [c.model_dump(mode="json") for c, _ in chunks],
            ],
            sort_keys=True,
        )
        return keyword.Snapshot(
            str(repo), str(rev), row["commit_sha"], sha(fingerprint), tuple(chunks)
        )

    async def search(self, uid, repo, rev, request):
        request = KeywordSearchRequest.model_validate(request)
        limits = keyword.SearchLimits.configured()
        if (
            request.top_k > service.settings.keyword_search_max_top_k
            or len(request.query) > service.settings.keyword_search_max_query_characters
        ):
            raise ValueError("Fixture query bound")
        before = await self.snapshot(uid, repo, rev, limits)
        ordered, _ = await keyword.rank_snapshot(before, request.query, limits)
        after = await self.snapshot(uid, repo, rev, limits)
        if before.digest != after.digest:
            raise ValueError("Fixture changed during rank")
        byid = {str(c.id): (c, path) for c, path in after.chunks}
        matches = []
        for n, (key, score) in enumerate(ordered[: request.top_k], 1):
            chunk, path = byid[key]
            if len(chunk.text.encode()) > limits.excerpt_bytes:
                raise ValueError("Fixture excerpt bound")
            matches.append(
                KeywordMatch(
                    source_id=chunk.source_id,
                    chunk_id=chunk.id,
                    path=path,
                    commit_sha=after.sha,
                    line_start=chunk.line_start,
                    line_end=chunk.line_end,
                    excerpt=chunk.text,
                    bm25_score=score,
                    rank=n,
                )
            )
        if (
            len(json.dumps([m.model_dump(mode="json") for m in matches]).encode())
            > limits.response_bytes
        ):
            raise ValueError("Fixture response bound")
        return SimpleNamespace(matches=matches)


@contextmanager
def adapted_reads():
    original_snapshot, original_search = keyword._snapshot, keyword.search

    async def snapshot(db, uid, repo, rev, limits):
        if isinstance(db, FixtureDB):
            return await db.snapshot(uid, repo, rev, limits)
        return await original_snapshot(db, uid, repo, rev, limits)

    async def search(db, uid, repo, rev, request):
        if isinstance(db, FixtureDB):
            return await db.search(uid, repo, rev, request)
        return await original_search(db, uid, repo, rev, request)

    with patch.object(keyword, "_snapshot", snapshot), patch.object(keyword, "search", search):
        yield


async def gather(db, requirement):
    with adapted_reads():
        return await service.gather(db, db.owner, db.manifest, "keyword", requirement)


async def audit_citations(db, match):
    for c in match["citations"]:
        snapshot = await db.snapshot(
            db.owner, c["repository_id"], c["revision_id"], keyword.SearchLimits.configured()
        )
        doc = next(
            (d for d in db.documents[c["repository_id"]] if str(d.id) == c["source_id"]), None
        )
        chunk = next((x for x, _ in snapshot.chunks if str(x.id) == c["chunk_id"]), None)
        if (
            doc is None
            or chunk is None
            or str(chunk.source_id) != str(doc.id)
            or str(chunk.profile_id) != str(doc.profile_id)
            or str(doc.revision_id) != str(c["revision_id"])
            or c["path"] != doc.path
            or c["commit_sha"] != snapshot.sha
            or not chunk.line_start <= c["line_start"] <= c["line_end"] <= chunk.line_end
            or c["quote"] != "\n".join(doc.text.splitlines()[c["line_start"] - 1 : c["line_end"]])
        ):
            raise ValueError("Noncanonical fixture citation")


async def evaluate(case, db, call):
    output = {
        "case_id": case["id"],
        "provenance": PROVENANCE,
        "case_sha256": sha(json.dumps(case, sort_keys=True)),
    }
    raw = await call(policy.EXTRACT, {"job": case["job"]}, Extraction, "extract")
    requirements = policy.requirements(raw, case["job"], sha(case["job"]))
    checked = await call(
        policy.EXTRACT_CHECK,
        {"job": case["job"], "requirements": requirements},
        ExtractionCheck,
        "verify_extraction",
    )
    policy.validate_extraction(checked, requirements)
    output["requirements"] = requirements
    if len(requirements) != 1:
        return {**output, "status": "QUALITY_FAILURE_GRANULARITY"}
    req = requirements[0]
    selected, scopes, context, coverage = await gather(db, req)
    output.update(
        context=context,
        selected={k: vars(v) for k, v in selected.items()},
        scopes=scopes,
        coverage=coverage,
    )
    if selected:
        raw = await call(
            policy.MATCH, {"requirement": req, "evidence": context}, MatchDraft, "match"
        )
        checked = await call(
            policy.MATCH_CHECK,
            {"requirement": req, "evidence": context, "candidate": json.loads(raw)},
            MatchCheck,
            "verify_match",
        )
        match = policy.accepted(raw, checked, selected, scopes)
    else:
        match = {
            "classification": "not evidenced",
            "explanation": "No safe matching excerpt within retrieval bounds.",
            "uncertainty": "Not a judgment of personal ability.",
            "citations": [],
            "support_verdict": "insufficient",
        }
    await audit_citations(db, match)
    paths = {c["path"] for c in match["citations"]}
    quote = "\n".join(c["quote"] for c in match["citations"])
    criteria = {
        "classification": match["classification"] == case["expected"],
        "required_paths": set(case["evidence_paths"]) <= paths,
        "numeric_fact": "67" in quote if "required_facts" in case else True,
        "strict_operator": "<" in quote if "required_facts" in case else True,
    }
    return {
        **output,
        "status": "DETERMINISTIC_GATES_PASS_PENDING_SEMANTIC_REVIEW"
        if all(criteria.values())
        else "QUALITY_FAILURE",
        "match": match,
        "quality": criteria,
        "semantic_review": None,
    }


class SuiteJournal:
    """New authorization, same SQL quota. No replay, skip, J3, retry or cap refill."""

    def __init__(self, path, grant, plan):
        self.path, self.grant, self.plan = Path(path), Path(grant), Path(plan)
        with self.connect() as c:
            c.execute(
                "CREATE TABLE IF NOT EXISTS attempts(id INTEGER PRIMARY KEY,case_id TEXT,phase TEXT,state TEXT,metadata TEXT,batch_id TEXT)"
            )
            c.execute(
                "CREATE TABLE IF NOT EXISTS control(id INTEGER PRIMARY KEY CHECK(id=1),stopped INTEGER)"
            )
            c.execute("INSERT OR IGNORE INTO control VALUES(1,0)")
            c.execute(
                "CREATE TABLE IF NOT EXISTS cases(case_id TEXT PRIMARY KEY,position INTEGER UNIQUE,status TEXT,allocation INTEGER)"
            )

    def connect(self):
        return sqlite3.connect(self.path, timeout=2)

    def authorize(self, arm):
        grant = json.loads(self.grant.read_text())
        plan = json.loads(self.plan.read_text())
        if (
            grant.get("approved") is not True
            or grant.get("billing_and_quota_confirmed") is not True
            or not grant.get("approval_reference")
            or grant.get("allowed_arm") != arm
            or arm != "frozen-j2-only"
            or type(grant.get("maximum_posts")) is not int
            or not 1 <= grant["maximum_posts"] <= 22
            or grant.get("plan_sha256") != hashlib.sha256(self.plan.read_bytes()).hexdigest()
            or grant.get("owner_id") != plan["owner_id"]
            or grant.get("input_id") != plan["input_id"]
            or datetime.fromisoformat(grant["expires_at"]) <= datetime.now(timezone.utc)
            or any(
                hashlib.sha256(Path(k).read_bytes()).hexdigest() != v
                for k, v in plan["frozen_files"].items()
            )
        ):
            raise ValueError("Suite authorization")
        return grant

    def count(self):
        with self.connect() as c:
            return c.execute("SELECT count(*) FROM attempts").fetchone()[0]

    def halt(self):
        with self.connect() as c:
            c.execute("UPDATE control SET stopped=1 WHERE id=1")

    def begin(self, case, allocation, owner_remaining, aggregate_remaining):
        grant = self.authorize("frozen-j2-only")
        with self.connect() as c:
            c.execute("BEGIN IMMEDIATE")
            rows = c.execute("SELECT case_id,status FROM cases ORDER BY position").fetchall()
            count = c.execute("SELECT count(*) FROM attempts").fetchone()[0]
            if (
                c.execute("SELECT stopped FROM control").fetchone()[0]
                or len(rows) >= len(ORDER)
                or case != ORDER[len(rows)]
                or any(state != "PASS" for _, state in rows)
                or allocation != (2 if case == "h4" else 4)
                or min(grant["maximum_posts"] - count, owner_remaining, aggregate_remaining)
                < allocation
            ):
                raise ValueError("Case admission/sequence/budget")
            c.execute("INSERT INTO cases VALUES(?,?,?,?)", (case, len(rows), "ACTIVE", allocation))

    def reserve(self, case, phase, arm):
        grant = self.authorize(arm)
        with self.connect() as c:
            c.execute("BEGIN IMMEDIATE")
            rows = c.execute(
                "SELECT phase,state FROM attempts WHERE case_id=? ORDER BY id", (case,)
            ).fetchall()
            active = c.execute(
                "SELECT allocation FROM cases WHERE case_id=? AND status='ACTIVE'", (case,)
            ).fetchone()
            count = c.execute("SELECT count(*) FROM attempts").fetchone()[0]
            if (
                c.execute("SELECT stopped FROM control").fetchone()[0]
                or active is None
                or len(rows) >= active[0]
                or count >= grant["maximum_posts"]
                or phase not in PHASES
                or phase != PHASES[len(rows)]
                or any(state != "provider_validated" for _, state in rows)
            ):
                raise ValueError("Phase/retry/budget denied")
            return c.execute(
                "INSERT INTO attempts(case_id,phase,state,metadata,batch_id) VALUES(?,?,?,?,?)",
                (case, phase, "reserved_before_shared", "{}", grant["plan_sha256"]),
            ).lastrowid

    def mark(self, ident, state, metadata=None):
        with self.connect() as c:
            prior = json.loads(
                c.execute("SELECT metadata FROM attempts WHERE id=?", (ident,)).fetchone()[0]
            )
            c.execute(
                "UPDATE attempts SET state=?,metadata=? WHERE id=?",
                (state, json.dumps({**prior, **(metadata or {})}), ident),
            )

    def finish(self, case, passed):
        with self.connect() as c:
            active = c.execute(
                "SELECT allocation FROM cases WHERE case_id=? AND status='ACTIVE'", (case,)
            ).fetchone()
            rows = c.execute(
                "SELECT phase,state FROM attempts WHERE case_id=? ORDER BY id", (case,)
            ).fetchall()
            if (
                active is None
                or passed
                and (
                    len(rows) != active[0]
                    or any(state != "provider_validated" for _, state in rows)
                )
            ):
                raise ValueError("Incomplete case cannot pass")
            c.execute(
                "UPDATE cases SET status=? WHERE case_id=? AND status='ACTIVE'",
                ("PASS" if passed else "FAIL", case),
            )
        if not passed:
            self.halt()
