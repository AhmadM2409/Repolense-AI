"""J2 real PostgreSQL/canonical multi-repository retrieval; model HTTP doubled only."""

import asyncio
import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest
import pytest_asyncio
from sqlalchemy import func, select, update

from api.schemas.career import content_hash
from api.schemas.career_matching import CONSENT
from core.config import settings
from core.models import (
    CareerAIBudget,
    CareerAssessment,
    CareerInput,
    CareerSelection,
    Evidence,
    Finding,
    IngestedSource,
    Profile,
    User,
)
from core.services import career_assessment as service
from core.services import career_evidence as policy
from core.services import github_ingestion
from ingestion.github import GitHubClient
from rag.generator.gemini_provider import MODEL, Gemini
from rag.generator.local_provider import GenerationError
from tests.github_responses import GitHubResponses
from tests.integration.test_auth_authorization import bearer

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]
JD = "Python functions required. SQL documentation preferred. Kubernetes required."
REQUIREMENTS = [
    {"quote": quote, "skill": skill, "priority": priority, "depth": "", "aliases": []}
    for quote, skill, priority in [
        ("Python functions required.", "python", "required"),
        ("SQL documentation preferred.", "sql", "preferred"),
        ("Kubernetes required.", "kubernetes", "required"),
    ]
]


class Model:
    def __init__(self):
        self.posts = []
        self.hook = None
        self.failure = None
        self.bad_extract = False
        self.incomplete = False
        self.bad_reference = False
        self.verdict = "supported"
        self.partial = False
        self.entered = asyncio.Event()
        self.release = None

    def output(self, messages):
        system = messages[0]["content"]
        body = json.loads(messages[1]["content"])
        if system == policy.EXTRACT:
            rows = [dict(r) for r in REQUIREMENTS]
            if self.bad_extract:
                rows[0]["quote"] = "Invented requirement."
            return {"requirements": rows}
        if system == policy.EXTRACT_CHECK:
            return {
                "complete": not self.incomplete,
                "items": [
                    {"requirement_id": r["id"], "supported": True} for r in body["requirements"]
                ],
            }
        label = "partially demonstrated" if self.partial else "demonstrated"
        if body["requirement"]["skill"] == "sql":
            label = "documented/claimed"
        if system == policy.MATCH:
            claim = (
                "The source defines Python functions."
                if label != "documented/claimed"
                else "The documentation claims SQL support."
            )
            return {
                "classification": label,
                "claim": claim,
                "references": [
                    {"evidence_id": ("E5" if self.bad_reference else k)} for k in body["evidence"]
                ],
                "uncertainty": "Bounded source evidence only.",
            }
        assert system == policy.MATCH_CHECK
        return {
            "classification": label,
            "verdict": self.verdict,
            "complete": label == "demonstrated",
        }

    def factory(self, budget):
        model = self

        class Fake:
            deadline = None

            async def generate(self, messages, before_request, *, response_schema):
                await before_request()
                budget.reserve()
                model.posts.append(messages)
                if model.hook:
                    await model.hook()
                model.entered.set()
                if model.release:
                    await model.release.wait()
                if model.failure:
                    raise GenerationError(model.failure)
                return SimpleNamespace(
                    content=json.dumps(model.output(messages)),
                    model=MODEL,
                    usage={"totalTokenCount": 1},
                    latency_seconds=0.01,
                )

        return Fake()


@pytest_asyncio.fixture
async def portfolio(database, actors):
    uid = actors["users"][0].id
    pid = actors["profiles"][0].id
    jobs = []
    async with database() as db:
        for name, files in [
            ("example/alpha", {"python_store.py": b"def python_store():\n    return True\n"}),
            (
                "example/beta",
                {
                    "python_cache.py": b"def python_cache():\n    return True\n",
                    "README.md": b"# SQL\nSQL support is documented.\n",
                },
            ),
        ]:
            fixture = GitHubResponses(name=name, files=files)
            job, _ = await github_ingestion.submit(db, pid, uid, name)
            async with GitHubClient(transport=httpx.MockTransport(fixture)) as client:
                await github_ingestion.process(db, job.id, uid, client)
            await db.refresh(job)
            assert job.status == "complete"
            jobs.append(
                SimpleNamespace(
                    id=job.id,
                    requested_repository=job.requested_repository,
                    revision_id=job.revision_id,
                    ingestion_result=job.ingestion_result,
                )
            )
        row = CareerInput(
            owner_id=uid,
            profile_id=pid,
            account="example",
            job_description=JD,
            job_hash=content_hash(JD),
            discovery={
                "status": "complete",
                "items": [{"full_name": j.requested_repository, "eligible": True} for j in jobs],
            },
            started_at=datetime.now(UTC),
        )
        db.add(row)
        await db.flush()
        for i, job in enumerate(jobs):
            db.add(
                CareerSelection(
                    input_id=row.id,
                    profile_id=pid,
                    position=i,
                    full_name=job.requested_repository,
                    job_id=job.id,
                    revision_id=job.revision_id,
                    commit_sha=job.ingestion_result["commit_sha"],
                )
            )
        await db.commit()
    return row.id


@pytest.fixture
def model(monkeypatch):
    model = Model()
    monkeypatch.setattr(settings, "career_j2_enabled", True)
    monkeypatch.setattr(service, "provider_identity", lambda: "f" * 64)
    monkeypatch.setattr(service, "make_provider", model.factory)
    return model


async def create(client, actors, portfolio, **extra):
    body = {"request_id": str(uuid4()), "hosted_processing_consent": CONSENT, **extra}
    response = await client.post(
        f"/career-inputs/{portfolio}/assessments", headers=bearer(actors["users"][0].id), json=body
    )
    return response, body


async def match(client, actors, identity):
    saved = await client.get(
        f"/career-assessments/{identity}", headers=bearer(actors["users"][0].id)
    )
    row = saved.json()
    requirement = row["requirements"][min(len(row["matches"]), len(row["requirements"]) - 1)]["id"]
    return await client.post(
        f"/career-assessments/{identity}/match",
        headers=bearer(actors["users"][0].id),
        json={
            "requirement_id": requirement,
            "reviewed": True,
            "hosted_processing_consent": CONSENT,
        },
    )


async def test_actual_multi_repository_matching_and_review(
    client, database, actors, portfolio, model
):
    response, body = await create(client, actors, portfolio)
    assert response.status_code == 201, response.text
    row = response.json()
    identity = row["id"]
    assert row["status"] == "review" and row["inference_spent"] == 2
    for req in row["requirements"]:
        assert JD[req["start"] : req["end"]] == req["quote"] and req["job_hash"] == content_hash(JD)
    repeat = await client.post(
        f"/career-inputs/{portfolio}/assessments", headers=bearer(actors["users"][0].id), json=body
    )
    assert repeat.json()["id"] == identity and len(model.posts) == 2
    for step in range(3):
        response = await match(client, actors, identity)
        assert response.status_code == 200, response.text
        if step == 0:
            count = len(model.posts)
            replay = await client.post(
                f"/career-assessments/{identity}/match",
                headers=bearer(actors["users"][0].id),
                json={
                    "requirement_id": row["requirements"][0]["id"],
                    "reviewed": True,
                    "hosted_processing_consent": CONSENT,
                },
            )
            assert replay.status_code == 200 and len(replay.json()["matches"]) == 1
            assert len(model.posts) == count
    result = response.json()
    assert result["status"] == "complete" and result["inference_spent"] == 6
    assert [r["classification"] for r in result["matches"]] == [
        "demonstrated",
        "documented/claimed",
        "not evidenced",
    ]
    citations = result["matches"][0]["citations"]
    assert len({c["repository_id"] for c in citations}) == 2
    async with database() as db:
        for found in result["matches"]:
            for cite in found["citations"]:
                source = await db.get(IngestedSource, cite["source_id"])
                assert source.revision_id == cite["revision_id"] and source.path == cite["path"]
                assert (
                    "\n".join(source.text.splitlines()[cite["line_start"] - 1 : cite["line_end"]])
                    == cite["quote"]
                )
        assert await db.scalar(select(func.count()).select_from(Finding)) == 0
        assert await db.scalar(select(func.count()).select_from(Evidence)) == 0
        assert (await db.get(CareerAIBudget, 1)).spent == 6
    count = len(model.posts)
    assert (
        await client.get(f"/career-assessments/{identity}", headers=bearer(actors["users"][0].id))
    ).status_code == 200
    assert (await match(client, actors, identity)).json()["status"] == "complete" and len(
        model.posts
    ) == count


@pytest.mark.parametrize(
    "case",
    ["unauthenticated", "foreign", "missing_consent", "question_consent", "inactive", "disabled"],
)
async def test_denials_no_inference(client, database, actors, portfolio, model, case, monkeypatch):
    headers = bearer(actors["users"][1 if case == "foreign" else 0].id)
    data = {"request_id": str(uuid4()), "hosted_processing_consent": CONSENT}
    if case == "unauthenticated":
        headers = {}
    if case == "missing_consent":
        data.pop("hosted_processing_consent")
    if case == "question_consent":
        data["hosted_processing_consent"] = "gemini-source-processing-v1"
    if case == "disabled":
        monkeypatch.setattr(settings, "career_j2_enabled", False)
    if case == "inactive":
        async with database() as db:
            await db.execute(
                update(User).where(User.id == actors["users"][0].id).values(is_active=False)
            )
            await db.commit()
    response = await client.post(
        f"/career-inputs/{portfolio}/assessments", headers=headers, json=data
    )
    assert (
        response.status_code
        == {
            "unauthenticated": 401,
            "foreign": 404,
            "missing_consent": 422,
            "question_consent": 422,
            "inactive": 401,
            "disabled": 409,
        }[case]
    ), response.text
    assert not model.posts


@pytest.mark.parametrize("case", ["invented_quote", "incomplete", "provider", "malformed"])
async def test_extraction_failures_preserved(client, actors, portfolio, model, case):
    model.bad_extract = case == "invented_quote"
    model.incomplete = case == "incomplete"
    model.failure = "transport_or_timeout" if case == "provider" else None
    if case == "malformed":
        model.output = lambda _: {"requirements": [], "secret": "must not echo"}
    response, body = await create(client, actors, portfolio)
    assert response.status_code in {502, 503}, response.text
    assert response.json()["status"] == "failed" and response.json()["matches"] == []
    assert "must not echo" not in response.text
    count = len(model.posts)
    repeated = await client.post(
        f"/career-inputs/{portfolio}/assessments", headers=bearer(actors["users"][0].id), json=body
    )
    assert repeated.json()["status"] == "failed" and len(model.posts) == count


@pytest.mark.parametrize("case", ["partial", "contradicted", "bad_reference"])
async def test_match_partial_or_rejected(client, actors, portfolio, model, case):
    response, _ = await create(client, actors, portfolio)
    identity = response.json()["id"]
    model.partial = case == "partial"
    model.bad_reference = case == "bad_reference"
    model.verdict = "contradicted" if case == "contradicted" else "supported"
    result = await match(client, actors, identity)
    if case == "partial":
        assert (
            result.status_code == 200
            and result.json()["matches"][0]["classification"] == "partially demonstrated"
        )
    else:
        assert result.status_code == 502 and result.json()["matches"] == []


@pytest.mark.parametrize("cap", ["global", "owner", "rate", "stopped"])
async def test_aggregate_budget_survives_and_fails_closed(
    client, database, actors, portfolio, model, cap
):
    response, _ = await create(client, actors, portfolio)
    identity = response.json()["id"]
    import time

    async with database() as db:
        budget = await db.get(CareerAIBudget, 1)
        if cap == "global":
            budget.spent = 100
        if cap == "owner":
            budget.owners = {actors["users"][0].id: 40}
            budget.spent = 40
        if cap == "rate":
            budget.recent = [time.time()] * 10
        if cap == "stopped":
            budget.stopped = True
        await db.commit()
    count = len(model.posts)
    result = await match(client, actors, identity)
    # Refused by the preflight before any status change or POST (2026-11-04): budget/latch 409,
    # full rolling minute a retryable 429. The assessment stays reviewable; nothing is spent.
    assert result.status_code == (429 if cap == "rate" else 409), result.text
    assert len(model.posts) == count
    saved = await client.get(f"/career-assessments/{identity}", headers=bearer(actors["users"][0].id))
    assert saved.json()["status"] == "review" and saved.json()["matches"] == []
    async with database() as db:
        after = await db.get(CareerAIBudget, 1)
        assert after.spent == {"global": 100, "owner": 40}.get(cap, 2)


async def test_inflight_transfer_and_canonical_change(client, database, actors, portfolio, model):
    async def revoke():
        async with database() as db:
            await db.execute(
                update(Profile)
                .where(Profile.id == actors["profiles"][0].id)
                .values(user_id=actors["users"][1].id)
            )
            await db.commit()

    model.hook = revoke
    response, _ = await create(client, actors, portfolio)
    assert response.status_code == 404 and len(model.posts) == 1
    async with database() as db:
        row = await db.scalar(select(CareerAssessment))
        assert row.status == "failed" and len(row.calls) == 1 and not row.matches
    for user in actors["users"]:
        assert (
            await client.get(f"/career-assessments/{row.id}", headers=bearer(user.id))
        ).status_code == 404


async def test_source_immutability_and_direct_service_denial(
    client, database, actors, portfolio, model
):
    response, _ = await create(client, actors, portfolio)
    identity = response.json()["id"]
    from fastapi import HTTPException

    async with database() as db:
        with pytest.raises(HTTPException):
            await service.detail(db, actors["users"][1].id, identity)
        await db.rollback()
        source = await db.scalar(
            select(IngestedSource).where(IngestedSource.path == "python_store.py")
        )
        from sqlalchemy.exc import DBAPIError

        source.text = "changed source"
        with pytest.raises(DBAPIError, match="immutable"):
            await db.commit()
        await db.rollback()
    result = await match(client, actors, identity)
    assert result.status_code == 200 and len(model.posts) == 4


async def test_concurrent_stage_and_cancel_no_publication(
    client, database, actors, portfolio, model
):
    response, _ = await create(client, actors, portfolio)
    identity = response.json()["id"]
    model.release = asyncio.Event()
    model.entered.clear()
    task = asyncio.create_task(match(client, actors, identity))
    await asyncio.wait_for(model.entered.wait(), 10)
    assert (await match(client, actors, identity)).status_code == 409
    cancelled = await client.post(
        f"/career-assessments/{identity}/cancel", headers=bearer(actors["users"][0].id)
    )
    assert cancelled.json()["status"] == "cancelled"
    model.release.set()
    result = await task
    assert result.status_code == 409
    final = await client.get(
        f"/career-assessments/{identity}", headers=bearer(actors["users"][0].id)
    )
    assert final.json()["matches"] == [] and final.json()["inference_spent"] == 3


async def test_expired_interrupted_stage_does_not_resend(
    client, database, actors, portfolio, model
):
    response, _ = await create(client, actors, portfolio)
    identity = response.json()["id"]
    async with database() as db:
        row = await db.get(CareerAssessment, identity)
        row.status = "matching"
        row.operation_token = str(uuid4())
        row.lease_until = datetime.now(UTC) - timedelta(seconds=1)
        await db.commit()
    result = await match(client, actors, identity)
    assert (
        result.json()["status"] == "failed"
        and result.json()["error_category"] == "interrupted_unknown"
        and len(model.posts) == 2
    )


async def test_real_adapter_mocked_http_and_quota_latch(
    client, database, actors, portfolio, model, monkeypatch
):
    def response(request):
        assert (
            request.url.host == "generativelanguage.googleapis.com"
            and request.headers["x-goog-api-key"] == "fixture-only-key"
        )
        return httpx.Response(429, json={"error": {"message": "private provider diagnostic"}})

    monkeypatch.setattr(
        service,
        "make_provider",
        lambda budget: Gemini("fixture-only-key", budget, transport=httpx.MockTransport(response)),
    )
    result, _ = await create(client, actors, portfolio)
    assert result.status_code == 503 and "private provider diagnostic" not in result.text
    async with database() as db:
        budget = await db.get(CareerAIBudget, 1)
        assert budget.spent == 1 and budget.stopped


@pytest.mark.parametrize("mode", ["semantic", "hybrid"])
async def test_real_pgvector_retrieval_modes_without_live_nomic(
    client, database, actors, portfolio, model, monkeypatch, mode
):
    from core.models import RetrievalIndex, Review
    from core.services import retrieval_indexing
    from workers.run import tick

    class Embeddings:
        query_calls = 0

        def __init__(self, config):
            self.config = config

        def vector(self, text):
            values = [0.0] * self.config.dimensions
            values[1 if text.startswith("#") else 0] = 1.0
            return values

        async def embed_documents(self, texts, reserve):
            await reserve()
            return [self.vector(t) for t in texts]

        async def embed_query(self, text, reserve):
            await reserve()
            type(self).query_calls += 1
            return self.vector("python")

    monkeypatch.setattr(retrieval_indexing, "make_provider", Embeddings)
    async with database() as db:
        jobs = (
            await db.scalars(
                select(Review)
                .where(Review.job_type == "github_ingestion")
                .order_by(Review.created_at)
            )
        ).all()
        job_ids = [j.id for j in jobs]
    for job_id in job_ids:
        async with database() as db:
            index_id, _ = await retrieval_indexing.submit(
                db, actors["users"][0].id, job_id, "lmstudio"
            )
        await tick(database, "j2-index-fixture")
        async with database() as db:
            assert (await db.get(RetrievalIndex, index_id)).state == "ready"
    response, _ = await create(client, actors, portfolio, retrieval_mode=mode)
    assert response.status_code == 201, response.text
    identity = response.json()["id"]
    result = await match(client, actors, identity)
    assert result.status_code == 200, result.text
    assert Embeddings.query_calls == 2
    assert result.json()["matches"][0]["coverage"]["mode"] == mode
    assert len({c["repository_id"] for c in result.json()["matches"][0]["citations"]}) == 2


async def test_missing_index_no_fallback_and_exact_request_contract(
    client, actors, portfolio, model
):
    response, _ = await create(client, actors, portfolio, retrieval_mode="hybrid")
    assert response.status_code == 409 and not model.posts
    response, _ = await create(client, actors, portfolio)
    identity = response.json()["id"]
    denied = await client.post(
        f"/career-assessments/{identity}/match",
        headers=bearer(actors["users"][0].id),
        json={
            "requirement_id": response.json()["requirements"][0]["id"],
            "reviewed": 1,
            "hosted_processing_consent": CONSENT,
        },
    )
    assert denied.status_code == 422 and len(model.posts) == 2


async def test_per_assessment_budget_and_immutable_requirements(
    client, database, actors, portfolio, model
):
    from sqlalchemy.exc import DBAPIError

    response, _ = await create(client, actors, portfolio)
    identity = response.json()["id"]
    async with database() as db:
        with pytest.raises(DBAPIError):
            await db.execute(
                update(CareerAssessment)
                .where(CareerAssessment.id == identity)
                .values(requirements=[])
            )
        await db.rollback()
        row = await db.get(CareerAssessment, identity)
        row.calls = [{"phase": "fixture", "at": 0, "state": "consumed"}] * 10
        budget = await db.get(CareerAIBudget, 1)
        budget.spent = 10
        budget.owners = {actors["users"][0].id: 10}
        await db.commit()
    result = await match(client, actors, identity)
    assert result.status_code == 409 and "fewer than two" in result.text  # Preflight, no POST.
    saved = await client.get(f"/career-assessments/{identity}", headers=bearer(actors["users"][0].id))
    assert saved.json()["inference_spent"] == 10 and len(model.posts) == 2


async def test_dirty_service_session_and_wrong_manifest_denied(
    client, database, actors, portfolio, model
):
    response, _ = await create(client, actors, portfolio)
    identity = response.json()["id"]
    async with database() as db:
        profile = await db.get(Profile, actors["profiles"][0].id)
        profile.github_username = "pending-user-change"
        with pytest.raises(RuntimeError, match="clean session"):
            await service.match_next(
                db,
                actors["users"][0].id,
                identity,
                {
                    "requirement_id": response.json()["requirements"][0]["id"],
                    "reviewed": True,
                    "hosted_processing_consent": CONSENT,
                },
            )
        assert profile in db.dirty
        await db.rollback()
        from sqlalchemy.exc import DBAPIError

        with pytest.raises(DBAPIError, match="immutable"):
            await db.execute(
                update(CareerAssessment)
                .where(CareerAssessment.id == identity)
                .values(signature="0" * 64)
            )
        await db.rollback()
    assert len(model.posts) == 2
    denied = await client.post(
        f"/career-assessments/{identity}/match",
        headers=bearer(actors["users"][0].id),
        json={
            "requirement_id": response.json()["requirements"][0]["id"],
            "reviewed": True,
            "hosted_processing_consent": CONSENT,
            "revision_id": str(uuid4()),
        },
    )
    assert denied.status_code == 422 and len(model.posts) == 2
    assert any(e["type"] == "extra_forbidden" for e in denied.json()["detail"])


async def test_concurrent_assessments_share_atomic_aggregate_budget(
    client, database, actors, portfolio, model
):
    from alembic.config import Config

    from alembic import command

    first, _ = await create(client, actors, portfolio)
    second, _ = await create(client, actors, portfolio)
    ids = [first.json()["id"], second.json()["id"]]
    assert len(model.posts) == 4
    async with database() as db:
        budget = await db.get(CareerAIBudget, 1)
        budget.spent = 98  # Both pass the advisory preflight; the locked reservation decides.
        await db.commit()
    results = await asyncio.gather(*[match(client, actors, i) for i in ids])
    assert {r.status_code for r in results} <= {200, 409, 503}
    assert any(r.status_code != 200 for r in results)
    assert len(model.posts) == 6  # exactly the two remaining aggregate reservations are spent
    async with database() as db:
        assert (await db.get(CareerAIBudget, 1)).spent == 100
        connection = await db.connection()

        def downgrade(conn):
            cfg = Config("alembic.ini")
            cfg.attributes["connection"] = conn
            with pytest.raises(RuntimeError, match="history/budget"):
                command.downgrade(cfg, "009")

        await connection.run_sync(downgrade)
        await db.rollback()
        assert (await db.get(CareerAIBudget, 1)).spent == 100


async def test_truncated_extraction_fails_closed_before_checker_and_replay(
    client, database, actors, portfolio, model, monkeypatch
):
    original = model.output

    def truncated(messages):
        result = original(messages)
        if messages[0]["content"] == policy.EXTRACT:
            result["requirements"][0]["quote"] = "Python functions"
        return result

    monkeypatch.setattr(model, "output", truncated)
    response, body = await create(client, actors, portfolio)
    assert response.status_code == 502
    row = response.json()
    assert row["status"] == "failed" and row["error_category"] == "validation_failure"
    assert row["requirements"] == [] and row["matches"] == []
    assert row["inference_spent"] == 1 and len(model.posts) == 1
    # Even the optimistic checker double cannot publish a truncated quote.
    assert model.output([{"content": policy.EXTRACT_CHECK}, {"content": '{"requirements": []}'}])[
        "complete"
    ]
    repeated = await client.post(
        f"/career-inputs/{portfolio}/assessments", headers=bearer(actors["users"][0].id), json=body
    )
    assert repeated.json()["status"] == "failed" and len(model.posts) == 1
    async with database() as db:
        budget = await db.get(CareerAIBudget, 1)
        assert budget.spent == 1 and budget.owners[actors["users"][0].id] == 1


async def test_obligation_retrieval_windows_fair_portfolio_and_canonical_citations(
    database, actors
):
    uid, pid = actors["users"][0].id, actors["profiles"][0].id
    manifest = []
    async with database() as db:
        for name, files in [
            (
                "sample/readers",
                {
                    "README.md": b"# Python\n" + b"CSV " * 100 + b"\n",
                    "reader.py": (
                        "# banner\n" * 82 + "def read_csv(rows):\n    return list(rows)\n"
                    ).encode(),
                    "extra.py": b"def validate_csv(rows):\n    return bool(rows)\n",
                },
            ),
            (
                "sample/consumer",
                {
                    "reader.py": b"def read_csv(rows):\n    return tuple(rows)\n",
                    "wrong.js": b"function read_csv(rows) { return rows; }\n",
                    "styles.css": b".python { display: block; }\n",
                },
            ),
        ]:
            fixture = GitHubResponses(name=name, files=files)
            job, _ = await github_ingestion.submit(db, pid, uid, name)
            async with GitHubClient(transport=httpx.MockTransport(fixture)) as http:
                await github_ingestion.process(db, job.id, uid, http)
            await db.refresh(job)
            assert job.status == "complete"
            manifest.append(
                {
                    "repository_id": job.ingestion_result["repository_id"],
                    "revision_id": job.revision_id,
                    "full_name": name,
                }
            )
        requirement = {
            "skill": "python",
            "aliases": [],
            "quote": "Read CSV records using Python.",
            "depth": "Read CSV records using Python.",
        }
        selected, scopes, context, coverage = await service.gather(
            db, uid, manifest, "keyword", requirement
        )
        assert len(selected) >= 2 and len(selected) <= 5
        assert set(scope[0] for scope in scopes.values()) == {r["repository_id"] for r in manifest}
        assert all(v.path.endswith(".py") for v in selected.values())
        assert any(v.line_start > 80 for v in selected.values())
        assert coverage["relevance_filtered_candidates"] > 0
        assert len(json.dumps({"requirement": requirement, "evidence": context}).encode()) <= 1800
        raw = {
            "classification": "demonstrated",
            "claim": "The code defines read_csv.",
            "references": [
                {"evidence_id": k} for k, v in selected.items() if "def read_csv" in v.excerpt
            ],
            "uncertainty": "Mocked support verdict; bounded source only.",
        }
        check = {"classification": "demonstrated", "verdict": "supported", "complete": True}
        accepted = policy.accepted(json.dumps(raw), json.dumps(check), selected, scopes)
        assert accepted["citations"]
        for cite in accepted["citations"]:
            source = await db.get(IngestedSource, cite["source_id"])
            assert source.revision_id == cite["revision_id"] and source.path == cite["path"]
            assert (
                "\n".join(source.text.splitlines()[cite["line_start"] - 1 : cite["line_end"]])
                == cite["quote"]
            )
            assert len(cite["quote"].encode()) <= 300 and cite["line_end"] - cite["line_start"] < 8
        with pytest.raises(ValueError):
            policy.accepted(
                json.dumps({**raw, "references": [{"evidence_id": "E5", "quote": "fabricated"}]}),
                json.dumps(check),
                selected,
                scopes,
            )
        missing = {
            **requirement,
            "skill": "erlang",
            "quote": "Erlang actors required.",
            "depth": "Erlang actors",
        }
        absent, _, _, absent_coverage = await service.gather(db, uid, manifest, "keyword", missing)
        assert absent == {} and absent_coverage["searched_revisions"] == 2
        with pytest.raises(Exception) as denied:
            await service.gather(db, actors["users"][1].id, manifest, "keyword", requirement)
        assert getattr(denied.value, "status_code", None) == 404


@pytest_asyncio.fixture
async def dependency_portfolio(database, actors, request):
    uid, pid = actors["users"][0].id, actors["profiles"][0].id
    files = {
        ("python_store.py"): (
            b"from helpers import select_value\n"
            b"def python_store(rows):\n"
            b"    return select_value(rows)\n"
        )
    }
    if request.param != "missing":
        files["helpers.py"] = (
            b"def select_value(rows):\n    pass\n"
            if request.param == "stub"
            else b"def select_value(rows):\n    return rows['record']\n"
        )
    jobs = []
    async with database() as db:
        for name, bodies in [
            ("example/linked", files),
            (
                "example/independent",
                {
                    "python_cache.py": b"def python_cache():\n    return True\n",
                    "README.md": b"# SQL\nSQL support is documented.\n",
                },
            ),
        ]:
            fixture = GitHubResponses(name=name, files=bodies)
            job, _ = await github_ingestion.submit(db, pid, uid, name)
            async with GitHubClient(transport=httpx.MockTransport(fixture)) as http:
                await github_ingestion.process(db, job.id, uid, http)
            await db.refresh(job)
            jobs.append(
                SimpleNamespace(
                    id=job.id,
                    name=name,
                    revision_id=job.revision_id,
                    sha=job.ingestion_result["commit_sha"],
                )
            )
        row = CareerInput(
            owner_id=uid,
            profile_id=pid,
            account="example",
            job_description=JD,
            job_hash=content_hash(JD),
            started_at=datetime.now(UTC),
            discovery={
                "status": "complete",
                "items": [{"full_name": j.name, "eligible": True} for j in jobs],
            },
        )
        db.add(row)
        await db.flush()
        for i, j in enumerate(jobs):
            db.add(
                CareerSelection(
                    input_id=row.id,
                    profile_id=pid,
                    position=i,
                    full_name=j.name,
                    job_id=j.id,
                    revision_id=j.revision_id,
                    commit_sha=j.sha,
                )
            )
        await db.commit()
        return row.id


@pytest.mark.parametrize("dependency_portfolio", ["resolved", "missing", "stub"], indirect=True)
async def test_implementation_context_reaches_model_and_missing_callee_cannot_be_promoted(
    client, database, actors, dependency_portfolio, model, request
):
    case = request.node.callspec.params["dependency_portfolio"]
    response, _ = await create(client, actors, dependency_portfolio)
    assert response.status_code == 201
    identity = response.json()["id"]
    original_output = model.output

    def anchor_output(messages):
        output = original_output(messages)
        if messages[0]["content"] == policy.MATCH:
            output["references"] = output["references"][:1]
        return output

    model.output = anchor_output
    result = await match(client, actors, identity)
    payload = json.loads(model.posts[2][1]["content"])
    values = list(payload["evidence"].values())
    assert any("return select_value(rows)" in v["text"] for v in values)
    assert any(v.get("requires") for v in values)
    assert len(json.dumps(payload, ensure_ascii=False).encode()) <= 1800
    if case == "resolved":
        assert any("return rows['record']" in v["text"] for v in values)
        assert any("from helpers import select_value" in v["text"] for v in values)
        assert (
            result.status_code == 200
            and result.json()["matches"][0]["classification"] == "demonstrated"
        )
        for cite in result.json()["matches"][0]["citations"]:
            async with database() as db:
                source = await db.get(IngestedSource, cite["source_id"])
                assert (
                    "\n".join(source.text.splitlines()[cite["line_start"] - 1 : cite["line_end"]])
                    == cite["quote"]
                )
    else:
        assert any(v.get("missing_local_implementation") for v in values)
        assert result.status_code == 502 and result.json()["status"] == "failed"
        assert result.json()["matches"] == []
    if case == "resolved":
        reopened = await client.get(
            "/career-assessments/" + identity, headers=bearer(actors["users"][0].id)
        )
        assert reopened.status_code == 200
        assert reopened.json()["matches"] == result.json()["matches"]
        assert len(model.posts) == 4
        async with database() as db:
            saved = await db.get(CareerAssessment, identity)
            historical_matches = json.loads(json.dumps(saved.matches))
            selected, scopes, _, _ = await service.gather(
                db,
                actors["users"][0].id,
                saved.snapshot["revisions"],
                "keyword",
                saved.requirements[0],
            )
            anchor = next(
                key for key, item in selected.items() if "return select_value" in item.excerpt
            )
            raw = {
                "classification": "demonstrated",
                "claim": "The code defines read_python.",
                "references": [{"evidence_id": anchor}],
                "uncertainty": "Mocked support.",
            }
            check = {"classification": "demonstrated", "verdict": "supported", "complete": True}
            for fault in ("missing", "foreign", "quote", "unrelated", "incomplete"):
                items = {key: SimpleNamespace(**vars(item)) for key, item in selected.items()}
                boundaries = dict(scopes)
                candidate = json.loads(json.dumps(raw))
                verdict = dict(check)
                dependency = next(key for key in items[anchor].required_evidence if key != anchor)
                if fault == "missing":
                    items.pop(dependency)
                if fault == "foreign":
                    boundaries[dependency] = (str(uuid4()), str(uuid4()))
                if fault == "quote":
                    candidate["references"][0]["quote"] = "fabricated text"
                if fault == "incomplete":
                    items[dependency].implementation_complete = False
                if fault == "unrelated":
                    verdict["verdict"] = "insufficient"
                with pytest.raises(ValueError):
                    policy.accepted(json.dumps(candidate), json.dumps(verdict), items, boundaries)
            await db.refresh(saved)
            assert saved.matches == historical_matches
    # An optimistic mocked checker cannot bypass context completeness; reservations remain.
    async with database() as db:
        row = await db.get(CareerAssessment, identity)
        assert len(row.calls) == 4
