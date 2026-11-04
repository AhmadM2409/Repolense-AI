"""Fresh offline end-to-end Career Intelligence journey on real PostgreSQL/pgvector.

Real: API routes, JWT auth, ownership, J1 discovery/selection/preparation, durable ingestion
and indexing workers, pinned revisions, canonical snapshots, BM25, pgvector, hybrid RRF,
J2 context/validation, J3 roadmap, persistence and budget accounting.
Doubled (no network): GitHub HTTP (mock transports), local Nomic embeddings (deterministic
concept vectors) and Gemini (a deterministic reviewer that reads the evidence it is given).

The portfolio is new synthetic code, unrelated to the frozen h1-h6 cases, and deliberately uses
code vocabulary that does not appear in the job text. Assertions concern server behaviour, not
model quality.
"""

import json
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest
from sqlalchemy import select, update

from api.schemas.career_matching import CONSENT
from core.config import settings
from core.models import CareerAIBudget, CareerAssessment, User
from core.services import career_assessment as assessment
from core.services import career_evidence as evidence_policy
from core.services import career_input, github_ingestion, retrieval_indexing
from core.services import career_roadmap_policy as roadmap_policy
from ingestion.github import GitHubClient
from ingestion.github_profile import Discovery
from rag.generator.gemini_provider import MODEL
from tests.github_responses import GitHubResponses
from tests.unit.test_career_input import repository
from workers.run import tick

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]

JD = (
    "Send events through a message broker. "
    "Keep database writes atomic when an error occurs. "
    "Validate incoming HTTP requests. "
    "Deploy services to Kubernetes."
)
REQUIREMENTS = [
    {"quote": q, "skill": s, "priority": "required", "depth": "", "aliases": []}
    for q, s in [
        ("Send events through a message broker.", "message broker"),
        ("Keep database writes atomic when an error occurs.", "atomic database writes"),
        ("Validate incoming HTTP requests.", "HTTP request validation"),
        ("Deploy services to Kubernetes.", "Kubernetes"),
    ]
]
REPOS = {
    "example/orders-service": {
        "orders/broker.py": (
            b"import pika\n\n\n"
            b"def publish_order(channel, order):\n"
            b'    channel.basic_publish(exchange="orders", routing_key="created", body=order)\n'
            b"    return True\n"
        ),
        "orders/store.py": (
            b"def save_order(connection, order):\n"
            b"    try:\n"
            b'        connection.execute("INSERT INTO orders VALUES (?)", (order,))\n'
            b"        connection.commit()\n"
            b"    except Exception:\n"
            b"        connection.rollback()\n"
            b"        raise\n"
            b"    return order\n"
        ),
        "orders/api.py": (
            b"from flask import Flask, abort, request\n\n"
            b"app = Flask(__name__)\n\n\n"
            b'@app.post("/orders")\n'
            b"def create_order():\n"
            b"    payload = request.get_json()\n"
            b'    if not payload or "sku" not in payload:\n'
            b"        abort(422)\n"
            b"    return payload\n"
        ),
    },
    "example/notes": {
        "README.md": b"# Notes\nA small notes app.\n",
        "notes.py": b"def add_note(notes, text):\n    notes.append(text)\n    return notes\n",
    },
}
# What the doubled reviewer looks for. It never sees file paths, only the evidence text sent.
MARKERS = {
    "message broker": "basic_publish",
    "atomic database writes": ".rollback(",
    "http request validation": "abort(",
    "kubernetes": "kind: Deployment",
}
CONCEPTS = [  # Fake Nomic: one dimension per concept; a weak shared bias dimension.
    {"broker", "pika", "basic_publish", "queue", "events"},
    {"atomic", "transaction", "rollback", "commit"},
    {"http", "request", "requests", "abort", "get_json", "validate"},
    {"kubernetes", "deployment", "cluster"},
]


class Embeddings:
    queries = 0
    documents = 0

    def __init__(self, config):
        self.config = config

    def vector(self, text):
        words = set(text.lower().replace("(", " ").replace(".", " ").split())
        values = [0.0] * self.config.dimensions
        for i, concept in enumerate(CONCEPTS):
            values[i] = float(len(words & concept))
        values[-1] = 0.1
        return values

    async def embed_documents(self, texts, reserve):
        await reserve()
        type(self).documents += len(texts)
        return [self.vector(t) for t in texts]

    async def embed_query(self, text, reserve):
        await reserve()
        type(self).queries += 1
        return self.vector(text)


class Reviewer:
    """Deterministic Gemini double. Answers only from the evidence actually sent to it."""

    def __init__(self):
        self.posts = []

    def output(self, system, body):
        if system == evidence_policy.EXTRACT:
            return {"requirements": [dict(r) for r in REQUIREMENTS]}
        if system == evidence_policy.EXTRACT_CHECK:
            return {
                "complete": True,
                "items": [
                    {"requirement_id": r["id"], "supported": True} for r in body["requirements"]
                ],
            }
        if system == evidence_policy.MATCH:
            marker = MARKERS[body["requirement"]["skill"].casefold()]
            items = body["evidence"]
            hit = next((k for k, v in items.items() if marker in v["text"]), None)
            if hit is None:
                return {
                    "classification": "not evidenced",
                    "claim": "",
                    "references": [],
                    "uncertainty": "No inspected excerpt shows this requirement.",
                }
            group = next(
                (v["requires"] for v in items.values() if hit in v.get("requires", [])), [hit]
            )
            return {
                "classification": "demonstrated",
                "claim": "The cited source implements the requirement.",
                "references": [{"evidence_id": k} for k in group],
                "uncertainty": "Bounded source evidence only.",
            }
        if system == evidence_policy.MATCH_CHECK:
            shown = body["candidate"]["classification"] == "demonstrated"
            return {
                "verdict": "supported" if shown else "insufficient",
                "complete": shown,
                "classification": body["candidate"]["classification"],
            }
        if system == roadmap_policy.GENERATE:
            return {
                "items": [
                    dict(
                        requirement_id=r["id"],
                        action_type="documenting" if r["citations"] else "learning",
                        action="Write proposed " + r["quote"] + " examples with verification.",
                        gap="This requirement is not demonstrated within inspected evidence.",
                        rationale="The job explicitly includes " + r["quote"],
                        completion_proof="Publish a reproducible example with verification output.",
                        project_citation=0 if r["citations"] else None,
                    )
                    for r in body["requirements"]
                ]
            }
        assert system == roadmap_policy.CHECK
        return {
            "complete": True,
            "items": [
                dict(
                    requirement_id=r["id"],
                    grounded=True,
                    individualized=True,
                    gap_accurate=True,
                    future_only=True,
                    observable_proof=True,
                    project_supported=True,
                    action_type_appropriate=True,
                )
                for r in body["requirements"]
            ],
        }

    def factory(self, budget):
        reviewer = self

        class Provider:
            deadline = None

            async def generate(self, messages, before_request, *, response_schema):
                await before_request()  # Real reservation runs BEFORE the (doubled) POST.
                budget.reserve()
                reviewer.posts.append(messages)
                content = reviewer.output(
                    messages[0]["content"], json.loads(messages[1]["content"])
                )
                return SimpleNamespace(
                    content=json.dumps(content),
                    model=MODEL,
                    usage={"totalTokenCount": 1},
                    latency_seconds=0.01,
                )

        return Provider()


@pytest.fixture
def offline(monkeypatch):
    reviewer = Reviewer()
    Embeddings.queries = Embeddings.documents = 0
    sources = {name: GitHubResponses(name=name, files=files) for name, files in REPOS.items()}

    def discovery(request):
        if request.url.path == "/users/example":
            return httpx.Response(200, json={"login": "example", "type": "User"})
        return httpx.Response(
            200,
            json=[
                repository(i, full_name=name, html_url="https://github.com/" + name)
                for i, name in enumerate(REPOS, 1)
            ],
        )

    def git(request):
        return sources["/".join(request.url.path.split("/")[2:4])](request)

    monkeypatch.setattr(settings, "career_j2_enabled", True)
    monkeypatch.setattr(assessment, "provider_identity", lambda: "f" * 64)
    monkeypatch.setattr(assessment, "make_provider", reviewer.factory)
    monkeypatch.setattr(retrieval_indexing, "make_provider", Embeddings)
    monkeypatch.setattr(
        career_input, "make_discovery", lambda: Discovery(transport=httpx.MockTransport(discovery))
    )
    monkeypatch.setattr(
        github_ingestion,
        "make_github_client",
        lambda: GitHubClient(transport=httpx.MockTransport(git)),
    )
    return SimpleNamespace(reviewer=reviewer, sources=sources)


async def account(client, database):
    credentials = {"email": f"fresh-{uuid4().hex[:8]}@example.com", "password": "Fresh-Demo-2026"}
    assert (await client.post("/auth/register", json=credentials)).status_code == 200
    login = await client.post(
        "/auth/login",
        data={"username": credentials["email"], "password": credentials["password"]},
    )
    assert login.status_code == 200, login.text
    headers = {"Authorization": "Bearer " + login.json()["access_token"]}
    profile = await client.post("/profiles", headers=headers, data={"github_username": "example"})
    assert profile.status_code == 200, profile.text
    async with database() as db:
        uid = await db.scalar(select(User.id).where(User.email == credentials["email"]))
    return headers, profile.json()["id"], uid


async def prepared_input(client, database, headers, profile_id):
    created = await client.post(
        "/career-inputs",
        headers=headers,
        json={
            "profile_id": profile_id,
            "github_profile_url": "https://github.com/example",
            "job_description": JD,
        },
    )
    assert created.status_code == 201, created.text
    path = "/career-inputs/" + created.json()["id"]
    chosen = await client.put(
        path + "/selection", headers=headers, json={"repositories": list(REPOS)}
    )
    assert chosen.status_code == 200, chosen.text
    state = None
    for _ in range(12):  # Caller-driven sequential preparation; the worker runs ingestion jobs.
        state = (await client.post(path + "/prepare", headers=headers)).json()
        if state["status"] == "ready":
            break
        await tick(database, "fresh-e2e-worker")
    assert state["status"] == "ready", state
    return created.json()["id"], state["selection"]


async def index(client, database, headers, selection):
    for item in selection:
        response = await client.post(
            "/retrieval-indexes",
            headers=headers,
            json={"ingestion_id": item["job_id"], "configuration": "lmstudio"},
        )
        assert response.status_code == 202, response.text
        await tick(database, "fresh-e2e-indexer")
        state = await client.get(
            "/retrieval-indexes/" + response.json()["index_id"], headers=headers
        )
        assert state.json()["state"] == "ready", state.text
        assert state.json()["dimensions"] == 768 and state.json()["provider"] == "lmstudio"


async def assess(client, headers, input_id, mode="hybrid"):
    created = await client.post(
        f"/career-inputs/{input_id}/assessments",
        headers=headers,
        json={
            "request_id": str(uuid4()),
            "retrieval_mode": mode,
            "hosted_processing_consent": CONSENT,
        },
    )
    return created


async def match_all(client, headers, identity):
    row = None
    for _ in range(4):
        row = (await client.get(f"/career-assessments/{identity}", headers=headers)).json()
        nxt = row["requirements"][len(row["matches"])]["id"]
        response = await client.post(
            f"/career-assessments/{identity}/match",
            headers=headers,
            json={"requirement_id": nxt, "reviewed": True, "hosted_processing_consent": CONSENT},
        )
        assert response.status_code == 200, response.text
        row = response.json()
    return row


async def test_fresh_profile_and_job_reach_verified_matches_and_roadmap(client, database, offline):
    headers, profile_id, uid = await account(client, database)
    input_id, selection = await prepared_input(client, database, headers, profile_id)
    pinned = {s["full_name"]: s["commit_sha"] for s in selection}
    assert pinned == {name: offline.sources[name].sha for name in REPOS}
    await index(client, database, headers, selection)
    assert Embeddings.documents > 0 and not offline.reviewer.posts

    created = await assess(client, headers, input_id)
    assert created.status_code == 201, created.text
    identity = created.json()["id"]
    assert created.json()["status"] == "review" and created.json()["retrieval_mode"] == "hybrid"
    assert len(created.json()["requirements"]) == 4 and created.json()["inference_spent"] == 2

    done = await match_all(client, headers, identity)
    assert done["status"] == "complete" and done["inference_spent"] == 10
    by_skill = {
        r["skill"].casefold(): next(m for m in done["matches"] if m["requirement_id"] == r["id"])
        for r in done["requirements"]
    }
    expected_paths = {
        "message broker": "orders/broker.py",
        "atomic database writes": "orders/store.py",
        "http request validation": "orders/api.py",
    }
    for skill, path in expected_paths.items():
        found = by_skill[skill]
        assert found["classification"] == "demonstrated", (skill, found)
        assert found["coverage"]["mode"] == "hybrid"
        cited = {c["path"] for c in found["citations"]}
        assert path in cited, (skill, cited)
        for citation in found["citations"]:
            repo = next(s for s in selection if s["repository_id"] == citation["repository_id"])
            assert citation["commit_sha"] == repo["commit_sha"]
            source = REPOS[repo["full_name"]][citation["path"]].decode().splitlines()
            assert citation["quote"] == "\n".join(
                source[citation["line_start"] - 1 : citation["line_end"]]
            )
    gap = by_skill["kubernetes"]
    assert gap["classification"] == "not evidenced" and gap["citations"] == []
    assert "not a judgment of personal ability" in gap["uncertainty"]

    # Ten calls just happened inside one rolling minute: the roadmap's two calls would exceed the
    # 10-per-minute cap. Refused up front (429), retryable, and no failed roadmap row is left.
    early = await client.post(
        f"/career-assessments/{identity}/roadmap",
        headers=headers,
        json={"hosted_processing_consent": CONSENT},
    )
    assert early.status_code == 429 and int(early.headers["Retry-After"]) >= 1, early.text
    assert len(offline.reviewer.posts) == 10
    missing = await client.get(f"/career-assessments/{identity}/roadmap", headers=headers)
    assert missing.status_code == 404
    async with database() as db:  # Simulate the minute elapsing (timestamps only).
        budget = await db.get(CareerAIBudget, 1)
        budget.recent = [t - 120 for t in budget.recent]
        await db.commit()
    roadmap = await client.post(
        f"/career-assessments/{identity}/roadmap",
        headers=headers,
        json={"hosted_processing_consent": CONSENT},
    )
    assert roadmap.status_code in {200, 201}, roadmap.text
    body = roadmap.json()
    assert body["status"] == "complete", body
    assert body["inference_spent"] == 12 and body["roadmap_inference_spent"] == 2
    recommendations = body["result"]["recommendations"]
    gap_id = next(r["id"] for r in done["requirements"] if r["skill"].casefold() == "kubernetes")
    assert [r["requirement_id"] for r in recommendations] == [gap_id]
    assert len(body["result"]["demonstrated_requirement_ids"]) == 3

    posts = len(offline.reviewer.posts)
    assert posts == 12
    for _ in range(2):  # Reopen: GET only, nothing regenerates or spends.
        again = await client.get(f"/career-assessments/{identity}", headers=headers)
        saved = await client.get(f"/career-assessments/{identity}/roadmap", headers=headers)
        assert again.json()["matches"] == done["matches"] and again.json()["inference_spent"] == 12
        assert saved.json()["result"] == body["result"]
    history = await client.get(f"/career-inputs/{input_id}/assessments", headers=headers)
    assert [x["id"] for x in history.json()["items"]] == [identity]
    assert len(offline.reviewer.posts) == posts
    async with database() as db:
        budget = await db.get(CareerAIBudget, 1)
        assert budget.spent == 12 and budget.owners == {str(uid): 12}

    # Another user cannot read any of it.
    other, _, _ = await account(client, database)
    assert (await client.get(f"/career-assessments/{identity}", headers=other)).status_code == 404
    assert (await client.get(f"/career-inputs/{input_id}", headers=other)).status_code == 404


async def test_hybrid_bridges_vocabulary_keyword_cannot(client, database, offline):
    headers, profile_id, uid = await account(client, database)
    input_id, selection = await prepared_input(client, database, headers, profile_id)
    await index(client, database, headers, selection)
    created = await assess(client, headers, input_id)
    manifest = created.json()["manifest"]["revisions"]
    requirement = created.json()["requirements"][0]
    assert requirement["skill"] == "message broker"
    async with database() as db:
        keyword, _, _, keyword_cov = await assessment.gather(
            db, uid, manifest, "keyword", requirement
        )
    async with database() as db:
        hybrid, _, _, hybrid_cov = await assessment.gather(db, uid, manifest, "hybrid", requirement)
    assert "orders/broker.py" not in {v.path for v in keyword.values()}
    broker = [v for v in hybrid.values() if v.path == "orders/broker.py"]
    assert broker and all(v.implementation_complete is True for v in broker)
    assert keyword_cov["mode"] == "keyword" and hybrid_cov["mode"] == "hybrid"


async def test_hybrid_requires_ready_indexes_and_never_falls_back(client, database, offline):
    headers, profile_id, _ = await account(client, database)
    input_id, _ = await prepared_input(client, database, headers, profile_id)
    denied = await assess(client, headers, input_id)
    assert denied.status_code == 409 and "index" in denied.json()["detail"].lower()
    assert not offline.reviewer.posts


@pytest.mark.parametrize("kind", ["owner", "aggregate"])
async def test_shared_caps_deny_a_two_call_step_before_any_request(client, database, offline, kind):
    headers, profile_id, uid = await account(client, database)
    input_id, selection = await prepared_input(client, database, headers, profile_id)
    await index(client, database, headers, selection)
    identity = (await assess(client, headers, input_id)).json()["id"]
    async with database() as db:
        budget = await db.get(CareerAIBudget, 1)
        if kind == "owner":  # 39 of 40: one attempt left cannot fund match + check.
            budget.owners = {str(uid): 39}
        else:  # 99 of 100.
            budget.spent = 99
        await db.commit()
    before = len(offline.reviewer.posts)
    row = (await client.get(f"/career-assessments/{identity}", headers=headers)).json()
    response = await client.post(
        f"/career-assessments/{identity}/match",
        headers=headers,
        json={
            "requirement_id": row["requirements"][0]["id"],
            "reviewed": True,
            "hosted_processing_consent": CONSENT,
        },
    )
    assert response.status_code == 409 and "fewer than two" in response.text, response.text
    assert len(offline.reviewer.posts) == before
    row = (await client.get(f"/career-assessments/{identity}", headers=headers)).json()
    assert row["status"] == "review" and row["matches"] == [] and row["error_category"] is None
    async with database() as db:
        budget = await db.get(CareerAIBudget, 1)
        assert (budget.owners[str(uid)] if kind == "owner" else budget.spent) == (
            39 if kind == "owner" else 99
        )


async def test_assessment_never_exceeds_twelve_and_matching_cannot_take_roadmap_slots(
    client, database, offline
):
    headers, profile_id, _ = await account(client, database)
    input_id, selection = await prepared_input(client, database, headers, profile_id)
    await index(client, database, headers, selection)
    identity = (await assess(client, headers, input_id)).json()["id"]
    done = await match_all(client, headers, identity)
    assert done["inference_spent"] == 10
    async with database() as db:  # Simulate an earlier failed roadmap that spent one attempt.
        row = await db.get(CareerAssessment, identity)
        row.calls = [*row.calls, {"phase": "roadmap/generate", "at": 0, "state": "consumed"}]
        await db.commit()
    before = len(offline.reviewer.posts)
    denied = await client.post(
        f"/career-assessments/{identity}/roadmap",
        headers=headers,
        json={"hosted_processing_consent": CONSENT},
    )
    assert denied.status_code == 409 and len(offline.reviewer.posts) == before
    async with database() as db:  # The database itself refuses a thirteenth attempt.
        from sqlalchemy.exc import DBAPIError

        with pytest.raises(DBAPIError):
            await db.execute(
                update(CareerAssessment)
                .where(CareerAssessment.id == identity)
                .values(calls=[{"phase": "x", "at": 0, "state": "consumed"}] * 13)
            )


async def _alembic(database, action, target):
    from alembic.config import Config

    from alembic import command

    async with database.kw["bind"].begin() as connection:

        def run(conn):
            cfg = Config("alembic.ini")
            cfg.attributes["connection"] = conn
            getattr(command, action)(cfg, target)

        await connection.run_sync(run)


async def test_existing_six_attempt_assessment_survives_migration_012_unchanged(
    client, database, offline
):
    from sqlalchemy import text

    headers, profile_id, _ = await account(client, database)
    input_id, selection = await prepared_input(client, database, headers, profile_id)
    await index(client, database, headers, selection)
    identity = (await assess(client, headers, input_id)).json()["id"]
    for _ in range(2):  # Two matches: 2 extraction + 2 x 2 = six recorded attempts.
        row = (await client.get(f"/career-assessments/{identity}", headers=headers)).json()
        await client.post(
            f"/career-assessments/{identity}/match",
            headers=headers,
            json={
                "requirement_id": row["requirements"][len(row["matches"])]["id"],
                "reviewed": True,
                "hosted_processing_consent": CONSENT,
            },
        )

    def frozen(r):
        return json.dumps(
            [r.calls, r.matches, r.requirements, r.snapshot, r.status, r.signature], sort_keys=True
        )

    async with database() as db:
        before = await db.get(CareerAssessment, identity)
        assert len(before.calls) == 6
        snapshot = frozen(before)
    await _alembic(database, "downgrade", "011")
    async with database() as db:
        assert await db.scalar(text("SELECT version_num FROM alembic_version")) == "011"
        assert frozen(await db.get(CareerAssessment, identity)) == snapshot
    await _alembic(database, "upgrade", "head")
    async with database() as db:
        assert await db.scalar(text("SELECT version_num FROM alembic_version")) == "012"
        row = await db.get(CareerAssessment, identity)
        assert frozen(row) == snapshot
        row.calls = [*row.calls, *[{"phase": "fixture", "at": 0, "state": "consumed"}] * 5]
        await db.commit()  # 11 attempts: valid only at 012.
    with pytest.raises(RuntimeError, match="more than 10 attempts"):
        await _alembic(database, "downgrade", "011")
    async with database() as db:
        assert await db.scalar(text("SELECT version_num FROM alembic_version")) == "012"
