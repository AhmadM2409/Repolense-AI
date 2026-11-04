"""Frozen deterministic J3 contract checks, not model quality measurements."""

import json
from types import SimpleNamespace
from uuid import uuid4

import pytest
from pydantic import ValidationError

from api.schemas.career_roadmap import RoadmapCreate
from core.services import career_roadmap_policy as policy

RID = "R123456789abc"


def parent(label="partially demonstrated", priority="required"):
    req = dict(
        id=RID,
        quote="Python testing required.",
        skill="python",
        aliases=[],
        priority=priority,
        depth="",
        start=0,
        end=24,
        job_hash="a" * 64,
    )
    repo, rev = str(uuid4()), str(uuid4())
    cite = dict(
        evidence_id="E1",
        repository_id=repo,
        revision_id=rev,
        source_id=str(uuid4()),
        chunk_id=str(uuid4()),
        commit_sha="a" * 40,
        path="python.py",
        line_start=1,
        line_end=2,
        quote="def f():\n    return True",
        evidence_kind="code",
    )
    match = dict(
        requirement_id=RID,
        classification=label,
        explanation="The source defines a Python function.",
        uncertainty="Testing behavior not established.",
        citations=[] if label == "not evidenced" else [cite],
        support_verdict="supported" if label != "not evidenced" else "insufficient",
        coverage={"context_limited": True},
    )
    return SimpleNamespace(
        requirements=[req],
        matches=[match],
        snapshot={
            "revisions": [
                dict(
                    repository_id=repo,
                    revision_id=rev,
                    commit_sha="a" * 40,
                    full_name="example/project",
                )
            ]
        },
    )


def draft():
    return dict(
        requirement_id=RID,
        action_type="testing",
        action="Add proposed Python regression tests for the quoted function.",
        gap="Automated test behavior remains unevidenced.",
        rationale="The job requires Python testing evidence.",
        completion_proof="Publish executable tests showing expected return behavior.",
        project_citation=0,
    )


def check():
    return dict(
        complete=True,
        items=[
            dict(
                requirement_id=RID,
                grounded=True,
                individualized=True,
                gap_accurate=True,
                future_only=True,
                observable_proof=True,
                project_supported=True,
                action_type_appropriate=True,
            )
        ],
    )


def bind(p=None, d=None, c=None):
    return policy.bind(
        p or parent(), json.dumps({"items": [d or draft()]}), json.dumps(c or check())
    )


def test_frozen_cases_exist():
    from pathlib import Path

    cases = json.loads(Path("tests/fixtures/career_roadmap_cases_v1.json").read_text())
    assert len(cases["cases"]) == 7 and "NOT_LIVE_QUALITY" in cases["status"]


@pytest.mark.parametrize("label", ["partially demonstrated", "documented/claimed", "not evidenced"])
def test_bound_requirement_evidence_and_future_work(label):
    p = parent(label)
    d = draft()
    if label == "not evidenced":
        d.update(project_citation=None, action_type="learning")
    item = bind(p, d)["recommendations"][0]
    assert item["requirement"] == p.requirements[0] and item["observation"] == p.matches[0]
    assert item["effort"] == "unknown" and item["priority"] == "high"
    assert item["proposal_notice"].startswith("Proposed future")
    assert (item["project"] is None) == (label == "not evidenced")


def test_preferred_priority_and_demonstrated_no_action():
    assert bind(parent(priority="preferred"))["recommendations"][0]["priority"] == "medium"
    p = parent("demonstrated")
    assert policy.eligible(p) == []
    assert policy.result(p, [])["demonstrated_requirement_ids"] == [RID]


@pytest.mark.parametrize(
    "field",
    [
        "grounded",
        "individualized",
        "gap_accurate",
        "future_only",
        "observable_proof",
        "project_supported",
        "action_type_appropriate",
    ],
)
def test_each_independent_check_is_mandatory(field):
    c = check()
    c["items"][0][field] = False
    with pytest.raises(ValueError):
        bind(c=c)


@pytest.mark.parametrize(
    "change",
    [
        {"requirement_id": "Raaaaaaaaaaaa"},
        {"project_citation": 4},
        {"action": "You lack Python expertise and must study."},
        {"rationale": "This guarantees you will get hired."},
        {"gap": "Your fit score is 85%."},
        {"action_type": "hiring"},
    ],
)
def test_invalid_or_fabricated_output(change):
    d = draft()
    d.update(change)
    with pytest.raises(ValueError):
        bind(d=d)


def test_missing_duplicate_foreign_and_incomplete_check():
    for c in [
        dict(complete=False, items=check()["items"]),
        dict(complete=True, items=[]),
        dict(complete=True, items=check()["items"] * 2),
    ]:
        with pytest.raises(ValueError):
            bind(c=c)
    p = parent()
    p.snapshot["revisions"] = []
    with pytest.raises(ValueError):
        bind(p)
    p = parent()
    p.matches = []
    with pytest.raises(ValueError):
        policy.eligible(p)


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"hosted_processing_consent": "gemini-source-processing-v1"},
        {"hosted_processing_consent": "career-gemini-job-and-source-v1", "reset_budget": True},
    ],
)
def test_no_consent_or_limit_bypass(body):
    with pytest.raises(ValidationError):
        RoadmapCreate.model_validate(body)


@pytest.mark.parametrize("kind", ["learning", "coding", "testing", "documenting", "deployment"])
def test_each_supported_action_type_retains_proof_and_uncertainty(kind):
    d = draft()
    d["action_type"] = kind
    value = bind(d=d)["recommendations"][0]
    assert (
        value["action_type"] == kind and value["completion_proof"] and value["effort"] == "unknown"
    )
