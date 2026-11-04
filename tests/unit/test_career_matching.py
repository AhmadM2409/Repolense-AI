"""Frozen deterministic J2 policy examples, not real Gemini quality evidence."""

import json
from types import SimpleNamespace
from uuid import uuid4

import pytest

from api.schemas.career import content_hash
from core.services import career_evidence as policy

JOB = "Python required. SQL preferred."
DRAFT = {
    "requirements": [
        {
            "quote": "Python required.",
            "skill": "PY",
            "aliases": ["python", "python"],
            "priority": "required",
            "depth": "",
        }
    ]
}


def test_trace_normalize_deduplicate_and_stable_ids():
    raw = json.dumps({"requirements": DRAFT["requirements"] * 2})
    values = policy.requirements(raw, JOB, content_hash(JOB))
    assert (
        len(values) == 1 and values[0]["skill"] == "python" and values[0]["aliases"] == ["python"]
    )
    assert JOB[values[0]["start"] : values[0]["end"]] == values[0]["quote"]
    assert policy.requirements(raw, JOB, content_hash(JOB)) == values
    good = {"complete": True, "items": [{"requirement_id": values[0]["id"], "supported": True}]}
    policy.validate_extraction(json.dumps(good), values)
    for bad in [
        {**good, "complete": False},
        {**good, "items": []},
        {**good, "items": [{**good["items"][0], "supported": False}]},
    ]:
        with pytest.raises(ValueError):
            policy.validate_extraction(json.dumps(bad), values)


@pytest.mark.parametrize(
    "change",
    [
        {"quote": "Kubernetes required."},
        {"priority": "other"},
        {"skill": ""},
        {"aliases": ["x" * 61]},
    ],
)
def test_invalid_or_invented_extraction(change):
    raw = json.dumps({"requirements": [{**DRAFT["requirements"][0], **change}]})
    with pytest.raises(ValueError):
        policy.requirements(raw, JOB, content_hash(JOB))


def evidence(path="main.py", excerpt="def python_api():\n    return True"):
    row = SimpleNamespace(
        path=path,
        excerpt=excerpt,
        source_id=uuid4(),
        chunk_id=uuid4(),
        commit_sha="a" * 40,
        line_start=1,
        line_end=len(excerpt.splitlines()),
    )
    return {"E1": row}, {"E1": (str(uuid4()), str(uuid4()))}


@pytest.mark.parametrize(
    "path,kind",
    [
        ("README.md", "documentation"),
        ("package.json", "configuration"),
        ("tests/test_api.py", "test"),
        ("main.py", "code"),
        ("claim.py", "documentation"),
    ],
)
def test_kind_distinguishes_implementation(path, kind):
    excerpt = '"""Claims Python support"""' if path == "claim.py" else "def api():\n    return True"
    assert policy.kind(path, excerpt) == kind


def result(label="demonstrated", claim="The code defines a Python function."):
    return {
        "classification": label,
        "claim": claim,
        "references": [{"evidence_id": "E1"}],
        "uncertainty": "Bounded source evidence only.",
    }


def checker(label="demonstrated", complete=True, verdict="supported"):
    return {"classification": label, "complete": complete, "verdict": verdict}


def test_positive_canonical_refs_and_partial():
    selected, scopes = evidence()
    accepted = policy.accepted(json.dumps(result()), json.dumps(checker()), selected, scopes)
    assert accepted["citations"][0]["repository_id"] == scopes["E1"][0]
    assert accepted["citations"][0]["quote"] == selected["E1"].excerpt
    partial = policy.accepted(
        json.dumps(result("partially demonstrated")),
        json.dumps(checker("partially demonstrated", False)),
        selected,
        scopes,
    )
    assert partial["classification"] == "partially demonstrated"


@pytest.mark.parametrize("kind", ["README.md", "package.json", "tests/test_main.py"])
def test_declarations_and_tests_cannot_be_demonstrated(kind):
    selected, scopes = evidence(kind)
    with pytest.raises(ValueError):
        policy.accepted(json.dumps(result()), json.dumps(checker()), selected, scopes)
    accepted = policy.accepted(
        json.dumps(result("documented/claimed")),
        json.dumps(checker("documented/claimed", False)),
        selected,
        scopes,
    )
    assert accepted["classification"] == "documented/claimed"


@pytest.mark.parametrize(
    "draft,check",
    [
        (result(), checker(complete=False)),
        (result(), checker(verdict="contradicted")),
        (result(), checker(verdict="insufficient")),
        (result(claim="python_api returns False."), checker()),
        ({**result(), "references": [{"evidence_id": "E5"}]}, checker()),
        (
            {**result(), "references": [{"evidence_id": "E1", "quote": "fabricated quote"}]},
            checker(),
        ),
    ],
)
def test_negative_incomplete_contradictory_or_fake_evidence(draft, check):
    selected, scopes = evidence()
    with pytest.raises(ValueError):
        policy.accepted(json.dumps(draft), json.dumps(check), selected, scopes)


def test_abstention_not_blanket_rejection():
    draft = result("not evidenced", "")
    draft["references"] = []
    check = checker("not evidenced", False, "insufficient")
    accepted = policy.accepted(json.dumps(draft), json.dumps(check), {}, {})
    assert not accepted["citations"] and "personal ability" in accepted["uncertainty"]
    with pytest.raises(ValueError):
        policy.accepted(json.dumps(draft), json.dumps(checker()), {}, {})


@pytest.mark.parametrize("claim", ["The fit is 1%.", "The developer lacks Python."])
def test_no_fit_percentage_or_personal_skill_deficit(claim):
    selected, scopes = evidence(excerpt="def python_api():\n    return 1")
    with pytest.raises(ValueError):
        policy.accepted(json.dumps(result(claim=claim)), json.dumps(checker()), selected, scopes)


def test_duplicate_aliases_merge_without_losing_meaning():
    first = {**DRAFT["requirements"][0], "aliases": ["py"]}
    second = {**first, "aliases": ["python"]}
    values = policy.requirements(
        json.dumps({"requirements": [first, second]}), JOB, content_hash(JOB)
    )
    assert len(values) == 1 and values[0]["aliases"] == ["py", "python"]


@pytest.mark.parametrize(
    "job,quotes",
    [
        (
            "Provide a Go function to retrieve a value from a JSON document.",
            ["Provide a Go function"],
        ),
        ("Build a parser that rejects invalid JSON.", ["a parser that rejects invalid JSON"]),
        ("Retrieve customer records from PostgreSQL.", ["Retrieve customer records"]),
        ("Process events without losing ordering.", ["Process events"]),
        ("Process events\nwithout losing ordering.", ["Process events"]),
        ("Return results only when latency < 67 ms.", ["Return results"]),
        ("Build an API and persist audit events atomically.", ["Build an API"]),
        (
            "Build an API and persist audit events atomically.",
            ["Build an API", "persist audit events"],
        ),
        ("Require latency < 67 ms.", ["Require latency", "67 ms"]),
        ("Reject value != 333.", ["Reject value", "333"]),
        ("Use offset -333.", ["Use offset", "333"]),
        ("Python required, SQL preferred.", ["Python", "SQL"]),
    ],
)
def test_rejects_truncated_action_object_constraint_or_connected_clause(job, quotes):
    rows = [{**DRAFT["requirements"][0], "quote": q} for q in quotes]
    with pytest.raises(ValueError, match="omit source clause content"):
        policy.requirements(json.dumps({"requirements": rows}), job, content_hash(job))


@pytest.mark.parametrize(
    "job,quotes",
    [
        (
            "Provide a Go function to retrieve a value from a JSON document.",
            ["Provide a Go function to retrieve a value from a JSON document."],
        ),
        (
            "Build an API and persist audit events atomically.",
            ["Build an API and persist audit events atomically."],
        ),
        (
            "Build an API and persist audit events atomically.",
            ["Build an API", "persist audit events atomically"],
        ),
        ("Python required; SQL preferred.", ["Python required", "SQL preferred"]),
        (
            "- Build an API\n- Persist events atomically",
            ["Build an API", "Persist events atomically"],
        ),
        ("Use Python 3.11 and .NET 8.", ["Use Python 3.11 and .NET 8."]),
        ("Reject value != 3.", ["Reject value != 3."]),
        ("Use offset -3.", ["Use offset -3."]),
        ("Process events\nwithout losing ordering.", ["Process events\nwithout losing ordering."]),
        ("Welcome to our team. Build an API.", ["Build an API."]),
        ("Build an API.", ["Build an API"]),
    ],
)
def test_full_obligations_allow_combined_split_layout_and_nontechnical_intro(job, quotes):
    rows = [
        {**DRAFT["requirements"][0], "quote": q, "skill": "API", "depth": "implementation"}
        for q in quotes
    ]
    values = policy.requirements(json.dumps({"requirements": rows}), job, content_hash(job))
    # Synonymous labels/depth are independently checked, not rejected by word overlap.
    check = {
        "complete": True,
        "items": [{"requirement_id": r["id"], "supported": True} for r in values],
    }
    policy.validate_extraction(json.dumps(check), values)
    assert len(values) == len(quotes)


def test_live_omission_rejected_even_with_false_positive_independent_checker():
    job = (
        "Provide a Go function to retrieve a value from a JSON document. "
        "Include Kubernetes deployment manifests."
    )
    raw = {
        "requirements": [
            {
                **DRAFT["requirements"][0],
                "quote": "Provide a Go function",
                "skill": "Go",
                "depth": "function development",
            },
            {
                **DRAFT["requirements"][0],
                "quote": "Include Kubernetes deployment manifests",
                "skill": "Kubernetes",
                "depth": "manifest creation",
            },
        ]
    }
    with pytest.raises(ValueError, match="omit source clause content"):
        policy.requirements(json.dumps(raw), job, content_hash(job))
