"""Isolated prototype contracts; no live model or production policy changes."""

import json
from uuid import UUID

import pytest

from api.main import app
from core.services import repository_qa
from tests.support.grounding import prototype as p
from rag.generator import grounding
from tests.unit.test_completeness import QUESTION, SELECTED, answer
from tests.unit.test_grounded_qa import row


def decision():
    return {
        "support": [{"claim_index": i, "verdict": "supported"} for i in range(2)],
        "coverage": [
            {"requirement_id": "R1", "complete": True, "claim_indices": [0]},
            {"requirement_id": "R2", "complete": True, "claim_indices": [1]},
        ],
    }


def test_citations_and_answer_text_are_bound_without_model_copying():
    value = answer()
    before = value.model_dump_json()
    result = p.validate_coverage(json.dumps(decision()), QUESTION, value, SELECTED)
    assert result.support[0].evidence_ids == ["E1"]
    assert result.support[1].evidence_ids == ["E2"]
    assert result.coverage[0].answer_quote == value.claims[0].text
    assert result.coverage[1].answer_quote == value.claims[1].text
    assert value.model_dump_json() == before


@pytest.mark.parametrize(
    "fault",
    [
        "index",
        "missing",
        "duplicate",
        "unit",
        "unsupported",
        "partial",
        "copy_ids",
        "copy_text",
        "string_bool",
        "irrelevant",
        "negative_index",
    ],
)
def test_all_independent_coverage_and_support_gates_remain(fault):
    value = decision()
    if fault == "index":
        value["coverage"][1]["claim_indices"] = [3]
    if fault == "missing":
        value["support"].pop()
    if fault == "duplicate":
        value["support"][1]["claim_index"] = 0
    if fault == "unit":
        value["coverage"].pop()
    if fault == "unsupported":
        value["support"][0]["verdict"] = "insufficient"
    if fault == "partial":
        value["coverage"][1]["complete"] = False
    if fault == "copy_ids":
        value["support"][0]["evidence_ids"] = ["E5"]
    if fault == "copy_text":
        value["coverage"][0]["answer_quote"] = "invented"
    if fault == "string_bool":
        value["coverage"][0]["complete"] = "true"
    if fault == "irrelevant":
        value["coverage"][1]["claim_indices"] = [0]
    if fault == "negative_index":
        value["coverage"][0]["claim_indices"] = [-1]
    with pytest.raises(ValueError):
        p.validate_coverage(json.dumps(value), QUESTION, answer(), SELECTED)


@pytest.mark.parametrize("verdict", ["contradicted", "insufficient", "invalid"])
def test_no_non_supported_verdict_publishes(verdict):
    value = decision()
    value["support"][0]["verdict"] = verdict
    with pytest.raises(ValueError):
        p.validate_coverage(json.dumps(value), QUESTION, answer(), SELECTED)


@pytest.mark.parametrize(
    "text,source,expected",
    [
        (
            "gate returns False when active is True.",
            "def gate(active):\n    return not active",
            True,
        ),
        (
            "When active is False, gate returns True.",
            "def gate(active):\n    return not active",
            True,
        ),
        (
            "gate returns True when active is True.",
            "def gate(active):\n    return not active",
            False,
        ),
        (
            "gate returns False when active is True. It uses external infrastructure.",
            "def gate(active):\n    return not active",
            False,
        ),
        (
            "gate returns False when active is True.",
            "@other\ndef gate(active):\n    return not active",
            False,
        ),
        (
            "gate returns False when active is True.",
            "async def gate(active):\n    return not active",
            False,
        ),
        (
            "gate returns False when active is True.",
            "def gate(active=True):\n    return not active",
            False,
        ),
        (
            "gate returns False when active is True.",
            "def gate(active):\n    if active:\n        return False\n    return True",
            False,
        ),
        (
            "gate returns False when active is True.",
            "def gate(active):\n    return not active\nother()",
            False,
        ),
        ("CAP is 9.", "CAP = 9", True),
        ("CAP is 9.", "CAP = 9\nCAP += 1", False),
        ("CAP is 1.", "CAP = True", False),
        ("CAP is 10.", "CAP = 9", False),
        ('state returns "waiting".', 'def state():\n    return "waiting"', True),
        ('state returns "waiting".', "def state():\n    return other()", False),
    ],
)
def test_static_proof_has_finite_full_claim_and_ast_preconditions(text, source, expected):
    assert p.static_fact(text, source) is expected


def test_static_interpretation_preserves_original_kind_text_and_citations():
    source = {"E1": row("def gate(active):\n    return not active", "gate.py")}
    value = {
        "status": "answered",
        "uncertainty": "",
        "claims": [
            {
                "text": "gate returns False when active is True.",
                "kind": "inference",
                "references": [{"evidence_id": "E1"}],
            }
        ],
    }
    encoded = json.dumps(value)
    with pytest.raises(ValueError, match="Unsupported answer"):
        grounding.validate_answer(encoded, source, UUID(int=3), UUID(int=4))
    parsed, citations = p.validate_answer(encoded, source, UUID(int=3), UUID(int=4))
    assert parsed.model_dump() == p.GeneratedAnswer.model_validate(value).model_dump()
    assert citations[0].quote == source["E1"].excerpt


@pytest.mark.parametrize(
    "fault", ["runtime", "quote", "id", "duplicate", "uncertain_is_not_support"]
)
def test_no_general_inference_or_reference_bypass(fault):
    source = {"E1": row("def gate(active):\n    return not active", "gate.py")}
    value = {
        "status": "answered",
        "uncertainty": "",
        "claims": [
            {
                "text": "gate returns False when active is True.",
                "kind": "inference",
                "references": [{"evidence_id": "E1"}],
            }
        ],
    }
    if fault == "runtime":
        value["claims"][0]["text"] = "The runtime always calls gate with True."
    if fault == "quote":
        value["claims"][0]["references"][0]["quote"] = "fabricated"
    if fault == "id":
        value["claims"][0]["references"][0]["evidence_id"] = "E5"
    if fault == "duplicate":
        value["claims"][0]["references"] *= 2
    if fault == "uncertain_is_not_support":
        value["uncertainty"] = "Runtime configuration is unknown."
        parsed, _ = p.validate_answer(json.dumps(value), source, UUID(int=3), UUID(int=4))
        with pytest.raises(ValueError):
            p.validate_coverage(
                json.dumps(
                    {
                        "support": [{"claim_index": 0, "verdict": "insufficient"}],
                        "coverage": [
                            {"requirement_id": "R1", "complete": True, "claim_indices": [0]}
                        ],
                    }
                ),
                "Describe gate.",
                parsed,
                source,
            )
        return
    with pytest.raises(ValueError):
        p.validate_answer(json.dumps(value), source, UUID(int=3), UUID(int=4))


def test_private_service_reuses_code_without_mutating_production_globals_or_routes():
    old = dict(repository_qa.answer.__globals__)
    routes = [(r.path, r.endpoint) for r in app.routes if hasattr(r, "endpoint")]
    private = p.make_answer()
    assert private.__code__ is repository_qa.answer.__code__
    assert private.__globals__ is not repository_qa.answer.__globals__
    assert private.__globals__["validate_answer"] is p.validate_answer
    assert repository_qa.answer.__globals__ == old
    assert [(r.path, r.endpoint) for r in app.routes if hasattr(r, "endpoint")] == routes
    assert p.make_private_app(private) is not app


def test_experimental_system_suffix_is_charged_to_existing_context_cap():
    match = row("VALUE = 2", "config.py")
    selected, payload = p.select_evidence([match], "Describe VALUE.", 4096)
    assert selected and len(p.SYSTEM.encode()) + len(payload.encode()) <= 4096
    selected, _ = p.select_evidence([match], "Describe VALUE.", len(grounding.SYSTEM.encode()))
    assert not selected
