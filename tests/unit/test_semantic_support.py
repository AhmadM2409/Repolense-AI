import json
from uuid import UUID

import pytest

from api.schemas.questions import GeneratedAnswer
from rag.generator.completeness import validate_coverage
from rag.generator.grounding import support_payload, validate_answer
from rag.generator.literal_support import literal_conflict
from tests.unit.test_completeness import QUESTION, SELECTED, answer, decision
from tests.unit.test_grounded_qa import row


@pytest.mark.parametrize("verdict", ["contradicted", "insufficient", "invalid", "relevant", True])
def test_only_explicit_supported_verdict_can_publish(verdict):
    value = decision()
    value["support"][0]["verdict"] = verdict
    with pytest.raises(ValueError):
        validate_coverage(json.dumps(value), QUESTION, answer(), SELECTED)


def test_source_fact_does_not_override_negative_verdict_or_citation_identity():
    value = decision()
    value["support"][0].update(
        source_fact="This looks highly relevant and correct.", verdict="contradicted"
    )
    with pytest.raises(ValueError, match="Contradicted"):
        validate_coverage(json.dumps(value), QUESTION, answer(), SELECTED)
    value = decision()
    value["support"][0]["source_fact"] = " "
    with pytest.raises(ValueError):
        validate_coverage(json.dumps(value), QUESTION, answer(), SELECTED)


@pytest.mark.parametrize(
    "qualifier",
    [
        "not",
        "never",
        "only",
        "unless",
        "before",
        "after",
        "until",
        "always",
        "sometimes",
        "on success",
        "on failure",
    ],
)
def test_claim_qualifiers_reach_checker_without_normalization(qualifier):
    parsed = answer()
    parsed.claims[0].text = "Operation " + qualifier + " changes state."
    payload = json.loads(support_payload(QUESTION, parsed, SELECTED))
    assert payload["claims"][0]["text"] == parsed.claims[0].text


@pytest.mark.parametrize(
    "claim,source,conflict",
    [
        ("ENABLED is True.", "ENABLED = False", True),
        ("ENABLED is False.", "ENABLED = False", False),
        ("ENABLED is 0.", "ENABLED = False", True),
        ("THREADS equals 9.", "THREADS = 8\nOTHER = 9", True),
        ("THREADS is 8.", "THREADS = 8", False),
        ('BACKEND is "remote".', 'BACKEND = "local"', True),
        ('state returns "active".', 'def state():\n    return "pending"', True),
        ('state returns "pending".', 'def state():\n    return "pending"', False),
        (
            "state returns True.",
            "def state():\n    if external():\n        return False\n    return True",
            False,
        ),
        ("FLAG is True.", "FLAG = False\nFLAG = external()", False),
        ("FLAG is True.", "FLAG = False\nFLAG = True", False),
    ],
)
def test_only_unambiguous_literal_facts_supply_deterministic_conflicts(claim, source, conflict):
    assert literal_conflict(claim, source) is conflict


def test_source_call_order_is_not_claimed_as_a_static_proof():
    source = 'def finalize():\n    store()\n    return "receipt"'
    assert not literal_conflict("finalize returns before store.", source)


def test_wrong_config_value_with_valid_citation_is_rejected_before_inference():
    value = GeneratedAnswer.model_validate(
        {
            "status": "answered",
            "uncertainty": "",
            "claims": [
                {
                    "text": "ENABLED is True.",
                    "kind": "observed",
                    "references": [{"evidence_id": "E1"}],
                }
            ],
        }
    )
    with pytest.raises(ValueError, match="Contradictory literal"):
        validate_answer(
            value.model_dump_json(),
            {"E1": row("ENABLED = False", "config.py")},
            UUID(int=3),
            UUID(int=4),
        )


def test_definition_facts_are_bounded_and_never_generalize_call_order_or_branches():
    from rag.generator.literal_support import literal_facts

    facts = literal_facts("def allowed(active):\n    return not active")
    assert facts == ["Definition allowed: active=False returns True; active=True returns False"]
    assert literal_facts('def finish():\n    persist()\n    return "receipt"') == []
    assert literal_facts("def allowed(x):\n    if x:\n        return True\n    return False") == []
    assert literal_facts("@external\ndef allowed(x):\n    return not x") == []
    assert len(literal_facts("\n".join(f"C{i} = {i}" for i in range(20)))) <= 4


def test_answer_span_cosmetics_preserve_material_words_and_literal_contents():
    from rag.generator.completeness import answer_span

    assert answer_span('returns "retry" on failure') == answer_span("returns 'retry' on failure")
    assert answer_span('does not return "retry"') != answer_span('does return "retry"')
    assert answer_span('returns "True"') != answer_span("returns True")
    assert answer_span("before publish") != answer_span("after publish")


def test_malformed_numeric_literal_fails_closed_without_syntax_error():
    with pytest.raises(ValueError, match="Unsupported literal"):
        literal_conflict("PORT is 05.", "PORT = 5")


@pytest.mark.parametrize("length", [161, 512])
def test_source_fact_metadata_accepts_bounded_function_summary(length):
    value = decision()
    value["support"][0]["source_fact"] = "x" * length
    validate_coverage(json.dumps(value), QUESTION, answer(), SELECTED)


def test_source_fact_metadata_still_rejects_over_bound():
    value = decision()
    value["support"][0]["source_fact"] = "x" * 513
    with pytest.raises(ValueError):
        validate_coverage(json.dumps(value), QUESTION, answer(), SELECTED)


@pytest.mark.parametrize("verdict", ["contradicted", "insufficient", "invalid"])
def test_long_source_summary_cannot_override_negative_verdict(verdict):
    value = decision()
    value["support"][0].update(source_fact="x" * 512, verdict=verdict)
    with pytest.raises(ValueError):
        validate_coverage(json.dumps(value), QUESTION, answer(), SELECTED)
