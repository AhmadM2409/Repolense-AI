import json
from uuid import UUID

import pytest

from api.schemas.questions import GeneratedAnswer
from rag.generator.completeness import requirements, validate_coverage
from rag.generator.grounding import validate_answer
from tests.unit.test_grounded_qa import row

QUESTION = "Which condition permits an upload and how many bytes are allowed?"
SELECTED = {
    "E1": row("return size <= MAX_PAYLOAD", "upload.py"),
    "E2": row("MAX_PAYLOAD = 8192", "options.py"),
}


def answer():
    return GeneratedAnswer.model_validate(
        {
            "status": "answered",
            "uncertainty": "",
            "claims": [
                {
                    "text": "Upload is permitted when size <= MAX_PAYLOAD.",
                    "kind": "observed",
                    "references": [{"evidence_id": "E1"}],
                },
                {
                    "text": "MAX_PAYLOAD is 8192 bytes.",
                    "kind": "observed",
                    "references": [{"evidence_id": "E2"}],
                },
            ],
        }
    )


def decision():
    return {
        "support": [
            {
                "claim_index": 0,
                "source_fact": "The cited source establishes this claim.",
                "verdict": "supported",
                "evidence_ids": ["E1"],
            },
            {
                "claim_index": 1,
                "source_fact": "The cited source establishes this claim.",
                "verdict": "supported",
                "evidence_ids": ["E2"],
            },
        ],
        "coverage": [
            {
                "requirement_id": "R1",
                "covered": True,
                "claim_indices": [0],
                "answer_quote": "size <= MAX_PAYLOAD",
                "evidence_ids": ["E1"],
            },
            {
                "requirement_id": "R2",
                "covered": True,
                "claim_indices": [1],
                "answer_quote": "8192 bytes",
                "evidence_ids": ["E2"],
            },
        ],
    }


def test_complete_supported_multifile_response_requires_every_gate():
    parsed, cites = validate_answer(answer().model_dump_json(), SELECTED, UUID(int=3), UUID(int=4))
    validate_coverage(json.dumps(decision()), QUESTION, parsed, SELECTED)
    assert {c.path for c in cites} == {"upload.py", "options.py"}


@pytest.mark.parametrize(
    "fault",
    [
        "drop",
        "duplicate",
        "rename",
        "missing",
        "index",
        "quote",
        "citation",
        "supported",
        "string_bool",
        "reuse",
        "numeric",
    ],
)
def test_no_aggregate_supported_flag_can_bypass_obligation_checks(fault):
    checked = decision()
    unit = checked["coverage"][1]
    if fault == "drop":
        checked["coverage"].pop()
    if fault == "duplicate":
        unit["requirement_id"] = "R1"
    if fault == "rename":
        unit["requirement_id"] = "R6"
    if fault == "missing":
        unit["covered"] = False
    if fault == "index":
        unit["claim_indices"] = [3]
    if fault == "quote":
        unit["answer_quote"] = "invented missing information"
    if fault == "citation":
        unit["evidence_ids"] = ["E1"]
    if fault == "supported":
        checked["supported"] = False
    if fault == "string_bool":
        unit["covered"] = "true"
    if fault == "reuse":
        unit.update(claim_indices=[0], answer_quote="size <= MAX_PAYLOAD", evidence_ids=["E1"])
    if fault == "numeric":
        unit.update(claim_indices=[0], answer_quote="MAX_PAYLOAD", evidence_ids=["E1"])
    with pytest.raises(ValueError):
        validate_coverage(json.dumps(checked), QUESTION, answer(), SELECTED)


def test_supported_partial_answer_rejected_even_if_model_says_every_part_covered():
    parsed = answer()
    parsed.claims.pop()
    checked = decision()
    checked["support"].pop()
    checked["coverage"][1].update(
        claim_indices=[0], answer_quote="MAX_PAYLOAD", evidence_ids=["E1"]
    )
    # Remaining claim is fully supported; absence of the requested value is independent.
    validate_answer(parsed.model_dump_json(), SELECTED, UUID(int=3), UUID(int=4))
    with pytest.raises(ValueError, match="Incomplete coverage"):
        validate_coverage(json.dumps(checked), QUESTION, parsed, SELECTED)


def test_nonnumeric_part_is_not_satisfied_by_reused_answer_span():
    checked = decision()
    checked["coverage"][1].update(
        claim_indices=[0], answer_quote="size <= MAX_PAYLOAD", evidence_ids=["E1"]
    )
    with pytest.raises(ValueError):
        validate_coverage(
            json.dumps(checked),
            "Which condition is checked and which function checks it?",
            answer(),
            SELECTED,
        )


def test_question_units_preserve_quoted_data_and_every_unquoted_part():
    parts = requirements('Compare "red and blue" and describe errors; identify the caller?')
    assert len(parts) == 3 and '"red and blue"' in parts[0]["text"]
    assert [p["requirement_id"] for p in parts] == ["R1", "R2", "R3"]
    with pytest.raises(ValueError):
        requirements(" and ".join(["part"] * 7))


@pytest.mark.parametrize("fault", ["missing", "duplicate", "verdict", "extra", "id", "unclaimed"])
def test_per_claim_support_is_required_even_when_aggregate_verdict_is_true(fault):
    value = decision()
    record = value["support"][1]
    if fault == "missing":
        value["support"].pop()
    if fault == "duplicate":
        record["claim_index"] = 0
    if fault == "verdict":
        record["verdict"] = "contradicted"
    if fault == "extra":
        record["invented_path"] = "foreign.py"
    if fault == "id":
        record["evidence_ids"] = ["E1"]
    if fault == "unclaimed":
        record["evidence_ids"].append("E1")
    with pytest.raises(ValueError):
        validate_coverage(json.dumps(value), QUESTION, answer(), SELECTED)


@pytest.mark.parametrize(
    "claim,conflict",
    [
        ("size >= MAX_PAYLOAD", True),
        ("size > MAX_PAYLOAD", True),
        ("size <= MAX_PAYLOAD", False),
        ("MAX_PAYLOAD >= size", False),
        ("unrelated > OTHER_BOUND", False),
    ],
)
def test_explicit_comparison_polarity_preserves_equivalent_operand_order(claim, conflict):
    from rag.generator.grounding import comparison_conflict

    assert comparison_conflict(claim, "return size <= MAX_PAYLOAD") is conflict


def test_cited_but_reversed_operator_is_rejected_before_model_support():
    value = answer()
    value.claims[0].text = "Upload is permitted when size >= MAX_PAYLOAD."
    with pytest.raises(ValueError, match="Contradictory comparison"):
        validate_answer(value.model_dump_json(), SELECTED, UUID(int=3), UUID(int=4))


@pytest.mark.parametrize("text", ["MAX_PAYLOAD is 8192.", "MAX_PAYLOAD is 8192 bytes."])
def test_numeric_value_with_sentence_punctuation_is_not_falsely_rejected(text):
    parsed = answer()
    parsed.claims[1].text = text
    checked = decision()
    checked["coverage"][1]["answer_quote"] = text
    validate_coverage(json.dumps(checked), QUESTION, parsed, SELECTED)


def test_cross_claim_answer_span_preserves_word_order_with_whitespace_only_normalization():
    parsed = answer()
    checked = decision()
    checked["coverage"] = [
        {
            "requirement_id": "R1",
            "covered": True,
            "claim_indices": [0, 1],
            "answer_quote": " ".join(c.text for c in parsed.claims),
            "evidence_ids": ["E1", "E2"],
        }
    ]
    validate_coverage(json.dumps(checked), "Explain this upload rule.", parsed, SELECTED)
    checked["coverage"][0]["answer_quote"] += " Invented extra fact."
    with pytest.raises(ValueError):
        validate_coverage(json.dumps(checked), "Explain this upload rule.", parsed, SELECTED)
