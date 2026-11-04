"""Synthetic offline checker fixture; not an archived inference result."""

import json
from types import SimpleNamespace

from api.schemas.questions import GeneratedAnswer, checker_schema
from rag.generator import completeness, grounding


def contract_fixture(root):
    value = json.loads((root / "tests/fixtures/checker_contract.json").read_text())
    generated = GeneratedAnswer.model_validate(value["answer"])
    selected = {
        key: SimpleNamespace(path=item["path"], excerpt=item["text"])
        for key, item in value["evidence"].items()
    }
    payload = grounding.support_payload(value["question"], generated, selected)
    messages = [
        {"role": "system", "content": grounding.VERIFY},
        {"role": "user", "content": payload},
    ]
    decoded = json.loads(payload)
    assert decoded["requirements"] == completeness.requirements(value["question"])
    for claim in generated.claims:
        for ref in claim.references:
            assert ref.evidence_id in selected
            if ref.quote is not None:
                assert selected[ref.evidence_id].excerpt.count(ref.quote) == 1
    return generated, decoded, selected, messages, checker_schema()


def validate(content, root):
    generated, payload, selected, _, _ = contract_fixture(root)
    return completeness.validate_coverage(content, payload["question"], generated, selected)
