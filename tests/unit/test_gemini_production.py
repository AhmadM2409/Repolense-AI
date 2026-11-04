import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from api.schemas.questions import GeneratedAnswer
from rag.generator.grounded_policy import comparison_obligation, validate_coverage
from rag.generator.hosted import PersistentBudget
from rag.generator.local_provider import GenerationError


def test_persistent_atomic_budget_and_restart(tmp_path):
    path = tmp_path / "ledger.sqlite3"

    def submit(_):
        try:
            return PersistentBudget(path).reserve()["number"]
        except GenerationError:
            return None

    # Initialize schema once before independent concurrent clients.
    PersistentBudget(path)
    with ThreadPoolExecutor(max_workers=4) as pool:
        values = list(pool.map(submit, range(14)))
    assert len([v for v in values if v]) == 10 and len(set(v for v in values if v)) == 10
    with pytest.raises(GenerationError):
        PersistentBudget(path).reserve()
    with sqlite3.connect(path) as db:
        db.execute("UPDATE requests SET at=0")
        for _ in range(50):
            db.execute('INSERT INTO requests(at,stage,metadata) VALUES(0,"development","{}")')
    with pytest.raises(GenerationError):
        PersistentBudget(path).reserve()
    final = PersistentBudget(path, "acceptance")
    final.reserve()
    final.stopped = True
    with pytest.raises(GenerationError):
        PersistentBudget(path, "acceptance").reserve()


def test_production_ledger_never_stores_source_or_receipts(tmp_path):
    budget = PersistentBudget(tmp_path / "ledger.sqlite3")
    row = budget.reserve()
    row.update(
        content="PRIVATE_SOURCE",
        receipt={"text": "PRIVATE_SOURCE"},
        usage={"totalTokenCount": 5},
        state="completed",
    )
    budget.save()
    with budget.connect() as db:
        value = db.execute("SELECT metadata FROM requests").fetchone()[0]
    assert "PRIVATE_SOURCE" not in value and json.loads(value)["usage"]["totalTokenCount"] == 5


@pytest.mark.parametrize(
    "operator,words",
    [
        (">", "greater than"),
        ("<", "less than"),
        (">=", "at least"),
        ("<=", "at most"),
        ("==", "equal"),
        ("!=", "not equal"),
    ],
)
def test_comparison_obligation_is_general_and_not_just_a_citation(operator, words):
    selected = {
        "E1": SimpleNamespace(
            path="fresh.py", excerpt=f"def permits(x):\n    return x {operator} 9"
        )
    }

    def answer(text):
        return GeneratedAnswer.model_validate(
            {
                "status": "answered",
                "uncertainty": "",
                "claims": [
                    {"text": text, "kind": "observed", "references": [{"evidence_id": "E1"}]}
                ],
            }
        )

    comparison_obligation(
        "What comparison does permits use?",
        answer(f"permits checks whether x is {words} 9."),
        selected,
    )
    with pytest.raises(ValueError):
        comparison_obligation(
            "What comparison does permits use?", answer("permits compares x to 9."), selected
        )


def test_compact_checker_cannot_change_binding_or_override_negative():
    from tests.unit.test_completeness import QUESTION, SELECTED, answer

    for verdict in ["contradicted", "insufficient", "invalid"]:
        value = {
            "support": [{"claim_index": 0, "verdict": verdict}],
            "coverage": [{"requirement_id": "R1", "complete": True, "claim_indices": [0]}],
        }
        with pytest.raises(ValueError):
            validate_coverage(json.dumps(value), QUESTION, answer(), SELECTED)


def test_runtime_ledger_storage_failure_is_sanitized(tmp_path):
    parent = tmp_path / "not-a-directory"
    parent.write_text("synthetic")
    with pytest.raises(GenerationError) as error:
        PersistentBudget(parent / "ledger.sqlite3")
    assert error.value.code == "budget_storage"


def test_acceptance_reserve_and_total_cap_cannot_reset_on_restart(tmp_path):
    path = tmp_path / "full-ledger.sqlite3"
    PersistentBudget(path)
    with sqlite3.connect(path) as db:
        for stage, count in [("development", 60), ("acceptance", 40)]:
            db.executemany(
                "INSERT INTO requests(at,stage,metadata) VALUES(0,?,?)", [(stage, "{}")] * count
            )
    for stage in ["development", "acceptance"]:
        with pytest.raises(GenerationError):
            PersistentBudget(path, stage).reserve()


@pytest.mark.asyncio
async def test_auth_failure_latches_stop_without_retry(tmp_path):
    import httpx

    from rag.generator.gemini_provider import Gemini

    requests = []

    def external(request):
        requests.append(request)
        return httpx.Response(401, text="DO_NOT_LOG_PROVIDER_BODY")

    budget = PersistentBudget(tmp_path / "auth.sqlite3")
    provider = Gemini("TEST_ONLY", budget, httpx.MockTransport(external))

    async def reserve():
        pass

    for _ in range(2):
        with pytest.raises(GenerationError):
            await provider.generate(
                [
                    {"role": "system", "content": "synthetic"},
                    {"role": "user", "content": "synthetic"},
                ],
                reserve,
                response_schema={},
            )
    assert len(requests) == 1 and budget.stopped


def test_compact_positive_checker_cannot_publish_missing_comparison_direction():
    selected = {
        "E1": SimpleNamespace(
            path="fresh_gate.py", excerpt="def decide(total):\n    return total > 19"
        )
    }
    answer = GeneratedAnswer.model_validate(
        {
            "status": "answered",
            "uncertainty": "",
            "claims": [
                {
                    "text": "decide compares total to 19.",
                    "kind": "observed",
                    "references": [{"evidence_id": "E1"}],
                }
            ],
        }
    )
    checker = {
        "support": [{"claim_index": 0, "verdict": "supported"}],
        "coverage": [{"requirement_id": "R1", "complete": True, "claim_indices": [0]}],
    }
    with pytest.raises(ValueError, match="Incomplete coverage"):
        validate_coverage(json.dumps(checker), "What comparison does decide use?", answer, selected)


# Deferred-stabilization regression (2026-11-04): a phrase that merely CONTAINS the expected
# relation's words (or negates it) must not satisfy the comparison obligation.
@pytest.mark.parametrize(
    "operator,wrong,right",
    [
        ("<", "less than or equal to", "below"),
        ("<", "not less than", "less than"),
        (">", "greater than or equal to", "more than"),
        (">", "not greater than", "greater than"),
        ("==", "not equal to", "equal to"),
        ("==", "less than or equal to", "equal to"),
        ("<=", "less than", "no more than"),
        (">=", "greater than", "no less than"),
        ("!=", "equal to", "different from"),
    ],
)
def test_comparison_obligation_rejects_wrong_or_negated_relation(operator, wrong, right):
    selected = {
        "E1": SimpleNamespace(
            path="fresh.py", excerpt=f"def permits(x):\n    return x {operator} 9"
        )
    }

    def answer(text):
        return GeneratedAnswer.model_validate(
            {
                "status": "answered",
                "uncertainty": "",
                "claims": [
                    {"text": text, "kind": "observed", "references": [{"evidence_id": "E1"}]}
                ],
            }
        )

    question = "What comparison does permits use?"
    with pytest.raises(ValueError, match="Incomplete coverage"):
        comparison_obligation(question, answer(f"permits accepts x {wrong} 9."), selected)
    comparison_obligation(question, answer(f"permits accepts x {right} 9."), selected)
