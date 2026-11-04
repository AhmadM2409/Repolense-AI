"""Phase-specific budgets and strict planner permissions; no network calls."""

import pytest

from core.services.investigation import Decision
from rag.generator.investigation_planner import ExecutionBudget, PhaseBudget
from rag.generator.local_provider import GenerationError


def test_reserve_acceptance_and_persist(tmp_path):
    path = tmp_path / "budget.sqlite3"
    dev = PhaseBudget(path)
    for _ in range(10):
        dev.reserve()
    with pytest.raises(GenerationError):
        PhaseBudget(path).reserve()
    with dev.connect() as db:
        db.execute("UPDATE requests SET at=0")
    final = PhaseBudget(path, "acceptance")
    for _ in range(10):
        final.reserve()
    with pytest.raises(GenerationError):
        PhaseBudget(path, "acceptance").reserve()
    with final.connect() as db:
        assert db.execute("SELECT count(*) FROM requests").fetchone()[0] == 20


def test_execution_limit_includes_all_operations(tmp_path):
    budget = ExecutionBudget(PhaseBudget(tmp_path / "cap.sqlite3"))
    for _ in range(6):
        budget.reserve()
    with pytest.raises(GenerationError):
        budget.reserve()
    assert budget.calls == 6


@pytest.mark.parametrize(
    "data",
    [
        {"operation": "fetch", "query": "http://evil"},
        {"operation": "search", "repository_id": "foreign"},
        {"operation": "answer", "reason": "x" * 161},
    ],
)
def test_planner_has_no_new_permissions(data):
    with pytest.raises(ValueError):
        Decision.model_validate(data)


@pytest.mark.asyncio
async def test_planner_uses_structured_generation_content():
    from types import SimpleNamespace

    from rag.generator.investigation_planner import Planner

    class Provider:
        async def generate(self, messages, before_request, **kwargs):
            await before_request()
            assert "UNTRUSTED" in messages[0]["content"]
            assert kwargs["response_schema"]["additionalProperties"] is False
            return SimpleNamespace(
                content=(
                    '{"operation":"search","query":"guard","reason":"Find the guard definition"}'
                )
            )

    planner = object.__new__(Planner)
    planner.provider = Provider()
    value = await planner("Describe guard.", ())
    assert value.operation == "search" and value.query == "guard"
