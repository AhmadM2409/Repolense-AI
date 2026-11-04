"""Finite readiness cases; no model verdict is used to grant readiness."""

from types import SimpleNamespace

import pytest

from core.services.evidence_readiness import assess


def row(path, text):
    return SimpleNamespace(path=path, excerpt=text)


@pytest.mark.parametrize("symbol,value", [("CEILING", 29), ("RETRY_CAP", -11), ("BOUND", 73)])
def test_import_reference_requires_definition(symbol, value):
    question = "What numeric threshold does admit compare to?"
    source = row(
        "admit.py", f"from limits import {symbol}\ndef admit(n):\n    return n < {symbol}\n"
    )
    missing = assess(question, [source])
    assert not missing.ready and missing.missing == (symbol,)
    definition = row("limits.py", f"{symbol} = {value}\n")
    assert assess(question, [source, definition]).ready


@pytest.mark.parametrize(
    "question",
    [
        "Does guard use Kafka?",
        "Describe guard and how storage is configured",
        "What numeric threshold does guard return?",
    ],
)
def test_partial_and_relationship_requirements_not_certified(question):
    assert not assess(question, [row("guard.py", "def guard():\n    return 'local'\n")]).ready


def test_direct_literal_needs_no_extra_definition():
    assert assess(
        "What literal string does mode return?",
        [row("mode.py", "def mode():\n    return 'local'\n")],
    ).ready


def test_arbitrary_runtime_call_is_not_static_proof():
    assert not assess(
        "What value does fetch return?", [row("f.py", "def fetch():\n    return client.get()\n")]
    ).ready


@pytest.mark.parametrize(
    "objective", ["What literal does identity return?", "How does identity compare values?"]
)
def test_parameter_return_does_not_prove_literal_or_comparison(objective):
    assert not assess(objective, [row("i.py", "def identity(value):\n    return value\n")]).ready


@pytest.mark.asyncio
async def test_planner_receives_server_owned_missing_definition_context():
    import json

    from rag.generator.investigation_planner import Planner

    class Provider:
        async def generate(self, messages, before_request, **kwargs):
            context = json.loads(messages[1]["content"])["server_readiness"]
            assert context["ready"] is False and context["missing_definitions"] == ["MAXIMUM"]
            return SimpleNamespace(content='{"operation":"search","query":"MAXIMUM"}')

    planner = object.__new__(Planner)
    planner.provider = Provider()
    planner.set_readiness(
        assess(
            "What numeric threshold does admit compare to?",
            [row("a.py", "from limits import MAXIMUM\ndef admit(n):\n    return n <= MAXIMUM\n")],
        )
    )
    assert (
        await planner("What numeric threshold does admit compare to?", ())
    ).operation == "search"


def test_model_cannot_supply_readiness_permission():
    from core.services.investigation import Decision

    with pytest.raises(ValueError):
        Decision.model_validate({"operation": "answer", "ready": True})


@pytest.mark.asyncio
async def test_actual_503_contract_is_not_insufficient_evidence(tmp_path):
    import httpx

    from rag.generator.gemini_provider import Budget, Gemini
    from rag.generator.local_provider import GenerationError

    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(503, text="PRIVATE_PROVIDER_ERROR")

    async def reserve():
        pass

    budget = Budget(tmp_path / "503-ledger.json")
    with pytest.raises(GenerationError) as error:
        await Gemini("TEST_ONLY", budget, httpx.MockTransport(handler)).generate(
            [{"role": "system", "content": "synthetic"}, {"role": "user", "content": "synthetic"}],
            reserve,
            response_schema={"type": "object"},
        )
    assert error.value.code == "provider_status" and len(calls) == 1
    assert len(budget.entries) == 1 and budget.entries[0]["http_status"] == 503
    assert not budget.stopped
