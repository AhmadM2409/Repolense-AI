"""Offline HTTP contracts for planner-only thinking and remaining deadlines."""

import asyncio
import json

import httpx
import pytest

from rag.generator.gemini_provider import Budget, Gemini
from rag.generator.local_provider import GenerationError
from tests.unit.test_gemini_experiment import reserve, response


@pytest.mark.asyncio
@pytest.mark.parametrize("level", [None, "minimal", "low"])
async def test_thinking_is_opt_in_and_deadline_bounded(tmp_path, level):
    def handler(request):
        config = json.loads(request.content)["generationConfig"]
        if level:
            assert config["thinkingConfig"] == {"thinkingLevel": level.upper()}
        else:
            assert "thinkingConfig" not in config
        assert 0 < request.extensions["timeout"]["read"] <= 2
        return httpx.Response(200, json=response())

    provider = Gemini(
        "TEST_ONLY",
        Budget(tmp_path / "ledger.json"),
        httpx.MockTransport(handler),
        thinking_level=level,
        deadline=asyncio.get_running_loop().time() + 2,
    )
    await provider.generate(
        [{"role": "system", "content": "synthetic"}, {"role": "user", "content": "synthetic"}],
        reserve,
        response_schema={"type": "object"},
    )


@pytest.mark.asyncio
async def test_expired_deadline_prevents_reservation(tmp_path):
    budget = Budget(tmp_path / "ledger.json")

    def forbidden(request):
        raise AssertionError("POST after deadline")

    p = Gemini(
        "TEST_ONLY",
        budget,
        httpx.MockTransport(forbidden),
        deadline=asyncio.get_running_loop().time() - 1,
    )
    with pytest.raises(GenerationError):
        await p.generate(
            [{"role": "system", "content": "x"}, {"role": "user", "content": "x"}],
            reserve,
            response_schema={"type": "object"},
        )
    assert not budget.entries


@pytest.mark.asyncio
async def test_cancellation_closes_http_and_consumes_one_attempt(tmp_path):
    entered = asyncio.Event()
    closed = asyncio.Event()
    budget = Budget(tmp_path / "ledger.json")

    async def handler(request):
        entered.set()
        try:
            await asyncio.sleep(5)
        finally:
            closed.set()

    p = Gemini("TEST_ONLY", budget, httpx.MockTransport(handler))
    task = asyncio.create_task(
        p.generate(
            [{"role": "system", "content": "x"}, {"role": "user", "content": "x"}],
            reserve,
            response_schema={"type": "object"},
        )
    )
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert closed.is_set() and len(budget.entries) == 1


@pytest.mark.asyncio
async def test_planner_deduplicates_canonical_observations():
    from types import SimpleNamespace
    from uuid import uuid4

    from core.services.investigation import Observation, Step
    from rag.generator.investigation_planner import Planner

    item = Observation(
        source_id=uuid4(),
        chunk_id=uuid4(),
        commit_sha="a" * 40,
        path="a.py",
        line_start=1,
        line_end=1,
        excerpt="VALUE = 7",
    )

    class Provider:
        async def generate(self, messages, before_request, **kwargs):
            import json

            data = json.loads(messages[1]["content"])
            assert len(data["untrusted_evidence"]) == 1
            return SimpleNamespace(content='{"operation":"stop","query":""}')

    planner = object.__new__(Planner)
    planner.provider = Provider()
    await planner(
        "Find value.",
        (
            Step(number=1, operation="search", observations=[item]),
            Step(number=2, operation="search", observations=[item]),
        ),
    )
