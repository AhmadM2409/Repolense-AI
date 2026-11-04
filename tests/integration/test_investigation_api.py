"""Real PostgreSQL authenticated API; hosted provider doubled only."""

import pytest

from core.config import settings
from core.services.investigation import Decision
from tests.integration import test_grounding_prototype as prior
from tests.integration import test_semantic_retrieval as base

pytest_plugins = ["tests.integration.test_gemini_production"]
pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


async def test_route_consent_scope_and_completion(client, database, actors, gemini, monkeypatch):
    from rag.generator.investigation_planner import Planner

    monkeypatch.setattr(settings, "investigation_enabled", True)
    # Only planner inference is replaced; production scope/retrieval/Q&A run normally.
    monkeypatch.setattr(Planner, "__init__", lambda self: None)

    async def decide(self, objective, steps):
        return (
            Decision(operation="search", query="guard")
            if not steps
            else Decision(operation="answer")
        )

    monkeypatch.setattr(Planner, "__call__", decide)
    source = await prior.setup(database, actors)
    endpoint = base.endpoint(source).replace("/search", "/investigations")
    body = dict(
        objective="Describe guard.",
        generation_provider="gemini",
        hosted_processing_consent="gemini-source-processing-v1",
    )
    assert (await client.post(endpoint, json=body)).status_code == 401
    assert (await client.post(endpoint, json=body, headers=base.auth(actors, 1))).status_code == 404
    assert (
        await client.post(
            endpoint, json={**body, "hosted_processing_consent": None}, headers=base.auth(actors)
        )
    ).status_code == 422
    assert not gemini.calls
    # The injected planner has no provider property, so existing doubled Q&A is used.
    response = await client.post(endpoint, json=body, headers=base.auth(actors))
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "completed"
    assert len(response.json()["steps"]) == 2 and response.json()["final"]["citations"]
