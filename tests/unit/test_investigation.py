"""Contract checks for the inactive investigation prototype; no inference."""

import pytest
from pydantic import ValidationError

from core.services.investigation import Decision, InvestigationRequest


@pytest.mark.parametrize("operation", ["shell", "fetch_url", "write", "python"])
def test_unavailable_operations_rejected(operation):
    with pytest.raises(ValidationError):
        Decision(operation=operation)


@pytest.mark.parametrize("steps", [0, 5, True])
def test_step_limit_cannot_be_expanded(steps):
    with pytest.raises(ValidationError):
        InvestigationRequest(objective="Explain auth", max_steps=steps)


def test_scope_cannot_be_added_to_model_decision():
    with pytest.raises(ValidationError):
        Decision(operation="search", query="auth", user_id="foreign")


def test_objective_has_bounded_size():
    with pytest.raises(ValidationError):
        InvestigationRequest(objective="x" * 1025)
