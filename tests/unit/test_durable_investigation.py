"""Durable checkpoint bounds and strict version contract."""

import pytest

from core.services.durable_investigation import validate_state


@pytest.mark.parametrize(
    "value",
    [
        {"version": 2},
        {"version": True},
        {"attempts": [{}] * 7},
        {"events": {str(i): {} for i in range(11)}},
        {"events": {"plan/0": "x" * 131073}},
        {"owner_id": "foreign"},
    ],
)
def test_reject_unbounded_or_unknown_state(value):
    with pytest.raises(ValueError):
        validate_state({"version": 1, **value})


def test_missing_state_version_is_not_silently_upgraded():
    with pytest.raises(ValueError, match="Unsupported checkpoint version"):
        validate_state({})
