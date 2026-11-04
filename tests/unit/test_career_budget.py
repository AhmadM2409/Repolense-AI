"""Per-assessment (12 = 10 matching + 2 roadmap), owner (40) and aggregate (100) arithmetic."""

from types import SimpleNamespace

import pytest

from core.services import career_assessment as budget


def calls(matching=0, roadmap=0):
    return [{"phase": "match/R1"}] * matching + [{"phase": "roadmap/generate"}] * roadmap


def test_constants_preserve_owner_and_aggregate_caps():
    assert (budget.MAX_ASSESSMENT_CALLS, budget.MAX_MATCHING_CALLS, budget.MAX_ROADMAP_CALLS) == (
        12,
        10,
        2,
    )
    assert (budget.MAX_OWNER_CALLS, budget.MAX_AGGREGATE_CALLS) == (40, 100)
    assert budget.MAX_CALLS_PER_MINUTE == 10


@pytest.mark.parametrize(
    ("matching", "roadmap", "match_left", "roadmap_left"),
    [
        (0, 0, 10, 2),
        (6, 0, 4, 2),  # Saved demo shape before its roadmap.
        (10, 0, 0, 2),  # Four evidenced requirements still leave the roadmap's two.
        (10, 1, 0, 1),
        (10, 2, 0, 0),  # Twelve: nothing more of any kind.
        (8, 2, 2, 0),
    ],
)
def test_remaining_by_phase(matching, roadmap, match_left, roadmap_left):
    history = calls(matching, roadmap)
    assert budget.remaining(history, "check/R1") == match_left
    assert budget.remaining(history, "roadmap/check") == roadmap_left
    assert budget.remaining(history, "extract") == match_left


def test_matching_can_never_consume_roadmap_slots_or_exceed_twelve():
    assert budget.remaining(calls(10), "match/R5") == 0
    assert budget.remaining(calls(12), "roadmap/generate") == 0
    assert budget.remaining(calls(13), "roadmap/generate") == 0  # Never negative.


@pytest.mark.parametrize(
    ("spent", "owner", "left"),
    [(0, 0, 40), (22, 22, 18), (99, 10, 1), (100, 10, 0), (50, 39, 1), (50, 40, 0), (60, 45, 0)],
)
def test_shared_remaining(spent, owner, left):
    row = SimpleNamespace(spent=spent, owners={"u": owner})
    assert budget.budget_remaining(row, "u") == left
