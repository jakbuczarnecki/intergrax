# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Recommendation lifecycle — no GENERATED → EXECUTED path (PREVENTIVE R6-Q)."""

from __future__ import annotations

from enum import StrEnum


class PreventiveRecommendationLifecycleState(StrEnum):
    GENERATED = "GENERATED"
    VALIDATED = "VALIDATED"
    PRESENTED = "PRESENTED"
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    EVALUATED = "EVALUATED"


_ALLOWED: dict[
    PreventiveRecommendationLifecycleState,
    frozenset[PreventiveRecommendationLifecycleState],
] = {
    PreventiveRecommendationLifecycleState.GENERATED: frozenset(
        {PreventiveRecommendationLifecycleState.VALIDATED},
    ),
    PreventiveRecommendationLifecycleState.VALIDATED: frozenset(
        {PreventiveRecommendationLifecycleState.PRESENTED},
    ),
    PreventiveRecommendationLifecycleState.PRESENTED: frozenset(
        {
            PreventiveRecommendationLifecycleState.ACCEPTED,
            PreventiveRecommendationLifecycleState.REJECTED,
        },
    ),
    PreventiveRecommendationLifecycleState.ACCEPTED: frozenset(
        {PreventiveRecommendationLifecycleState.EVALUATED},
    ),
    PreventiveRecommendationLifecycleState.REJECTED: frozenset(
        {PreventiveRecommendationLifecycleState.EVALUATED},
    ),
    PreventiveRecommendationLifecycleState.EVALUATED: frozenset(),
}


class PreventiveLifecycleViolation(ValueError):
    """Illegal lifecycle transition — governance breach."""


def assert_lifecycle_transition(
    current: PreventiveRecommendationLifecycleState,
    target: PreventiveRecommendationLifecycleState,
) -> None:
    """Enforce monotonic, human-governed lifecycle (never skip to execution)."""
    if current == target:
        return
    allowed = _ALLOWED.get(current, frozenset())
    if target not in allowed:
        raise PreventiveLifecycleViolation(f"illegal lifecycle transition: {current} -> {target}")


__all__ = [
    "PreventiveLifecycleViolation",
    "PreventiveRecommendationLifecycleState",
    "assert_lifecycle_transition",
]
