# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Preventive action proposal lifecycle (PREVENTIVE R7)."""

from __future__ import annotations

from enum import StrEnum


class PreventiveActionLifecycleState(StrEnum):
    PROPOSED = "PROPOSED"
    WAITING_APPROVAL = "WAITING_APPROVAL"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    EXECUTED = "EXECUTED"
    FAILED = "FAILED"


_ALLOWED_TRANSITIONS: dict[PreventiveActionLifecycleState, frozenset[PreventiveActionLifecycleState]] = {
    PreventiveActionLifecycleState.PROPOSED: frozenset(
        {
            PreventiveActionLifecycleState.WAITING_APPROVAL,
            PreventiveActionLifecycleState.APPROVED,
            PreventiveActionLifecycleState.REJECTED,
        },
    ),
    PreventiveActionLifecycleState.WAITING_APPROVAL: frozenset(
        {
            PreventiveActionLifecycleState.APPROVED,
            PreventiveActionLifecycleState.REJECTED,
        },
    ),
    PreventiveActionLifecycleState.APPROVED: frozenset(
        {
            PreventiveActionLifecycleState.EXECUTED,
            PreventiveActionLifecycleState.FAILED,
        },
    ),
    PreventiveActionLifecycleState.REJECTED: frozenset(),
    PreventiveActionLifecycleState.EXECUTED: frozenset(),
    PreventiveActionLifecycleState.FAILED: frozenset(),
}


def assert_preventive_action_lifecycle_transition(
    current: PreventiveActionLifecycleState,
    target: PreventiveActionLifecycleState,
) -> None:
    allowed = _ALLOWED_TRANSITIONS.get(current, frozenset())
    if target not in allowed:
        raise ValueError(
            f"illegal lifecycle transition {current.value} -> {target.value}",
        )


__all__ = [
    "PreventiveActionLifecycleState",
    "assert_preventive_action_lifecycle_transition",
]
