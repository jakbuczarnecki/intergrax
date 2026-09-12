# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Domain-neutral execution lifecycle decisions after ERL resolution and compensation (ERL)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class RecoveryLifecycleAction(StrEnum):
    """Recommended execution lifecycle posture — no business-domain semantics."""

    CONTINUE = "continue"
    PAUSE = "pause"
    ESCALATE = "escalate"
    TERMINATE = "terminate"
    WAIT = "wait"


@dataclass(frozen=True, slots=True)
class RecoveryDecision:
    """Typed recovery outcome — execution runtime applies lifecycle changes in later phases."""

    action: RecoveryLifecycleAction
    rationale: str = ""


def missing_recovery_strategy_decision() -> RecoveryDecision:
    """Fail closed when no recovery plugin is registered for the requested id."""
    return RecoveryDecision(
        action=RecoveryLifecycleAction.ESCALATE,
        rationale="recovery_strategy_missing",
    )


def abstained_recovery_strategy_decision() -> RecoveryDecision:
    """Strategy registered but returned no decision — lifecycle remains unresolved."""
    return RecoveryDecision(
        action=RecoveryLifecycleAction.WAIT,
        rationale="recovery_strategy_abstained",
    )


__all__ = [
    "RecoveryDecision",
    "RecoveryLifecycleAction",
    "abstained_recovery_strategy_decision",
    "missing_recovery_strategy_decision",
]
