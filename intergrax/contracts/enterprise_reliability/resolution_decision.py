# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Domain-neutral platform resolution decisions after reconciliation evidence (ERL)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ResolutionPlatformAction(StrEnum):
    """What the platform should do next — no business-domain semantics."""

    CONTINUE = "continue"
    STOP = "stop"
    ESCALATE = "escalate"
    COMPENSATION_REQUIRED = "compensation_required"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class ResolutionDecision:
    """Typed strategy outcome — execution phases consume this in later ERL work."""

    action: ResolutionPlatformAction
    rationale: str = ""

    def __post_init__(self) -> None:
        if self.action is ResolutionPlatformAction.CONTINUE and not self.rationale.strip():
            # CONTINUE must be explicit; empty rationale still allowed at contract level.
            pass


def missing_resolution_strategy_decision() -> ResolutionDecision:
    """Fail closed when no resolution plugin is registered for the requested id."""
    return ResolutionDecision(
        action=ResolutionPlatformAction.ESCALATE,
        rationale="resolution_strategy_missing",
    )


def abstained_resolution_decision() -> ResolutionDecision:
    """Strategy registered but returned no decision — platform does not guess."""
    return ResolutionDecision(
        action=ResolutionPlatformAction.UNKNOWN,
        rationale="resolution_strategy_abstained",
    )


__all__ = [
    "ResolutionDecision",
    "ResolutionPlatformAction",
    "abstained_resolution_decision",
    "missing_resolution_strategy_decision",
]
