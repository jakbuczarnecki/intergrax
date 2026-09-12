# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Domain-neutral platform compensation decisions after resolution (ERL)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class CompensationPlatformIntent(StrEnum):
    """What the platform should do next — no business-domain semantics."""

    COMPENSATION_REQUIRED = "compensation_required"
    APPROVED = "approved"
    UNAVAILABLE = "unavailable"
    DEFERRED = "deferred"
    ESCALATE = "escalate"


@dataclass(frozen=True, slots=True)
class CompensationDecision:
    """Typed strategy outcome — execution phases consume this in later ERL work."""

    intent: CompensationPlatformIntent
    compensation_operation_ref: str | None = None
    rationale: str = ""

    def __post_init__(self) -> None:
        if self.intent in (
            CompensationPlatformIntent.COMPENSATION_REQUIRED,
            CompensationPlatformIntent.APPROVED,
        ):
            ref = self.compensation_operation_ref
            if ref is None or not ref.strip():
                raise ValueError(
                    "compensation_operation_ref required when intent approves compensation",
                )


def missing_compensation_strategy_decision() -> CompensationDecision:
    """Fail closed when no compensation plugin is registered for the requested id."""
    return CompensationDecision(
        intent=CompensationPlatformIntent.ESCALATE,
        rationale="compensation_strategy_missing",
    )


def abstained_compensation_strategy_decision() -> CompensationDecision:
    """Strategy registered but returned no decision — platform does not guess."""
    return CompensationDecision(
        intent=CompensationPlatformIntent.UNAVAILABLE,
        rationale="compensation_strategy_abstained",
    )


__all__ = [
    "CompensationDecision",
    "CompensationPlatformIntent",
    "abstained_compensation_strategy_decision",
    "missing_compensation_strategy_decision",
]
