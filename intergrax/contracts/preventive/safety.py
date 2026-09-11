# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Prevention safety boundary — execution is never permitted (PREVENTIVE R6-Q)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PreventiveSafetyAssessment:
    """
    Formal safety envelope for a recommendation.

    Invariant: ``execution_allowed`` is always false — enforced by validator, not convention.
    """

    execution_allowed: bool
    requires_human_review: bool
    risk_level: str
    governance_status: str

    def __post_init__(self) -> None:
        if self.execution_allowed:
            raise ValueError(
                "execution_allowed must remain false — preventive layer cannot hold remediation authority",
            )
        if not self.risk_level.strip():
            raise ValueError("risk_level must be non-empty")
        if not self.governance_status.strip():
            raise ValueError("governance_status must be non-empty")


def preventive_safety_always_disabled(
    *,
    requires_human_review: bool,
    risk_level: str,
    governance_status: str = "PENDING",
) -> PreventiveSafetyAssessment:
    """Mint a governed safety assessment — execution remains impossible."""
    return PreventiveSafetyAssessment(
        execution_allowed=False,
        requires_human_review=requires_human_review,
        risk_level=risk_level,
        governance_status=governance_status,
    )


__all__ = ["PreventiveSafetyAssessment", "preventive_safety_always_disabled"]
