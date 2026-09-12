# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Human approval requirement contracts — no workflow or UI (SELF-HEALING R6.1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from intergrax.contracts.self_healing.autonomy.level import AutonomyLevel
from intergrax.contracts.self_healing.autonomy.policy import AutonomyPolicyOutcome
from intergrax.contracts.self_healing.autonomy.request import AutonomyControlRequest
from intergrax.contracts.self_healing.autonomy.risk import AutonomyRiskAssessment


@dataclass(frozen=True, slots=True)
class HumanApprovalRequirement:
    required: bool
    reason_code: str
    rationale: str
    escalation_hint: str | None = None

    def __post_init__(self) -> None:
        if not self.reason_code.strip():
            raise ValueError("reason_code required")
        if not self.rationale.strip():
            raise ValueError("rationale required")


@runtime_checkable
class HumanApprovalRequirementResolver(Protocol):
    @property
    def resolver_id(self) -> str: ...

    def resolve(
        self,
        request: AutonomyControlRequest,
        policy_outcome: AutonomyPolicyOutcome,
        risk_assessment: AutonomyRiskAssessment,
        effective_level: AutonomyLevel,
    ) -> HumanApprovalRequirement:
        """Determine whether human approval is required before decision authority may act."""
        ...


__all__ = [
    "HumanApprovalRequirement",
    "HumanApprovalRequirementResolver",
]
