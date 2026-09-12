# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Self-healing safety evaluation before governance (SELF-HEALING R1)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from intergrax.contracts.self_healing.context import SelfHealingContext
from intergrax.contracts.self_healing.decision import SelfHealingDecision


class SelfHealingRiskLevel(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


@dataclass(frozen=True, slots=True)
class SelfHealingSafetyVerdict:
    may_proceed_to_governance: bool
    risk_level: SelfHealingRiskLevel
    blast_radius: str
    rollback_available: bool
    requires_approval: bool
    reason: str


class SelfHealingSafetyEvaluator:
    """Strategies propose — safety constrains what may reach governance."""

    def evaluate(
        self,
        decision: SelfHealingDecision,
        context: SelfHealingContext,
    ) -> SelfHealingSafetyVerdict:
        if decision.confidence >= 0.99 and decision.required_approval is False:
            return SelfHealingSafetyVerdict(
                may_proceed_to_governance=False,
                risk_level=SelfHealingRiskLevel.HIGH,
                blast_radius=context.constraints.max_blast_radius,
                rollback_available=False,
                requires_approval=True,
                reason="confidence is not authorization — approval required",
            )
        risk = SelfHealingRiskLevel.MEDIUM
        if context.constraints.production_target:
            risk = SelfHealingRiskLevel.HIGH
        requires_approval = decision.required_approval or context.constraints.production_target
        return SelfHealingSafetyVerdict(
            may_proceed_to_governance=True,
            risk_level=risk,
            blast_radius=context.constraints.max_blast_radius,
            rollback_available=True,
            requires_approval=requires_approval,
            reason="safety pre-check passed",
        )


__all__ = [
    "SelfHealingRiskLevel",
    "SelfHealingSafetyEvaluator",
    "SelfHealingSafetyVerdict",
]
