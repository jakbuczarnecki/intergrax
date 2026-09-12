# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Autonomy control decision — classification only, not execution (SELF-HEALING R6.1)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.self_healing.autonomy.approval import HumanApprovalRequirement
from intergrax.contracts.self_healing.autonomy.audit import AutonomyAuditBundle
from intergrax.contracts.self_healing.autonomy.level import AutonomyLevel
from intergrax.contracts.self_healing.autonomy.policy import AutonomyConstraintDescriptor, AutonomyPolicyOutcome
from intergrax.contracts.self_healing.autonomy.risk import AutonomyRiskAssessment


@dataclass(frozen=True, slots=True)
class AutonomyControlDecision:
    """
    Effective autonomy posture after plugin evaluation.

    ``auto_path_allowed`` is advisory; execution still requires decision authority and guard.
    """

    decision_id: str
    autonomy_level: AutonomyLevel
    auto_path_allowed: bool
    constraints: tuple[AutonomyConstraintDescriptor, ...]
    policy_outcome: AutonomyPolicyOutcome
    risk_outcome: AutonomyRiskAssessment
    human_approval: HumanApprovalRequirement
    audit_bundle: AutonomyAuditBundle
    engine_id: str
    recommendation_correlation_id: str

    def __post_init__(self) -> None:
        if not self.decision_id.strip():
            raise ValueError("decision_id required")
        if not self.engine_id.strip():
            raise ValueError("engine_id required")
        if not self.recommendation_correlation_id.strip():
            raise ValueError("recommendation_correlation_id required")
        self.autonomy_level.ensure_runtime_activatable()
        if self.audit_bundle.recommendation_correlation_id != self.recommendation_correlation_id:
            raise ValueError("recommendation_correlation_id mismatch with audit_bundle")
        if self.auto_path_allowed and self.autonomy_level is not AutonomyLevel.CONTROLLED_EXECUTION:
            raise ValueError("auto_path_allowed requires CONTROLLED_EXECUTION level")
        if self.auto_path_allowed and self.human_approval.required:
            raise ValueError("auto_path_allowed incompatible with required human approval")


__all__ = ["AutonomyControlDecision"]
