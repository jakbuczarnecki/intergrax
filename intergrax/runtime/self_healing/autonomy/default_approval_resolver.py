# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Default human approval resolver — policy and risk driven (SELF-HEALING R6.1)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.self_healing.autonomy.approval import HumanApprovalRequirement
from intergrax.contracts.self_healing.autonomy.level import AutonomyLevel
from intergrax.contracts.self_healing.autonomy.policy import AutonomyPolicyOutcome
from intergrax.contracts.self_healing.autonomy.request import AutonomyControlRequest
from intergrax.contracts.self_healing.autonomy.risk import AutonomyRiskAssessment, AutonomyRiskBand


@dataclass(frozen=True, slots=True)
class DefaultHumanApprovalRequirementResolver:
    _resolver_id: str = "platform.default_human_approval"

    @property
    def resolver_id(self) -> str:
        return self._resolver_id

    def resolve(
        self,
        request: AutonomyControlRequest,
        policy_outcome: AutonomyPolicyOutcome,
        risk_assessment: AutonomyRiskAssessment,
        effective_level: AutonomyLevel,
    ) -> HumanApprovalRequirement:
        _ = request
        _ = policy_outcome
        if effective_level is AutonomyLevel.APPROVAL_REQUIRED:
            return HumanApprovalRequirement(
                required=True,
                reason_code="autonomy.level.approval_required",
                rationale="Effective autonomy level mandates human approval before action.",
            )
        if risk_assessment.risk_band in (AutonomyRiskBand.HIGH, AutonomyRiskBand.CRITICAL):
            return HumanApprovalRequirement(
                required=True,
                reason_code="autonomy.risk.elevated",
                rationale=f"Risk band {risk_assessment.risk_band.value} requires human approval.",
                escalation_hint="route_to_operator",
            )
        return HumanApprovalRequirement(
            required=False,
            reason_code="autonomy.approval.not_required",
            rationale="Policy and risk posture do not mandate human approval for classification.",
        )


__all__ = ["DefaultHumanApprovalRequirementResolver"]
