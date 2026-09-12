# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Adapter from R6.1 approval resolver to R6.2 approval evaluation result."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.self_healing.autonomy.approval import HumanApprovalRequirementResolver
from intergrax.contracts.self_healing.autonomy.approval_evaluation import HumanApprovalEvaluationResult
from intergrax.contracts.self_healing.autonomy.level import AutonomyLevel
from intergrax.contracts.self_healing.autonomy.policy import AutonomyPolicyOutcome
from intergrax.contracts.self_healing.autonomy.policy_evaluation import AutonomyPolicyEvaluationResult
from intergrax.contracts.self_healing.autonomy.request import AutonomyControlRequest
from intergrax.contracts.self_healing.autonomy.risk import AutonomyRiskEvaluationResult


@dataclass(frozen=True, slots=True)
class HumanApprovalPluginEvaluator:
    resolver: HumanApprovalRequirementResolver
    _evaluator_id: str = "platform.approval_plugin_evaluator"

    @property
    def evaluator_id(self) -> str:
        return self._evaluator_id

    def evaluate(
        self,
        request: AutonomyControlRequest,
        policy_result: AutonomyPolicyEvaluationResult,
        risk_result: AutonomyRiskEvaluationResult,
        effective_level: AutonomyLevel,
    ) -> HumanApprovalEvaluationResult:
        policy_outcome = AutonomyPolicyOutcome(
            policy_id=policy_result.policy_id,
            policy_version=policy_result.policy_version,
            suggested_level=policy_result.suggested_level,
            constraint_descriptors=policy_result.constraint_descriptors,
            rationale=policy_result.rationale,
        )
        requirement = self.resolver.resolve(
            request,
            policy_outcome,
            risk_result.assessment,
            effective_level,
        )
        return HumanApprovalEvaluationResult.from_requirement(self.evaluator_id, requirement)


__all__ = ["HumanApprovalPluginEvaluator"]
