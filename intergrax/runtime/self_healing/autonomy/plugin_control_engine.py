# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Plugin-based autonomy control engine (SELF-HEALING R6.1)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.self_healing.autonomy.approval import HumanApprovalRequirementResolver
from intergrax.contracts.self_healing.autonomy.decision import AutonomyControlDecision
from intergrax.contracts.self_healing.autonomy.ids import mint_autonomy_control_decision_id
from intergrax.contracts.self_healing.autonomy.level import AutonomyLevel
from intergrax.contracts.self_healing.autonomy.policy import AutonomyPolicy
from intergrax.contracts.self_healing.autonomy.request import AutonomyControlRequest
from intergrax.contracts.self_healing.autonomy.risk import AutonomyRiskEvaluator, AutonomyRiskBand


def _effective_level(
    policy_level: AutonomyLevel,
    required_level: AutonomyLevel,
) -> AutonomyLevel:
    """Conservative merge: lower autonomy wins (OBSERVE < RECOMMEND < APPROVAL < CONTROLLED)."""
    order = (
        AutonomyLevel.OBSERVE_ONLY,
        AutonomyLevel.RECOMMEND_ONLY,
        AutonomyLevel.APPROVAL_REQUIRED,
        AutonomyLevel.CONTROLLED_EXECUTION,
    )
    policy_index = order.index(policy_level)
    required_index = order.index(required_level)
    merged = order[min(policy_index, required_index)]
    merged.ensure_runtime_activatable()
    return merged


def _auto_path_allowed(
    effective_level: AutonomyLevel,
    risk_band: AutonomyRiskBand,
    human_approval_required: bool,
) -> bool:
    if effective_level is not AutonomyLevel.CONTROLLED_EXECUTION:
        return False
    if human_approval_required:
        return False
    return risk_band is AutonomyRiskBand.LOW


@dataclass(frozen=True, slots=True)
class PluginAutonomyControlEngine:
    policy: AutonomyPolicy
    risk_evaluator: AutonomyRiskEvaluator
    approval_resolver: HumanApprovalRequirementResolver
    _engine_id: str = "platform.plugin_autonomy_control"

    @property
    def engine_id(self) -> str:
        return self._engine_id

    def evaluate(self, request: AutonomyControlRequest) -> AutonomyControlDecision:
        policy_outcome = self.policy.evaluate(request)
        effective = _effective_level(
            policy_outcome.suggested_level,
            request.decision_context.required_autonomy_level,
        )
        risk_outcome = self.risk_evaluator.evaluate(request, effective)
        human_approval = self.approval_resolver.resolve(
            request,
            policy_outcome,
            risk_outcome,
            effective,
        )
        auto_allowed = _auto_path_allowed(
            effective,
            risk_outcome.risk_band,
            human_approval.required,
        )
        correlation_id = request.decision_context.recommendation_correlation_id
        return AutonomyControlDecision(
            decision_id=mint_autonomy_control_decision_id(),
            autonomy_level=effective,
            auto_path_allowed=auto_allowed,
            constraints=policy_outcome.constraint_descriptors,
            policy_outcome=policy_outcome,
            risk_outcome=risk_outcome,
            human_approval=human_approval,
            audit_bundle=request.decision_context.audit,
            engine_id=self.engine_id,
            recommendation_correlation_id=correlation_id,
        )


__all__ = ["PluginAutonomyControlEngine"]
