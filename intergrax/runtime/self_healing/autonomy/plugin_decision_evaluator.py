# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Plugin-based autonomy decision evaluator (SELF-HEALING R6.2)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.self_healing.autonomy.approval_evaluation import HumanApprovalEvaluator
from intergrax.contracts.self_healing.autonomy.evaluation_result import (
    AUTONOMY_EVALUATION_CONTRACT_VERSION,
    AutonomyEvaluationResult,
    AutonomyEvaluationSourceKind,
    AutonomyEvaluationSourceRef,
)
from intergrax.contracts.self_healing.autonomy.ids import mint_autonomy_evaluation_id
from intergrax.contracts.self_healing.autonomy.policy_evaluation import AutonomyPolicyEvaluator
from intergrax.contracts.self_healing.autonomy.request import AutonomyControlRequest
from intergrax.contracts.self_healing.autonomy.risk import (
    AutonomyRiskEvaluator,
    autonomy_risk_evaluation_result_from_assessment,
)
from intergrax.runtime.self_healing.autonomy.evaluation_support import (
    adjust_confidence_for_risk,
    build_evaluation_explanation,
    collect_evaluation_reasons,
    confidence_from_recommendation,
    merge_effective_autonomy_level,
    resolve_evaluation_verdict,
)


@dataclass(frozen=True, slots=True)
class PluginAutonomyDecisionEvaluator:
    policy_evaluator: AutonomyPolicyEvaluator
    risk_evaluator: AutonomyRiskEvaluator
    approval_evaluator: HumanApprovalEvaluator
    _evaluator_id: str = "platform.plugin_autonomy_decision_evaluation"

    @property
    def evaluator_id(self) -> str:
        return self._evaluator_id

    def evaluate(self, request: AutonomyControlRequest) -> AutonomyEvaluationResult:
        policy_result = self.policy_evaluator.evaluate(request)
        effective_level = merge_effective_autonomy_level(
            policy_result.suggested_level,
            request.decision_context.required_autonomy_level,
        )
        risk_assessment = self.risk_evaluator.evaluate(request, effective_level)
        base_confidence = confidence_from_recommendation(request)
        risk_confidence = adjust_confidence_for_risk(base_confidence, risk_assessment.risk_band)
        risk_result = autonomy_risk_evaluation_result_from_assessment(
            risk_assessment,
            risk_confidence,
        )
        approval_result = self.approval_evaluator.evaluate(
            request,
            policy_result,
            risk_result,
            effective_level,
        )
        verdict = resolve_evaluation_verdict(policy_result, risk_result, approval_result)
        explanation = build_evaluation_explanation(
            policy_result,
            risk_result,
            approval_result,
            effective_level,
        )
        correlation_id = request.decision_context.recommendation_correlation_id
        sources = (
            AutonomyEvaluationSourceRef(
                kind=AutonomyEvaluationSourceKind.POLICY,
                source_id=policy_result.policy_id,
            ),
            AutonomyEvaluationSourceRef(
                kind=AutonomyEvaluationSourceKind.RISK,
                source_id=risk_result.evaluator_id,
            ),
            AutonomyEvaluationSourceRef(
                kind=AutonomyEvaluationSourceKind.APPROVAL,
                source_id=approval_result.evaluator_id,
            ),
            AutonomyEvaluationSourceRef(
                kind=AutonomyEvaluationSourceKind.RECOMMENDATION,
                source_id=request.recommendation.engine_id,
            ),
            AutonomyEvaluationSourceRef(
                kind=AutonomyEvaluationSourceKind.EVALUATOR,
                source_id=self.evaluator_id,
            ),
        )
        return AutonomyEvaluationResult(
            evaluation_id=mint_autonomy_evaluation_id(),
            verdict=verdict,
            autonomy_level=effective_level,
            confidence=risk_confidence,
            reasons=collect_evaluation_reasons(policy_result, risk_result, approval_result),
            sources=sources,
            explanation=explanation,
            policy_result=policy_result,
            risk_result=risk_result,
            approval_result=approval_result,
            audit_bundle=request.decision_context.audit,
            contract_version=AUTONOMY_EVALUATION_CONTRACT_VERSION,
            recommendation_correlation_id=correlation_id,
            evaluator_id=self.evaluator_id,
        )


__all__ = ["PluginAutonomyDecisionEvaluator"]
