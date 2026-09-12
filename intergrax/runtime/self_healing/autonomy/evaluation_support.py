# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Shared helpers for autonomy decision evaluation (SELF-HEALING R6.2)."""

from __future__ import annotations

from intergrax.contracts.self_healing.autonomy.approval_evaluation import HumanApprovalEvaluationResult
from intergrax.contracts.self_healing.autonomy.evaluation_confidence import (
    AutonomyEvaluationConfidence,
    AutonomyEvaluationConfidenceLevel,
)
from intergrax.contracts.self_healing.autonomy.evaluation_result import AutonomyEvaluationVerdict
from intergrax.contracts.self_healing.autonomy.explanation import (
    AutonomyDecisionExplanation,
    AutonomyExplanationBullet,
    AutonomyRuleInfluence,
)
from intergrax.contracts.self_healing.autonomy.level import AutonomyLevel
from intergrax.contracts.self_healing.autonomy.policy_evaluation import (
    AutonomyPolicyEvaluationResult,
    AutonomyPolicyEvaluationVerdict,
)
from intergrax.contracts.self_healing.autonomy.request import AutonomyControlRequest
from intergrax.contracts.self_healing.autonomy.risk import AutonomyRiskBand, AutonomyRiskEvaluationResult
from intergrax.contracts.self_healing.strategy_recommendation.confidence import (
    StrategyRecommendationConfidenceLevel,
)

_LEVEL_ORDER: tuple[AutonomyLevel, ...] = (
    AutonomyLevel.OBSERVE_ONLY,
    AutonomyLevel.RECOMMEND_ONLY,
    AutonomyLevel.APPROVAL_REQUIRED,
    AutonomyLevel.CONTROLLED_EXECUTION,
)


def merge_effective_autonomy_level(
    policy_level: AutonomyLevel,
    required_level: AutonomyLevel,
) -> AutonomyLevel:
    policy_index = _LEVEL_ORDER.index(policy_level)
    required_index = _LEVEL_ORDER.index(required_level)
    merged = _LEVEL_ORDER[min(policy_index, required_index)]
    merged.ensure_runtime_activatable()
    return merged


def confidence_from_recommendation(request: AutonomyControlRequest) -> AutonomyEvaluationConfidence:
    level = request.recommendation.confidence
    match level:
        case StrategyRecommendationConfidenceLevel.HIGH:
            return AutonomyEvaluationConfidence(
                level=AutonomyEvaluationConfidenceLevel.HIGH,
                score=0.9,
            )
        case StrategyRecommendationConfidenceLevel.MEDIUM:
            return AutonomyEvaluationConfidence(
                level=AutonomyEvaluationConfidenceLevel.MEDIUM,
                score=0.7,
            )
        case StrategyRecommendationConfidenceLevel.LOW:
            return AutonomyEvaluationConfidence(
                level=AutonomyEvaluationConfidenceLevel.LOW,
                score=0.5,
            )
        case StrategyRecommendationConfidenceLevel.INSUFFICIENT_DATA:
            return AutonomyEvaluationConfidence(
                level=AutonomyEvaluationConfidenceLevel.UNKNOWN,
                score=0.3,
            )


def adjust_confidence_for_risk(
    base: AutonomyEvaluationConfidence,
    risk_band: AutonomyRiskBand,
) -> AutonomyEvaluationConfidence:
    penalty = {
        AutonomyRiskBand.LOW: 0.0,
        AutonomyRiskBand.MEDIUM: 0.1,
        AutonomyRiskBand.HIGH: 0.25,
        AutonomyRiskBand.CRITICAL: 0.4,
        AutonomyRiskBand.UNKNOWN: 0.15,
    }[risk_band]
    score = max(0.0, base.score - penalty)
    if score >= 0.8:
        conf_level = AutonomyEvaluationConfidenceLevel.HIGH
    elif score >= 0.55:
        conf_level = AutonomyEvaluationConfidenceLevel.MEDIUM
    elif score >= 0.35:
        conf_level = AutonomyEvaluationConfidenceLevel.LOW
    else:
        conf_level = AutonomyEvaluationConfidenceLevel.UNKNOWN
    return AutonomyEvaluationConfidence(level=conf_level, score=score)


def resolve_evaluation_verdict(
    policy_result: AutonomyPolicyEvaluationResult,
    risk_result: AutonomyRiskEvaluationResult,
    approval_result: HumanApprovalEvaluationResult,
) -> AutonomyEvaluationVerdict:
    if policy_result.verdict is AutonomyPolicyEvaluationVerdict.FAIL:
        return AutonomyEvaluationVerdict.DENIED
    if risk_result.risk_band in (AutonomyRiskBand.HIGH, AutonomyRiskBand.CRITICAL):
        return AutonomyEvaluationVerdict.DENIED
    if approval_result.approval_required:
        return AutonomyEvaluationVerdict.CONDITIONAL
    return AutonomyEvaluationVerdict.CLEARED


def build_evaluation_explanation(
    policy_result: AutonomyPolicyEvaluationResult,
    risk_result: AutonomyRiskEvaluationResult,
    approval_result: HumanApprovalEvaluationResult,
    effective_level: AutonomyLevel,
) -> AutonomyDecisionExplanation:
    because: list[AutonomyExplanationBullet] = []
    if policy_result.verdict is AutonomyPolicyEvaluationVerdict.PASS:
        because.append(
            AutonomyExplanationBullet(
                code="policy.permits_posture",
                message="Enterprise policy permits the evaluated autonomy posture.",
            ),
        )
    else:
        because.append(
            AutonomyExplanationBullet(
                code="policy.blocks_required_level",
                message="Policy suggested level is below the required autonomy level.",
            ),
        )
    if risk_result.risk_band in (AutonomyRiskBand.LOW, AutonomyRiskBand.MEDIUM):
        because.append(
            AutonomyExplanationBullet(
                code="risk.acceptable_band",
                message=f"Risk band {risk_result.risk_band.value} is within evaluation tolerance.",
            ),
        )
    elif risk_result.risk_band is AutonomyRiskBand.UNKNOWN:
        because.append(
            AutonomyExplanationBullet(
                code="risk.unknown_band",
                message="Risk band is unknown; evaluation remains conservative.",
            ),
        )
    else:
        because.append(
            AutonomyExplanationBullet(
                code="risk.elevated_band",
                message=f"Risk band {risk_result.risk_band.value} exceeds automatic clearance.",
            ),
        )
    if approval_result.approval_required:
        because.append(
            AutonomyExplanationBullet(
                code="approval.required",
                message=approval_result.rationale,
            ),
        )
    else:
        because.append(
            AutonomyExplanationBullet(
                code="approval.not_required",
                message="Human approval gate not required for this evaluation.",
            ),
        )
    rules = (
        AutonomyRuleInfluence(
            rule_id=policy_result.policy_id,
            influence_summary=policy_result.rationale,
        ),
        AutonomyRuleInfluence(
            rule_id=risk_result.evaluator_id,
            influence_summary=risk_result.explanation,
        ),
    )
    summary = (
        f"Autonomy evaluation at {effective_level.value} with policy {policy_result.verdict.value} "
        f"and risk {risk_result.risk_band.value}."
    )
    return AutonomyDecisionExplanation(
        summary=summary,
        because=tuple(because),
        rules_influenced=rules,
        constraints_active=policy_result.constraint_descriptors,
    )


def collect_evaluation_reasons(
    policy_result: AutonomyPolicyEvaluationResult,
    risk_result: AutonomyRiskEvaluationResult,
    approval_result: HumanApprovalEvaluationResult,
) -> tuple[str, ...]:
    reasons = (
        f"policy:{policy_result.verdict.value}",
        f"risk:{risk_result.risk_band.value}",
        f"approval:{'required' if approval_result.approval_required else 'not_required'}",
    )
    return reasons


__all__ = [
    "adjust_confidence_for_risk",
    "build_evaluation_explanation",
    "collect_evaluation_reasons",
    "confidence_from_recommendation",
    "merge_effective_autonomy_level",
    "resolve_evaluation_verdict",
]
