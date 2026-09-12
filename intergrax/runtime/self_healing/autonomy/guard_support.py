# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Authorization resolution for autonomy execution guard (SELF-HEALING R6.3)."""

from __future__ import annotations

from datetime import datetime

from intergrax.contracts.self_healing.autonomy.evaluation_result import (
    AutonomyEvaluationResult,
    AutonomyEvaluationVerdict,
)
from intergrax.contracts.self_healing.autonomy.execution_authorization import (
    AutonomyExecutionAuthorization,
    AutonomyExecutionAuthorizationStatus,
)
from intergrax.contracts.self_healing.autonomy.guard import (
    AutonomyExecutionAdmissionContext,
    AutonomyGuardVerdict,
)
from intergrax.contracts.self_healing.autonomy.guard_rule import (
    AutonomyExecutionGuardRule,
    AutonomyExecutionGuardRuleVerdict,
)
from intergrax.contracts.self_healing.autonomy.ids import mint_autonomy_execution_authorization_id
from intergrax.contracts.self_healing.autonomy.level import AutonomyLevel


def _denied(
    *,
    admission: AutonomyExecutionAdmissionContext,
    guard_id: str,
    recorded_at: datetime,
    reasons: tuple[str, ...],
    evaluation: AutonomyEvaluationResult | None,
) -> AutonomyExecutionAuthorization:
    level = admission.prior_decision.autonomy_level
    evaluation_id = evaluation.evaluation_id if evaluation is not None else None
    policy_result = evaluation.policy_result if evaluation is not None else None
    return AutonomyExecutionAuthorization(
        authorization_id=mint_autonomy_execution_authorization_id(),
        status=AutonomyExecutionAuthorizationStatus.DENIED,
        decision_id=admission.decision_id,
        evaluation_id=evaluation_id,
        autonomy_level=level,
        policy_result=policy_result,
        recorded_at=recorded_at,
        reasons=reasons,
        guard_id=guard_id,
        recommendation_correlation_id=admission.recommendation_correlation_id,
    )


def resolve_execution_authorization(
    admission: AutonomyExecutionAdmissionContext,
    evaluation: AutonomyEvaluationResult | None,
    *,
    guard_id: str,
    recorded_at: datetime,
    guard_rules: tuple[AutonomyExecutionGuardRule, ...] = (),
) -> AutonomyExecutionAuthorization:
    """Fail-safe default: missing or insufficient evidence yields DENIED."""
    if evaluation is None:
        return _denied(
            admission=admission,
            guard_id=guard_id,
            recorded_at=recorded_at,
            reasons=("missing autonomy evaluation",),
            evaluation=None,
        )

    if evaluation.recommendation_correlation_id != admission.recommendation_correlation_id:
        return _denied(
            admission=admission,
            guard_id=guard_id,
            recorded_at=recorded_at,
            reasons=("evaluation correlation mismatch",),
            evaluation=evaluation,
        )

    decision = admission.prior_decision
    if decision.decision_id != admission.decision_id:
        return _denied(
            admission=admission,
            guard_id=guard_id,
            recorded_at=recorded_at,
            reasons=("decision id mismatch",),
            evaluation=evaluation,
        )

    if evaluation.verdict is AutonomyEvaluationVerdict.DENIED:
        return _denied(
            admission=admission,
            guard_id=guard_id,
            recorded_at=recorded_at,
            reasons=tuple(evaluation.reasons),
            evaluation=evaluation,
        )

    if decision.autonomy_level is not AutonomyLevel.CONTROLLED_EXECUTION:
        return _denied(
            admission=admission,
            guard_id=guard_id,
            recorded_at=recorded_at,
            reasons=("autonomy level does not permit controlled execution",),
            evaluation=evaluation,
        )

    if not decision.auto_path_allowed:
        return _denied(
            admission=admission,
            guard_id=guard_id,
            recorded_at=recorded_at,
            reasons=("auto execution path not allowed by autonomy decision",),
            evaluation=evaluation,
        )

    approval_required = evaluation.approval_result.approval_required
    token = admission.approval_token_ref
    if approval_required and (token is None or not token.strip()):
        return _denied(
            admission=admission,
            guard_id=guard_id,
            recorded_at=recorded_at,
            reasons=("human approval required but approval token missing",),
            evaluation=evaluation,
        )

    for rule in guard_rules:
        rule_outcome: AutonomyExecutionGuardRuleVerdict = rule.assess(admission, evaluation)
        if not rule_outcome.permitted:
            return _denied(
                admission=admission,
                guard_id=guard_id,
                recorded_at=recorded_at,
                reasons=(rule_outcome.rationale,),
                evaluation=evaluation,
            )

    if evaluation.verdict is AutonomyEvaluationVerdict.CONDITIONAL:
        return AutonomyExecutionAuthorization(
            authorization_id=mint_autonomy_execution_authorization_id(),
            status=AutonomyExecutionAuthorizationStatus.CONDITIONAL,
            decision_id=admission.decision_id,
            evaluation_id=evaluation.evaluation_id,
            autonomy_level=decision.autonomy_level,
            policy_result=evaluation.policy_result,
            recorded_at=recorded_at,
            reasons=tuple(evaluation.reasons),
            guard_id=guard_id,
            recommendation_correlation_id=admission.recommendation_correlation_id,
        )

    return AutonomyExecutionAuthorization(
        authorization_id=mint_autonomy_execution_authorization_id(),
        status=AutonomyExecutionAuthorizationStatus.AUTHORIZED,
        decision_id=admission.decision_id,
        evaluation_id=evaluation.evaluation_id,
        autonomy_level=decision.autonomy_level,
        policy_result=evaluation.policy_result,
        recorded_at=recorded_at,
        reasons=tuple(evaluation.reasons),
        guard_id=guard_id,
        recommendation_correlation_id=admission.recommendation_correlation_id,
    )


def authorization_to_guard_verdict(
    authorization: AutonomyExecutionAuthorization,
) -> tuple[AutonomyGuardVerdict, str]:
    if authorization.status is AutonomyExecutionAuthorizationStatus.AUTHORIZED:
        return AutonomyGuardVerdict.ALLOWED, authorization.reasons[0]
    if authorization.status is AutonomyExecutionAuthorizationStatus.CONDITIONAL:
        return AutonomyGuardVerdict.DEFERRED, authorization.reasons[0]
    return AutonomyGuardVerdict.DENIED, authorization.reasons[0]


__all__ = ["authorization_to_guard_verdict", "resolve_execution_authorization"]
