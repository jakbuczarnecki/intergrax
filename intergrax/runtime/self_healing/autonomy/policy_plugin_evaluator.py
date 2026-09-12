# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Adapter from R6.1 ``AutonomyPolicy`` to R6.2 policy evaluation result."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.self_healing.autonomy.level import AutonomyLevel
from intergrax.contracts.self_healing.autonomy.policy import AutonomyPolicy
from intergrax.contracts.self_healing.autonomy.policy_evaluation import (
    AutonomyPolicyEvaluationResult,
    AutonomyPolicyEvaluationVerdict,
)
from intergrax.contracts.self_healing.autonomy.request import AutonomyControlRequest

_LEVEL_ORDER: tuple[AutonomyLevel, ...] = (
    AutonomyLevel.OBSERVE_ONLY,
    AutonomyLevel.RECOMMEND_ONLY,
    AutonomyLevel.APPROVAL_REQUIRED,
    AutonomyLevel.CONTROLLED_EXECUTION,
)


def _policy_verdict_for_required_level(
    suggested_level: AutonomyLevel,
    required_level: AutonomyLevel,
) -> AutonomyPolicyEvaluationVerdict:
    suggested_index = _LEVEL_ORDER.index(suggested_level)
    required_index = _LEVEL_ORDER.index(required_level)
    if suggested_index >= required_index:
        return AutonomyPolicyEvaluationVerdict.PASS
    return AutonomyPolicyEvaluationVerdict.FAIL


@dataclass(frozen=True, slots=True)
class AutonomyPolicyPluginEvaluator:
    policy: AutonomyPolicy
    _evaluator_id: str = "platform.policy_plugin_evaluator"

    @property
    def evaluator_id(self) -> str:
        return self._evaluator_id

    def evaluate(self, request: AutonomyControlRequest) -> AutonomyPolicyEvaluationResult:
        outcome = self.policy.evaluate(request)
        verdict = _policy_verdict_for_required_level(
            outcome.suggested_level,
            request.decision_context.required_autonomy_level,
        )
        return AutonomyPolicyEvaluationResult(
            evaluator_id=self.evaluator_id,
            policy_id=outcome.policy_id,
            policy_version=outcome.policy_version,
            verdict=verdict,
            suggested_level=outcome.suggested_level,
            constraint_descriptors=outcome.constraint_descriptors,
            rationale=outcome.rationale,
        )


__all__ = ["AutonomyPolicyPluginEvaluator"]
