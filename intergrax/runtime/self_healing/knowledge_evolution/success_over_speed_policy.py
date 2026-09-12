# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Minimal success-over-speed comparison policy (SELF-HEALING R5.4)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.self_healing.knowledge_evolution.comparison import (
    StrategyComparisonPreference,
    StrategyComparisonResult,
    StrategyComparisonScope,
    StrategyComparisonSubject,
)


def _metric_value(subject: StrategyComparisonSubject, name: str) -> float | None:
    for metric in subject.metric_bundle.metrics:
        if metric.name == name:
            return metric.value
    return None


@dataclass(frozen=True, slots=True)
class SuccessOverSpeedComparisonPolicy:
    policy_id: str = "platform.success_over_speed"

    def compare(
        self,
        scope: StrategyComparisonScope,
        left: StrategyComparisonSubject,
        right: StrategyComparisonSubject,
    ) -> StrategyComparisonResult:
        left_success = _metric_value(left, "success_rate")
        right_success = _metric_value(right, "success_rate")
        if left_success is None or right_success is None:
            return StrategyComparisonResult(
                policy_id=self.policy_id,
                preference=StrategyComparisonPreference.INCONCLUSIVE,
                dimension_weights_ref=scope.dimension_weights_ref,
                evidence_refs=(),
                rationale="Missing success_rate metrics for comparison.",
            )
        if left_success > right_success:
            preference = StrategyComparisonPreference.PREFER_LEFT
            rationale = f"{left.strategy_id} has higher success_rate than {right.strategy_id}."
        elif right_success > left_success:
            preference = StrategyComparisonPreference.PREFER_RIGHT
            rationale = f"{right.strategy_id} has higher success_rate than {left.strategy_id}."
        else:
            preference = StrategyComparisonPreference.INCONCLUSIVE
            rationale = "Equal success_rate — no preference."
        evidence = left.metric_bundle.metrics[0].evidence_refs + right.metric_bundle.metrics[0].evidence_refs
        return StrategyComparisonResult(
            policy_id=self.policy_id,
            preference=preference,
            dimension_weights_ref=scope.dimension_weights_ref,
            evidence_refs=evidence,
            rationale=rationale,
        )


__all__ = ["SuccessOverSpeedComparisonPolicy"]
