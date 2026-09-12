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


def _tie_break_by_freshness(
    left: StrategyComparisonSubject,
    right: StrategyComparisonSubject,
) -> tuple[StrategyComparisonPreference, str]:
    left_fresh = left.knowledge_freshness_score
    right_fresh = right.knowledge_freshness_score
    if left_fresh is not None and right_fresh is not None and left_fresh != right_fresh:
        if left_fresh > right_fresh:
            return (
                StrategyComparisonPreference.PREFER_LEFT,
                "Equal success_rate — prefer fresher knowledge (left).",
            )
        return (
            StrategyComparisonPreference.PREFER_RIGHT,
            "Equal success_rate — prefer fresher knowledge (right).",
        )
    return StrategyComparisonPreference.INCONCLUSIVE, "Equal success_rate — no preference."


def _append_context_rationale(
    scope: StrategyComparisonScope,
    left: StrategyComparisonSubject,
    right: StrategyComparisonSubject,
    rationale: str,
) -> str:
    if scope.operating_context is not None and not scope.operating_context.is_empty:
        label = scope.operating_context.environment_label or scope.operating_context.problem_type
        if label is not None:
            return f"{rationale} Scoped context: {label}."
    if left.operating_context is not None and right.operating_context is not None:
        if (
            left.operating_context.environment_label
            and right.operating_context.environment_label
            and left.operating_context.environment_label != right.operating_context.environment_label
        ):
            return (
                f"{rationale} Operating environments differ "
                f"({left.operating_context.environment_label} vs "
                f"{right.operating_context.environment_label})."
            )
    return rationale


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
            preference, rationale = _tie_break_by_freshness(left, right)
        rationale = _append_context_rationale(scope, left, right, rationale)
        evidence = left.metric_bundle.metrics[0].evidence_refs + right.metric_bundle.metrics[0].evidence_refs
        return StrategyComparisonResult(
            policy_id=self.policy_id,
            preference=preference,
            dimension_weights_ref=scope.dimension_weights_ref,
            evidence_refs=evidence,
            rationale=rationale,
        )


__all__ = ["SuccessOverSpeedComparisonPolicy"]
