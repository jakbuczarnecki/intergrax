# © Artur Czarnecki. All rights reserved.

"""Policy-only advisory Token Optimization recommendations (TOKEN-7A)."""

from __future__ import annotations

from intergrax.runtime.token_optimization.contracts import (
    TokenOptimizationAdvisoryRecommendation,
    TokenOptimizationAdvisorySignal,
    TokenOptimizationRecommendationAction,
    TokenOptimizationRecommendationConfidence,
    TokenOptimizationRecommendationReason,
    TokenOptimizationStrategyKind,
)
from intergrax.runtime.token_optimization.prompt_cache import (
    PREFIX_STABILITY_INVALIDATED,
    PREFIX_STABILITY_STABLE,
)

_HIGH_FALLBACK_RATE_THRESHOLD = 0.25
_MIN_SAFE_SAVINGS_RATIO = 0.10
_MIN_VALIDATION_PASS_RATE = 0.95
_MAX_SAFE_FALLBACK_RATE = 0.05
_LOW_SAVINGS_RATIO_THRESHOLD = 0.02


def _has_insufficient_signals(signal: TokenOptimizationAdvisorySignal) -> bool:
    if signal.sample_count == 0:
        return True
    return (
        signal.measured_saved_ratio is None
        and signal.validation_pass_rate is None
        and signal.fallback_rate is None
        and signal.regression_gate_passed is None
    )


def _recommendation(
    signal: TokenOptimizationAdvisorySignal,
    *,
    action: TokenOptimizationRecommendationAction,
    reason: TokenOptimizationRecommendationReason,
    confidence: TokenOptimizationRecommendationConfidence,
    strategy_kind: TokenOptimizationStrategyKind | None = None,
) -> TokenOptimizationAdvisoryRecommendation:
    resolved_strategy_kind = strategy_kind
    if resolved_strategy_kind is None and action in (
        TokenOptimizationRecommendationAction.ENABLE_STRATEGY,
        TokenOptimizationRecommendationAction.DISABLE_STRATEGY,
    ):
        resolved_strategy_kind = signal.strategy_kind
    return TokenOptimizationAdvisoryRecommendation(
        action=action,
        reason=reason,
        confidence=confidence,
        source_type=signal.source_type,
        strategy_kind=resolved_strategy_kind,
        auto_apply_allowed=False,
        raw_content_included=False,
    )


def recommend_token_optimization_action(
    signal: TokenOptimizationAdvisorySignal,
) -> TokenOptimizationAdvisoryRecommendation:
    """Return a deterministic, conservative advisory recommendation (no auto-apply)."""
    if signal.policy_review_required:
        return _recommendation(
            signal,
            action=TokenOptimizationRecommendationAction.REQUIRE_MANUAL_REVIEW,
            reason=TokenOptimizationRecommendationReason.POLICY_REQUIRES_REVIEW,
            confidence=TokenOptimizationRecommendationConfidence.HIGH,
        )

    if signal.protected_region_failure_detected:
        return _recommendation(
            signal,
            action=TokenOptimizationRecommendationAction.ESCALATE_TO_FULL_CONTEXT,
            reason=TokenOptimizationRecommendationReason.PROTECTED_REGION_RISK,
            confidence=TokenOptimizationRecommendationConfidence.HIGH,
        )

    if signal.quality_regression_detected:
        return _recommendation(
            signal,
            action=TokenOptimizationRecommendationAction.ESCALATE_TO_FULL_CONTEXT,
            reason=TokenOptimizationRecommendationReason.QUALITY_REGRESSION_RISK,
            confidence=TokenOptimizationRecommendationConfidence.HIGH,
        )

    if signal.regression_gate_passed is False:
        return _recommendation(
            signal,
            action=TokenOptimizationRecommendationAction.REQUIRE_MANUAL_REVIEW,
            reason=TokenOptimizationRecommendationReason.REGRESSION_GATE_FAILED,
            confidence=TokenOptimizationRecommendationConfidence.HIGH,
        )

    if _has_insufficient_signals(signal):
        return _recommendation(
            signal,
            action=TokenOptimizationRecommendationAction.INSUFFICIENT_DATA,
            reason=TokenOptimizationRecommendationReason.INSUFFICIENT_SIGNALS,
            confidence=TokenOptimizationRecommendationConfidence.NOT_ENOUGH_DATA,
        )

    if (
        signal.fallback_rate is not None
        and signal.fallback_rate >= _HIGH_FALLBACK_RATE_THRESHOLD
        and signal.strategy_kind is not None
    ):
        return _recommendation(
            signal,
            action=TokenOptimizationRecommendationAction.DISABLE_STRATEGY,
            reason=TokenOptimizationRecommendationReason.HIGH_FALLBACK_RATE,
            confidence=TokenOptimizationRecommendationConfidence.HIGH,
            strategy_kind=signal.strategy_kind,
        )

    if (
        signal.cache_hot is True
        and signal.cache_prefix_stability_status == PREFIX_STABILITY_STABLE
    ):
        return _recommendation(
            signal,
            action=TokenOptimizationRecommendationAction.PRESERVE_CACHEABLE_PREFIX,
            reason=TokenOptimizationRecommendationReason.CACHE_PREFIX_HOT,
            confidence=TokenOptimizationRecommendationConfidence.MEDIUM,
        )

    if signal.cache_prefix_stability_status == PREFIX_STABILITY_INVALIDATED:
        return _recommendation(
            signal,
            action=TokenOptimizationRecommendationAction.REQUIRE_MANUAL_REVIEW,
            reason=TokenOptimizationRecommendationReason.CACHE_PREFIX_UNSTABLE,
            confidence=TokenOptimizationRecommendationConfidence.MEDIUM,
        )

    if signal.dynamic_tail_reduction_available:
        return _recommendation(
            signal,
            action=TokenOptimizationRecommendationAction.PREFER_DYNAMIC_TAIL_REDUCTION,
            reason=TokenOptimizationRecommendationReason.DYNAMIC_TAIL_CAN_BE_REDUCED,
            confidence=TokenOptimizationRecommendationConfidence.MEDIUM,
        )

    if (
        signal.regression_gate_passed is True
        and signal.measured_saved_ratio is not None
        and signal.measured_saved_ratio >= _MIN_SAFE_SAVINGS_RATIO
        and signal.validation_pass_rate is not None
        and signal.validation_pass_rate >= _MIN_VALIDATION_PASS_RATE
        and signal.fallback_rate is not None
        and signal.fallback_rate <= _MAX_SAFE_FALLBACK_RATE
        and signal.strategy_kind is not None
    ):
        return _recommendation(
            signal,
            action=TokenOptimizationRecommendationAction.ENABLE_STRATEGY,
            reason=TokenOptimizationRecommendationReason.MEASURED_SAFE_SAVINGS,
            confidence=TokenOptimizationRecommendationConfidence.HIGH,
            strategy_kind=signal.strategy_kind,
        )

    if (
        signal.measured_saved_ratio is not None
        and signal.measured_saved_ratio < _LOW_SAVINGS_RATIO_THRESHOLD
    ):
        return _recommendation(
            signal,
            action=TokenOptimizationRecommendationAction.KEEP_CURRENT,
            reason=TokenOptimizationRecommendationReason.LOW_OR_NO_SAVINGS,
            confidence=TokenOptimizationRecommendationConfidence.MEDIUM,
        )

    if signal.regression_gate_passed is True:
        return _recommendation(
            signal,
            action=TokenOptimizationRecommendationAction.KEEP_CURRENT,
            reason=TokenOptimizationRecommendationReason.REGRESSION_GATE_PASSED,
            confidence=TokenOptimizationRecommendationConfidence.MEDIUM,
        )

    return _recommendation(
        signal,
        action=TokenOptimizationRecommendationAction.INSUFFICIENT_DATA,
        reason=TokenOptimizationRecommendationReason.INSUFFICIENT_SIGNALS,
        confidence=TokenOptimizationRecommendationConfidence.NOT_ENOUGH_DATA,
    )
