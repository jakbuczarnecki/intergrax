# © Artur Czarnecki. All rights reserved.

"""Synthetic corpus for advisory evaluation and reporting (TOKEN-7B)."""

from __future__ import annotations

from intergrax.runtime.token_optimization.advisory_evaluation import (
    TokenOptimizationAdvisoryEvaluationCase,
)
from intergrax.runtime.token_optimization.contracts import (
    TokenOptimizationAdvisorySignal,
    TokenOptimizationRecommendationAction,
    TokenOptimizationRecommendationConfidence,
    TokenOptimizationRecommendationReason,
    TokenOptimizationSourceType,
    TokenOptimizationStrategyKind,
)
from intergrax.runtime.token_optimization.prompt_cache import (
    PREFIX_STABILITY_INVALIDATED,
    PREFIX_STABILITY_STABLE,
)

ADVISORY_EVALUATION_SYNTHETIC_CORPUS_MARKER = "SYNTHETIC_ADVISORY_EVALUATION_CORPUS_V1"

_METADATA = {"synthetic_marker": ADVISORY_EVALUATION_SYNTHETIC_CORPUS_MARKER}


ADVISORY_EVALUATION_CORPUS: tuple[TokenOptimizationAdvisoryEvaluationCase, ...] = (
    TokenOptimizationAdvisoryEvaluationCase(
        case_id="advisory_eval.policy_review_requires_manual_review",
        title="Policy review flag requires manual review",
        signal=TokenOptimizationAdvisorySignal(
            source_type=TokenOptimizationSourceType.RAG_CONTEXT_PACK,
            policy_review_required=True,
            sample_count=10,
            measured_saved_ratio=0.15,
            regression_gate_passed=True,
        ),
        expected_action=TokenOptimizationRecommendationAction.REQUIRE_MANUAL_REVIEW,
        expected_reason=TokenOptimizationRecommendationReason.POLICY_REQUIRES_REVIEW,
        expected_confidence=TokenOptimizationRecommendationConfidence.HIGH,
        metadata=_METADATA,
    ),
    TokenOptimizationAdvisoryEvaluationCase(
        case_id="advisory_eval.protected_region_risk_escalates_to_full_context",
        title="Protected region failure escalates to full context",
        signal=TokenOptimizationAdvisorySignal(
            source_type=TokenOptimizationSourceType.TOOL_OUTPUT,
            protected_region_failure_detected=True,
            sample_count=10,
            measured_saved_ratio=0.12,
            regression_gate_passed=True,
        ),
        expected_action=TokenOptimizationRecommendationAction.ESCALATE_TO_FULL_CONTEXT,
        expected_reason=TokenOptimizationRecommendationReason.PROTECTED_REGION_RISK,
        expected_confidence=TokenOptimizationRecommendationConfidence.HIGH,
        metadata=_METADATA,
    ),
    TokenOptimizationAdvisoryEvaluationCase(
        case_id="advisory_eval.quality_regression_escalates_to_full_context",
        title="Quality regression escalates to full context",
        signal=TokenOptimizationAdvisorySignal(
            source_type=TokenOptimizationSourceType.RAG_CONTEXT_PACK,
            quality_regression_detected=True,
            sample_count=10,
            measured_saved_ratio=0.12,
            regression_gate_passed=True,
        ),
        expected_action=TokenOptimizationRecommendationAction.ESCALATE_TO_FULL_CONTEXT,
        expected_reason=TokenOptimizationRecommendationReason.QUALITY_REGRESSION_RISK,
        expected_confidence=TokenOptimizationRecommendationConfidence.HIGH,
        metadata=_METADATA,
    ),
    TokenOptimizationAdvisoryEvaluationCase(
        case_id="advisory_eval.regression_gate_failed_requires_review",
        title="Regression gate failure requires manual review",
        signal=TokenOptimizationAdvisorySignal(
            source_type=TokenOptimizationSourceType.TOOL_OUTPUT,
            sample_count=10,
            measured_saved_ratio=0.12,
            regression_gate_passed=False,
        ),
        expected_action=TokenOptimizationRecommendationAction.REQUIRE_MANUAL_REVIEW,
        expected_reason=TokenOptimizationRecommendationReason.REGRESSION_GATE_FAILED,
        expected_confidence=TokenOptimizationRecommendationConfidence.HIGH,
        metadata=_METADATA,
    ),
    TokenOptimizationAdvisoryEvaluationCase(
        case_id="advisory_eval.insufficient_data",
        title="Missing signals return insufficient data",
        signal=TokenOptimizationAdvisorySignal(
            source_type=TokenOptimizationSourceType.PROMPT,
            sample_count=0,
        ),
        expected_action=TokenOptimizationRecommendationAction.INSUFFICIENT_DATA,
        expected_reason=TokenOptimizationRecommendationReason.INSUFFICIENT_SIGNALS,
        expected_confidence=TokenOptimizationRecommendationConfidence.NOT_ENOUGH_DATA,
        metadata=_METADATA,
    ),
    TokenOptimizationAdvisoryEvaluationCase(
        case_id="advisory_eval.high_fallback_disables_strategy",
        title="High fallback rate disables strategy",
        signal=TokenOptimizationAdvisorySignal(
            source_type=TokenOptimizationSourceType.TOOL_OUTPUT,
            strategy_kind=TokenOptimizationStrategyKind.EXTRACTIVE_FILTERING,
            sample_count=20,
            measured_saved_ratio=0.08,
            fallback_rate=0.30,
            regression_gate_passed=True,
        ),
        expected_action=TokenOptimizationRecommendationAction.DISABLE_STRATEGY,
        expected_reason=TokenOptimizationRecommendationReason.HIGH_FALLBACK_RATE,
        expected_confidence=TokenOptimizationRecommendationConfidence.HIGH,
        metadata=_METADATA,
    ),
    TokenOptimizationAdvisoryEvaluationCase(
        case_id="advisory_eval.hot_stable_cache_preserves_prefix",
        title="Hot stable cache recommends preserving cacheable prefix",
        signal=TokenOptimizationAdvisorySignal(
            source_type=TokenOptimizationSourceType.SYSTEM_POLICY,
            sample_count=15,
            measured_saved_ratio=0.06,
            fallback_rate=0.08,
            regression_gate_passed=True,
            cache_hot=True,
            cache_prefix_stability_status=PREFIX_STABILITY_STABLE,
        ),
        expected_action=TokenOptimizationRecommendationAction.PRESERVE_CACHEABLE_PREFIX,
        expected_reason=TokenOptimizationRecommendationReason.CACHE_PREFIX_HOT,
        expected_confidence=TokenOptimizationRecommendationConfidence.MEDIUM,
        metadata=_METADATA,
    ),
    TokenOptimizationAdvisoryEvaluationCase(
        case_id="advisory_eval.invalidated_cache_requires_review",
        title="Invalidated cache prefix requires manual review",
        signal=TokenOptimizationAdvisorySignal(
            source_type=TokenOptimizationSourceType.PROMPT,
            sample_count=12,
            measured_saved_ratio=0.07,
            regression_gate_passed=True,
            cache_prefix_stability_status=PREFIX_STABILITY_INVALIDATED,
        ),
        expected_action=TokenOptimizationRecommendationAction.REQUIRE_MANUAL_REVIEW,
        expected_reason=TokenOptimizationRecommendationReason.CACHE_PREFIX_UNSTABLE,
        expected_confidence=TokenOptimizationRecommendationConfidence.MEDIUM,
        metadata=_METADATA,
    ),
    TokenOptimizationAdvisoryEvaluationCase(
        case_id="advisory_eval.dynamic_tail_reduction_preferred",
        title="Dynamic tail reduction preferred when available",
        signal=TokenOptimizationAdvisorySignal(
            source_type=TokenOptimizationSourceType.CONVERSATION_HISTORY,
            sample_count=10,
            measured_saved_ratio=0.05,
            validation_pass_rate=0.90,
            fallback_rate=0.10,
            regression_gate_passed=True,
            dynamic_tail_reduction_available=True,
        ),
        expected_action=TokenOptimizationRecommendationAction.PREFER_DYNAMIC_TAIL_REDUCTION,
        expected_reason=TokenOptimizationRecommendationReason.DYNAMIC_TAIL_CAN_BE_REDUCED,
        expected_confidence=TokenOptimizationRecommendationConfidence.MEDIUM,
        metadata=_METADATA,
    ),
    TokenOptimizationAdvisoryEvaluationCase(
        case_id="advisory_eval.measured_safe_savings_enables_strategy",
        title="Measured safe savings enables strategy",
        signal=TokenOptimizationAdvisorySignal(
            source_type=TokenOptimizationSourceType.RAG_CONTEXT_PACK,
            strategy_kind=TokenOptimizationStrategyKind.DEDUPLICATION,
            sample_count=50,
            measured_saved_ratio=0.15,
            validation_pass_rate=0.98,
            fallback_rate=0.02,
            regression_gate_passed=True,
        ),
        expected_action=TokenOptimizationRecommendationAction.ENABLE_STRATEGY,
        expected_reason=TokenOptimizationRecommendationReason.MEASURED_SAFE_SAVINGS,
        expected_confidence=TokenOptimizationRecommendationConfidence.HIGH,
        metadata=_METADATA,
    ),
    TokenOptimizationAdvisoryEvaluationCase(
        case_id="advisory_eval.low_savings_keeps_current",
        title="Low measured savings keeps current posture",
        signal=TokenOptimizationAdvisorySignal(
            source_type=TokenOptimizationSourceType.TOOL_CATALOG,
            sample_count=25,
            measured_saved_ratio=0.01,
            regression_gate_passed=True,
        ),
        expected_action=TokenOptimizationRecommendationAction.KEEP_CURRENT,
        expected_reason=TokenOptimizationRecommendationReason.LOW_OR_NO_SAVINGS,
        expected_confidence=TokenOptimizationRecommendationConfidence.MEDIUM,
        metadata=_METADATA,
    ),
    TokenOptimizationAdvisoryEvaluationCase(
        case_id="advisory_eval.regression_gate_passed_keeps_current",
        title="Regression gate passed keeps current posture",
        signal=TokenOptimizationAdvisorySignal(
            source_type=TokenOptimizationSourceType.MEMORY,
            sample_count=30,
            measured_saved_ratio=0.05,
            validation_pass_rate=0.90,
            fallback_rate=0.10,
            regression_gate_passed=True,
        ),
        expected_action=TokenOptimizationRecommendationAction.KEEP_CURRENT,
        expected_reason=TokenOptimizationRecommendationReason.REGRESSION_GATE_PASSED,
        expected_confidence=TokenOptimizationRecommendationConfidence.MEDIUM,
        metadata=_METADATA,
    ),
)
