# © Artur Czarnecki. All rights reserved.

"""TOKEN-7A: advisory recommendation contract and policy-only recommender tests."""

from __future__ import annotations

import dataclasses

import pytest

from intergrax.runtime.token_optimization.advisory import recommend_token_optimization_action
from intergrax.runtime.token_optimization.contracts import (
    TokenOptimizationAdvisoryRecommendation,
    TokenOptimizationAdvisorySignal,
    TokenOptimizationProfile,
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
from tests.fixtures.token_optimization.advisory_recommendation_corpus import (
    ADVISORY_RECOMMENDATION_CORPUS,
    ADVISORY_RECOMMENDATION_SYNTHETIC_CORPUS_MARKER,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_FORBIDDEN_SAVINGS_FIELD_NAMES = frozenset(
    {
        "saved_tokens",
        "optimized_tokens",
        "baseline_tokens",
        "compressed_tokens",
    }
)

_FORBIDDEN_CONTENT_FIELD_NAMES = frozenset(
    {
        "prompt",
        "content",
        "context",
        "evidence",
        "tool_output",
        "raw_payload",
        "document_text",
    }
)

_REQUIRED_CORPUS_CASE_IDS = frozenset(
    {
        "advisory.policy_review_requires_manual_review",
        "advisory.protected_region_risk_escalates_to_full_context",
        "advisory.quality_regression_escalates_to_full_context",
        "advisory.regression_gate_failed_requires_review",
        "advisory.insufficient_data",
        "advisory.high_fallback_disables_strategy",
        "advisory.hot_stable_cache_preserves_prefix",
        "advisory.invalidated_cache_requires_review",
        "advisory.dynamic_tail_reduction_preferred",
        "advisory.measured_safe_savings_enables_strategy",
        "advisory.low_savings_keeps_current",
        "advisory.regression_gate_passed_keeps_current",
    }
)


def test_recommendation_action_values_are_stable() -> None:
    assert [action.value for action in TokenOptimizationRecommendationAction] == [
        "keep_current",
        "use_conservative_profile",
        "use_balanced_profile",
        "escalate_to_full_context",
        "enable_strategy",
        "disable_strategy",
        "prefer_dynamic_tail_reduction",
        "preserve_cacheable_prefix",
        "require_manual_review",
        "insufficient_data",
    ]


def test_recommendation_reason_values_are_stable() -> None:
    assert [reason.value for reason in TokenOptimizationRecommendationReason] == [
        "quality_regression_risk",
        "protected_region_risk",
        "high_fallback_rate",
        "low_or_no_savings",
        "measured_safe_savings",
        "cache_prefix_hot",
        "cache_prefix_unstable",
        "dynamic_tail_can_be_reduced",
        "regression_gate_failed",
        "regression_gate_passed",
        "insufficient_signals",
        "policy_requires_review",
    ]


def test_recommendation_confidence_values_are_stable() -> None:
    assert [confidence.value for confidence in TokenOptimizationRecommendationConfidence] == [
        "low",
        "medium",
        "high",
        "not_enough_data",
    ]


def test_advisory_signal_validates_ratios_and_sample_count() -> None:
    with pytest.raises(ValueError, match="sample_count cannot be negative"):
        TokenOptimizationAdvisorySignal(
            source_type=TokenOptimizationSourceType.PROMPT,
            sample_count=-1,
        )
    with pytest.raises(ValueError, match="measured_saved_ratio must be between"):
        TokenOptimizationAdvisorySignal(
            source_type=TokenOptimizationSourceType.PROMPT,
            measured_saved_ratio=1.5,
        )
    with pytest.raises(ValueError, match="validation_pass_rate must be between"):
        TokenOptimizationAdvisorySignal(
            source_type=TokenOptimizationSourceType.PROMPT,
            validation_pass_rate=-0.1,
        )
    with pytest.raises(ValueError, match="fallback_rate must be between"):
        TokenOptimizationAdvisorySignal(
            source_type=TokenOptimizationSourceType.PROMPT,
            fallback_rate=2.0,
        )
    with pytest.raises(ValueError, match="cache_prefix_stability_status cannot be empty"):
        TokenOptimizationAdvisorySignal(
            source_type=TokenOptimizationSourceType.PROMPT,
            cache_prefix_stability_status="   ",
        )


def test_advisory_recommendation_rejects_auto_apply() -> None:
    with pytest.raises(ValueError, match="auto_apply_allowed must remain False"):
        TokenOptimizationAdvisoryRecommendation(
            action=TokenOptimizationRecommendationAction.KEEP_CURRENT,
            reason=TokenOptimizationRecommendationReason.REGRESSION_GATE_PASSED,
            confidence=TokenOptimizationRecommendationConfidence.MEDIUM,
            source_type=TokenOptimizationSourceType.PROMPT,
            auto_apply_allowed=True,
        )


def test_advisory_recommendation_rejects_raw_content_flag() -> None:
    with pytest.raises(ValueError, match="raw_content_included must remain False"):
        TokenOptimizationAdvisoryRecommendation(
            action=TokenOptimizationRecommendationAction.KEEP_CURRENT,
            reason=TokenOptimizationRecommendationReason.REGRESSION_GATE_PASSED,
            confidence=TokenOptimizationRecommendationConfidence.MEDIUM,
            source_type=TokenOptimizationSourceType.PROMPT,
            raw_content_included=True,
        )


def test_enable_or_disable_strategy_requires_strategy_kind() -> None:
    with pytest.raises(ValueError, match="strategy_kind must be present"):
        TokenOptimizationAdvisoryRecommendation(
            action=TokenOptimizationRecommendationAction.ENABLE_STRATEGY,
            reason=TokenOptimizationRecommendationReason.MEASURED_SAFE_SAVINGS,
            confidence=TokenOptimizationRecommendationConfidence.HIGH,
            source_type=TokenOptimizationSourceType.RAG_CONTEXT_PACK,
        )
    with pytest.raises(ValueError, match="strategy_kind must be present"):
        TokenOptimizationAdvisoryRecommendation(
            action=TokenOptimizationRecommendationAction.DISABLE_STRATEGY,
            reason=TokenOptimizationRecommendationReason.HIGH_FALLBACK_RATE,
            confidence=TokenOptimizationRecommendationConfidence.HIGH,
            source_type=TokenOptimizationSourceType.TOOL_OUTPUT,
        )


def test_profile_recommendations_require_matching_profile() -> None:
    with pytest.raises(ValueError, match="recommended_profile must be CONSERVATIVE"):
        TokenOptimizationAdvisoryRecommendation(
            action=TokenOptimizationRecommendationAction.USE_CONSERVATIVE_PROFILE,
            reason=TokenOptimizationRecommendationReason.MEASURED_SAFE_SAVINGS,
            confidence=TokenOptimizationRecommendationConfidence.MEDIUM,
            source_type=TokenOptimizationSourceType.PROMPT,
            recommended_profile=TokenOptimizationProfile.BALANCED,
        )
    with pytest.raises(ValueError, match="recommended_profile must be BALANCED"):
        TokenOptimizationAdvisoryRecommendation(
            action=TokenOptimizationRecommendationAction.USE_BALANCED_PROFILE,
            reason=TokenOptimizationRecommendationReason.MEASURED_SAFE_SAVINGS,
            confidence=TokenOptimizationRecommendationConfidence.MEDIUM,
            source_type=TokenOptimizationSourceType.PROMPT,
            recommended_profile=TokenOptimizationProfile.CONSERVATIVE,
        )


def test_policy_review_requires_manual_review() -> None:
    result = recommend_token_optimization_action(
        TokenOptimizationAdvisorySignal(
            source_type=TokenOptimizationSourceType.PROMPT,
            policy_review_required=True,
            sample_count=5,
            regression_gate_passed=True,
        )
    )
    assert result.action is TokenOptimizationRecommendationAction.REQUIRE_MANUAL_REVIEW
    assert result.reason is TokenOptimizationRecommendationReason.POLICY_REQUIRES_REVIEW
    assert result.confidence is TokenOptimizationRecommendationConfidence.HIGH


def test_protected_region_risk_escalates_to_full_context() -> None:
    result = recommend_token_optimization_action(
        TokenOptimizationAdvisorySignal(
            source_type=TokenOptimizationSourceType.TOOL_OUTPUT,
            protected_region_failure_detected=True,
            sample_count=5,
            regression_gate_passed=True,
        )
    )
    assert result.action is TokenOptimizationRecommendationAction.ESCALATE_TO_FULL_CONTEXT
    assert result.reason is TokenOptimizationRecommendationReason.PROTECTED_REGION_RISK


def test_quality_regression_escalates_to_full_context() -> None:
    result = recommend_token_optimization_action(
        TokenOptimizationAdvisorySignal(
            source_type=TokenOptimizationSourceType.RAG_CONTEXT_PACK,
            quality_regression_detected=True,
            sample_count=5,
            regression_gate_passed=True,
        )
    )
    assert result.action is TokenOptimizationRecommendationAction.ESCALATE_TO_FULL_CONTEXT
    assert result.reason is TokenOptimizationRecommendationReason.QUALITY_REGRESSION_RISK


def test_regression_gate_failed_requires_review() -> None:
    result = recommend_token_optimization_action(
        TokenOptimizationAdvisorySignal(
            source_type=TokenOptimizationSourceType.TOOL_OUTPUT,
            sample_count=5,
            regression_gate_passed=False,
            measured_saved_ratio=0.10,
        )
    )
    assert result.action is TokenOptimizationRecommendationAction.REQUIRE_MANUAL_REVIEW
    assert result.reason is TokenOptimizationRecommendationReason.REGRESSION_GATE_FAILED


def test_insufficient_data_returns_insufficient_data() -> None:
    result = recommend_token_optimization_action(
        TokenOptimizationAdvisorySignal(
            source_type=TokenOptimizationSourceType.PROMPT,
            sample_count=0,
        )
    )
    assert result.action is TokenOptimizationRecommendationAction.INSUFFICIENT_DATA
    assert result.reason is TokenOptimizationRecommendationReason.INSUFFICIENT_SIGNALS
    assert result.confidence is TokenOptimizationRecommendationConfidence.NOT_ENOUGH_DATA


def test_high_fallback_rate_disables_strategy() -> None:
    result = recommend_token_optimization_action(
        TokenOptimizationAdvisorySignal(
            source_type=TokenOptimizationSourceType.TOOL_OUTPUT,
            strategy_kind=TokenOptimizationStrategyKind.EXTRACTIVE_FILTERING,
            sample_count=10,
            fallback_rate=0.30,
            measured_saved_ratio=0.08,
            regression_gate_passed=True,
        )
    )
    assert result.action is TokenOptimizationRecommendationAction.DISABLE_STRATEGY
    assert result.reason is TokenOptimizationRecommendationReason.HIGH_FALLBACK_RATE
    assert result.strategy_kind is TokenOptimizationStrategyKind.EXTRACTIVE_FILTERING


def test_hot_stable_cache_recommends_preserve_prefix() -> None:
    result = recommend_token_optimization_action(
        TokenOptimizationAdvisorySignal(
            source_type=TokenOptimizationSourceType.SYSTEM_POLICY,
            sample_count=10,
            measured_saved_ratio=0.06,
            fallback_rate=0.08,
            regression_gate_passed=True,
            cache_hot=True,
            cache_prefix_stability_status=PREFIX_STABILITY_STABLE,
        )
    )
    assert result.action is TokenOptimizationRecommendationAction.PRESERVE_CACHEABLE_PREFIX
    assert result.reason is TokenOptimizationRecommendationReason.CACHE_PREFIX_HOT


def test_invalidated_cache_requires_review() -> None:
    result = recommend_token_optimization_action(
        TokenOptimizationAdvisorySignal(
            source_type=TokenOptimizationSourceType.PROMPT,
            sample_count=10,
            measured_saved_ratio=0.07,
            regression_gate_passed=True,
            cache_prefix_stability_status=PREFIX_STABILITY_INVALIDATED,
        )
    )
    assert result.action is TokenOptimizationRecommendationAction.REQUIRE_MANUAL_REVIEW
    assert result.reason is TokenOptimizationRecommendationReason.CACHE_PREFIX_UNSTABLE


def test_dynamic_tail_reduction_is_preferred() -> None:
    result = recommend_token_optimization_action(
        TokenOptimizationAdvisorySignal(
            source_type=TokenOptimizationSourceType.CONVERSATION_HISTORY,
            sample_count=10,
            measured_saved_ratio=0.05,
            validation_pass_rate=0.90,
            fallback_rate=0.10,
            regression_gate_passed=True,
            dynamic_tail_reduction_available=True,
        )
    )
    assert (
        result.action is TokenOptimizationRecommendationAction.PREFER_DYNAMIC_TAIL_REDUCTION
    )
    assert result.reason is TokenOptimizationRecommendationReason.DYNAMIC_TAIL_CAN_BE_REDUCED


def test_measured_safe_savings_enables_strategy() -> None:
    result = recommend_token_optimization_action(
        TokenOptimizationAdvisorySignal(
            source_type=TokenOptimizationSourceType.RAG_CONTEXT_PACK,
            strategy_kind=TokenOptimizationStrategyKind.DEDUPLICATION,
            sample_count=50,
            measured_saved_ratio=0.15,
            validation_pass_rate=0.98,
            fallback_rate=0.02,
            regression_gate_passed=True,
        )
    )
    assert result.action is TokenOptimizationRecommendationAction.ENABLE_STRATEGY
    assert result.reason is TokenOptimizationRecommendationReason.MEASURED_SAFE_SAVINGS
    assert result.strategy_kind is TokenOptimizationStrategyKind.DEDUPLICATION


def test_low_savings_keeps_current() -> None:
    result = recommend_token_optimization_action(
        TokenOptimizationAdvisorySignal(
            source_type=TokenOptimizationSourceType.TOOL_CATALOG,
            sample_count=25,
            measured_saved_ratio=0.01,
            regression_gate_passed=True,
        )
    )
    assert result.action is TokenOptimizationRecommendationAction.KEEP_CURRENT
    assert result.reason is TokenOptimizationRecommendationReason.LOW_OR_NO_SAVINGS


def test_regression_gate_passed_keeps_current() -> None:
    result = recommend_token_optimization_action(
        TokenOptimizationAdvisorySignal(
            source_type=TokenOptimizationSourceType.MEMORY,
            sample_count=30,
            measured_saved_ratio=0.05,
            validation_pass_rate=0.90,
            fallback_rate=0.10,
            regression_gate_passed=True,
        )
    )
    assert result.action is TokenOptimizationRecommendationAction.KEEP_CURRENT
    assert result.reason is TokenOptimizationRecommendationReason.REGRESSION_GATE_PASSED


def test_synthetic_corpus_case_ids_are_unique() -> None:
    case_ids = [case.case_id for case in ADVISORY_RECOMMENDATION_CORPUS]
    assert len(case_ids) == len(set(case_ids))


def test_synthetic_corpus_has_required_cases() -> None:
    case_ids = {case.case_id for case in ADVISORY_RECOMMENDATION_CORPUS}
    assert _REQUIRED_CORPUS_CASE_IDS.issubset(case_ids)
    for case in ADVISORY_RECOMMENDATION_CORPUS:
        result = recommend_token_optimization_action(case.signal)
        assert result.action is case.expected_action
        assert result.reason is case.expected_reason
        assert result.confidence is case.expected_confidence


def test_synthetic_corpus_has_marker() -> None:
    assert (
        ADVISORY_RECOMMENDATION_SYNTHETIC_CORPUS_MARKER
        == "SYNTHETIC_ADVISORY_RECOMMENDATION_CORPUS_V1"
    )
    assert all(
        case.synthetic_marker == ADVISORY_RECOMMENDATION_SYNTHETIC_CORPUS_MARKER
        for case in ADVISORY_RECOMMENDATION_CORPUS
    )


def test_recommendation_reports_are_raw_content_safe() -> None:
    signal = TokenOptimizationAdvisorySignal(
        source_type=TokenOptimizationSourceType.RAG_CONTEXT_PACK,
        sample_count=10,
        measured_saved_ratio=0.05,
        regression_gate_passed=True,
    )
    result = recommend_token_optimization_action(signal)
    signal_fields = {field.name for field in dataclasses.fields(TokenOptimizationAdvisorySignal)}
    result_fields = {
        field.name for field in dataclasses.fields(TokenOptimizationAdvisoryRecommendation)
    }
    assert not _FORBIDDEN_CONTENT_FIELD_NAMES.intersection(signal_fields)
    assert not _FORBIDDEN_CONTENT_FIELD_NAMES.intersection(result_fields)
    assert not _FORBIDDEN_SAVINGS_FIELD_NAMES.intersection(result_fields)
    assert result.raw_content_included is False
    assert result.auto_apply_allowed is False


def test_recommendations_are_deterministic() -> None:
    signal = TokenOptimizationAdvisorySignal(
        source_type=TokenOptimizationSourceType.TOOL_OUTPUT,
        strategy_kind=TokenOptimizationStrategyKind.DEDUPLICATION,
        sample_count=50,
        measured_saved_ratio=0.15,
        validation_pass_rate=0.98,
        fallback_rate=0.02,
        regression_gate_passed=True,
    )
    first = recommend_token_optimization_action(signal)
    second = recommend_token_optimization_action(signal)
    assert first == second
