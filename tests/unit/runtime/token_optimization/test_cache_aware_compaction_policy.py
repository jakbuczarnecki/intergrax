# © Artur Czarnecki. All rights reserved.

"""TOKEN-OPT-5E: cache-aware compaction timing policy tests."""

from __future__ import annotations

import dataclasses

import pytest

from intergrax.runtime.token_optimization.contracts import (
    CacheAwareCompactionDecision,
    CacheAwareCompactionReason,
    CacheAwareCompactionTarget,
    CacheAwareCompactionTimingDecision,
    CacheAwareCompactionTimingInput,
    PromptCacheInvalidationReason,
)
from intergrax.runtime.token_optimization.prompt_cache import (
    PREFIX_STABILITY_INVALIDATED,
    PREFIX_STABILITY_STABLE,
    decide_cache_aware_compaction_timing,
)
from tests.fixtures.token_optimization.cache_aware_compaction_corpus import (
    CACHE_AWARE_COMPACTION_CORPUS,
    CACHE_AWARE_COMPACTION_SYNTHETIC_CORPUS_MARKER,
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

_REQUIRED_CORPUS_CASE_IDS = frozenset(
    {
        "cache_aware_compaction.dynamic_tail_safe_to_reduce",
        "cache_aware_compaction.cold_history_safe_to_compact",
        "cache_aware_compaction.hot_stable_prefix_deferred",
        "cache_aware_compaction.near_expiry_stable_prefix_runs",
        "cache_aware_compaction.prefix_not_stable_defers",
        "cache_aware_compaction.low_benefit_bypasses",
        "cache_aware_compaction.full_thread_rewrite_requires_review",
        "cache_aware_compaction.protected_or_semantic_risk_requires_review",
        "cache_aware_compaction.unknown_target_requires_review",
        "cache_aware_compaction.insufficient_hotness_signals_requires_review",
    }
)


def test_cache_aware_compaction_target_values_are_stable() -> None:
    assert [target.value for target in CacheAwareCompactionTarget] == [
        "stable_prefix",
        "dynamic_tail",
        "cold_history",
        "full_thread",
        "unknown",
    ]


def test_cache_aware_compaction_decision_values_are_stable() -> None:
    assert [decision.value for decision in CacheAwareCompactionDecision] == [
        "run",
        "defer",
        "bypass",
        "require_manual_review",
    ]


def test_cache_aware_compaction_reason_values_are_stable() -> None:
    assert [reason.value for reason in CacheAwareCompactionReason] == [
        "dynamic_tail_safe_to_reduce",
        "cold_history_safe_to_compact",
        "cache_invalidation_cost_too_high",
        "prefix_not_stable",
        "cache_near_expiry",
        "low_content_reduction_benefit",
        "full_thread_rewrite_risk",
        "protected_or_semantic_risk",
        "insufficient_signals",
    ]


def test_timing_input_validates_non_negative_fields() -> None:
    with pytest.raises(ValueError, match="ttl_seconds_remaining cannot be negative"):
        CacheAwareCompactionTimingInput(
            target=CacheAwareCompactionTarget.DYNAMIC_TAIL,
            ttl_seconds_remaining=-1,
        )
    with pytest.raises(
        ValueError, match="near_expiry_threshold_seconds cannot be negative"
    ):
        CacheAwareCompactionTimingInput(
            target=CacheAwareCompactionTarget.DYNAMIC_TAIL,
            near_expiry_threshold_seconds=-1,
        )
    with pytest.raises(
        ValueError, match="estimated_content_reduction_chars cannot be negative"
    ):
        CacheAwareCompactionTimingInput(
            target=CacheAwareCompactionTarget.DYNAMIC_TAIL,
            estimated_content_reduction_chars=-1,
        )
    with pytest.raises(
        ValueError,
        match="estimated_cache_invalidation_cost_tokens cannot be negative",
    ):
        CacheAwareCompactionTimingInput(
            target=CacheAwareCompactionTarget.DYNAMIC_TAIL,
            estimated_cache_invalidation_cost_tokens=-1,
        )
    with pytest.raises(ValueError, match="prefix_stability_status cannot be empty"):
        CacheAwareCompactionTimingInput(
            target=CacheAwareCompactionTarget.DYNAMIC_TAIL,
            prefix_stability_status="   ",
        )


def test_timing_decision_rejects_raw_content_flag() -> None:
    with pytest.raises(ValueError, match="raw_content_included must remain False"):
        CacheAwareCompactionTimingDecision(
            decision=CacheAwareCompactionDecision.BYPASS,
            reason=CacheAwareCompactionReason.LOW_CONTENT_REDUCTION_BENEFIT,
            target=CacheAwareCompactionTarget.DYNAMIC_TAIL,
            cache_hot=False,
            ttl_seconds_remaining=0,
            estimated_content_reduction_chars=0,
            estimated_cache_invalidation_cost_tokens=0,
            raw_content_included=True,
        )


def test_dynamic_tail_safe_to_reduce_runs() -> None:
    timing_input = CacheAwareCompactionTimingInput(
        target=CacheAwareCompactionTarget.DYNAMIC_TAIL,
        prefix_stability_status=PREFIX_STABILITY_STABLE,
        dynamic_tail_reduction_available=True,
        estimated_content_reduction_chars=100,
    )
    first = decide_cache_aware_compaction_timing(timing_input)
    second = decide_cache_aware_compaction_timing(timing_input)
    assert first == second
    assert first.decision is CacheAwareCompactionDecision.RUN
    assert first.reason is CacheAwareCompactionReason.DYNAMIC_TAIL_SAFE_TO_REDUCE


def test_cold_history_safe_to_compact_runs() -> None:
    result = decide_cache_aware_compaction_timing(
        CacheAwareCompactionTimingInput(
            target=CacheAwareCompactionTarget.COLD_HISTORY,
            cache_hot=False,
            estimated_content_reduction_chars=200,
        )
    )
    assert result.decision is CacheAwareCompactionDecision.RUN
    assert result.reason is CacheAwareCompactionReason.COLD_HISTORY_SAFE_TO_COMPACT


def test_hot_stable_prefix_defers_when_invalidation_cost_too_high() -> None:
    result = decide_cache_aware_compaction_timing(
        CacheAwareCompactionTimingInput(
            target=CacheAwareCompactionTarget.STABLE_PREFIX,
            prefix_stability_status=PREFIX_STABILITY_STABLE,
            cache_hot=True,
            ttl_seconds_remaining=900,
            estimated_content_reduction_chars=50,
            estimated_cache_invalidation_cost_tokens=200,
        )
    )
    assert result.decision is CacheAwareCompactionDecision.DEFER
    assert result.reason is CacheAwareCompactionReason.CACHE_INVALIDATION_COST_TOO_HIGH


def test_near_expiry_stable_prefix_runs() -> None:
    result = decide_cache_aware_compaction_timing(
        CacheAwareCompactionTimingInput(
            target=CacheAwareCompactionTarget.STABLE_PREFIX,
            prefix_stability_status=PREFIX_STABILITY_STABLE,
            cache_hot=True,
            ttl_seconds_remaining=10,
            near_expiry_threshold_seconds=60,
            estimated_content_reduction_chars=50,
            estimated_cache_invalidation_cost_tokens=500,
        )
    )
    assert result.decision is CacheAwareCompactionDecision.RUN
    assert result.reason is CacheAwareCompactionReason.CACHE_NEAR_EXPIRY


def test_prefix_not_stable_defers_for_stable_prefix_target() -> None:
    result = decide_cache_aware_compaction_timing(
        CacheAwareCompactionTimingInput(
            target=CacheAwareCompactionTarget.STABLE_PREFIX,
            prefix_stability_status=PREFIX_STABILITY_INVALIDATED,
            invalidation_reason=PromptCacheInvalidationReason.PREFIX_CHANGED,
            cache_hot=True,
            ttl_seconds_remaining=400,
        )
    )
    assert result.decision is CacheAwareCompactionDecision.DEFER
    assert result.reason is CacheAwareCompactionReason.PREFIX_NOT_STABLE


def test_low_benefit_bypasses() -> None:
    result = decide_cache_aware_compaction_timing(
        CacheAwareCompactionTimingInput(
            target=CacheAwareCompactionTarget.DYNAMIC_TAIL,
            estimated_content_reduction_chars=0,
            dynamic_tail_reduction_available=False,
        )
    )
    assert result.decision is CacheAwareCompactionDecision.BYPASS
    assert result.reason is CacheAwareCompactionReason.LOW_CONTENT_REDUCTION_BENEFIT


def test_full_thread_rewrite_requires_review() -> None:
    result = decide_cache_aware_compaction_timing(
        CacheAwareCompactionTimingInput(
            target=CacheAwareCompactionTarget.FULL_THREAD,
            prefix_stability_status=PREFIX_STABILITY_STABLE,
            cache_hot=False,
            ttl_seconds_remaining=200,
            estimated_content_reduction_chars=300,
            estimated_cache_invalidation_cost_tokens=50,
        )
    )
    assert result.decision is CacheAwareCompactionDecision.REQUIRE_MANUAL_REVIEW
    assert result.reason is CacheAwareCompactionReason.FULL_THREAD_REWRITE_RISK


def test_protected_or_semantic_risk_requires_review() -> None:
    result = decide_cache_aware_compaction_timing(
        CacheAwareCompactionTimingInput(
            target=CacheAwareCompactionTarget.DYNAMIC_TAIL,
            protected_or_semantic_risk=True,
            dynamic_tail_reduction_available=True,
        )
    )
    assert result.decision is CacheAwareCompactionDecision.REQUIRE_MANUAL_REVIEW
    assert result.reason is CacheAwareCompactionReason.PROTECTED_OR_SEMANTIC_RISK


def test_unknown_target_requires_review() -> None:
    result = decide_cache_aware_compaction_timing(
        CacheAwareCompactionTimingInput(target=CacheAwareCompactionTarget.UNKNOWN)
    )
    assert result.decision is CacheAwareCompactionDecision.REQUIRE_MANUAL_REVIEW
    assert result.reason is CacheAwareCompactionReason.INSUFFICIENT_SIGNALS


def test_insufficient_hotness_signals_requires_review() -> None:
    result = decide_cache_aware_compaction_timing(
        CacheAwareCompactionTimingInput(
            target=CacheAwareCompactionTarget.STABLE_PREFIX,
            prefix_stability_status=PREFIX_STABILITY_STABLE,
            cache_hot=None,
            ttl_seconds_remaining=None,
        )
    )
    assert result.decision is CacheAwareCompactionDecision.REQUIRE_MANUAL_REVIEW
    assert result.reason is CacheAwareCompactionReason.INSUFFICIENT_SIGNALS


def test_synthetic_corpus_case_ids_are_unique() -> None:
    case_ids = [case.case_id for case in CACHE_AWARE_COMPACTION_CORPUS]
    assert len(case_ids) == len(set(case_ids))


def test_synthetic_corpus_has_required_cases() -> None:
    case_ids = {case.case_id for case in CACHE_AWARE_COMPACTION_CORPUS}
    assert _REQUIRED_CORPUS_CASE_IDS.issubset(case_ids)
    for case in CACHE_AWARE_COMPACTION_CORPUS:
        result = decide_cache_aware_compaction_timing(case.timing_input)
        assert result.decision is case.expected_decision
        assert result.reason is case.expected_reason
        assert result.target is case.timing_input.target


def test_synthetic_corpus_has_marker() -> None:
    assert (
        CACHE_AWARE_COMPACTION_SYNTHETIC_CORPUS_MARKER
        == "SYNTHETIC_CACHE_AWARE_COMPACTION_CORPUS_V1"
    )
    assert all(
        case.synthetic_marker == CACHE_AWARE_COMPACTION_SYNTHETIC_CORPUS_MARKER
        for case in CACHE_AWARE_COMPACTION_CORPUS
    )


def test_decision_reports_are_raw_content_safe() -> None:
    result = decide_cache_aware_compaction_timing(
        CacheAwareCompactionTimingInput(
            target=CacheAwareCompactionTarget.DYNAMIC_TAIL,
            dynamic_tail_reduction_available=True,
            estimated_content_reduction_chars=40,
            estimated_cache_invalidation_cost_tokens=10,
        )
    )
    field_names = {field.name for field in dataclasses.fields(result)}
    assert "raw_content_included" in field_names
    assert result.raw_content_included is False
    assert not _FORBIDDEN_SAVINGS_FIELD_NAMES.intersection(field_names)
    assert not any(
        name in {"content", "prompt", "raw_content", "thread", "payload"}
        for name in field_names
    )
    # Estimates are policy signals, not measured token savings fields.
    assert result.estimated_content_reduction_chars == 40
    assert result.estimated_cache_invalidation_cost_tokens == 10
