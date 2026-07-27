# © Artur Czarnecki. All rights reserved.

"""Synthetic corpus for cache-aware compaction timing policy (TOKEN-OPT-5E)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.runtime.token_optimization.contracts import (
    CacheAwareCompactionDecision,
    CacheAwareCompactionReason,
    CacheAwareCompactionTarget,
    CacheAwareCompactionTimingInput,
    PromptCacheInvalidationReason,
)
from intergrax.runtime.token_optimization.prompt_cache import (
    PREFIX_STABILITY_INVALIDATED,
    PREFIX_STABILITY_STABLE,
)

CACHE_AWARE_COMPACTION_SYNTHETIC_CORPUS_MARKER = (
    "SYNTHETIC_CACHE_AWARE_COMPACTION_CORPUS_V1"
)


@dataclass(frozen=True, slots=True)
class CacheAwareCompactionCorpusCase:
    """One synthetic cache-aware compaction timing case (no raw prompts)."""

    case_id: str
    title: str
    timing_input: CacheAwareCompactionTimingInput
    expected_decision: CacheAwareCompactionDecision
    expected_reason: CacheAwareCompactionReason
    synthetic_marker: str = CACHE_AWARE_COMPACTION_SYNTHETIC_CORPUS_MARKER


CACHE_AWARE_COMPACTION_CORPUS: tuple[CacheAwareCompactionCorpusCase, ...] = (
    CacheAwareCompactionCorpusCase(
        case_id="cache_aware_compaction.dynamic_tail_safe_to_reduce",
        title="Dynamic tail reduction runs without rewriting stable prefix",
        timing_input=CacheAwareCompactionTimingInput(
            target=CacheAwareCompactionTarget.DYNAMIC_TAIL,
            prefix_stability_status=PREFIX_STABILITY_STABLE,
            cache_hot=True,
            ttl_seconds_remaining=600,
            estimated_content_reduction_chars=400,
            estimated_cache_invalidation_cost_tokens=50,
            dynamic_tail_reduction_available=True,
        ),
        expected_decision=CacheAwareCompactionDecision.RUN,
        expected_reason=CacheAwareCompactionReason.DYNAMIC_TAIL_SAFE_TO_REDUCE,
    ),
    CacheAwareCompactionCorpusCase(
        case_id="cache_aware_compaction.cold_history_safe_to_compact",
        title="Cold history compaction runs when cache is not hot",
        timing_input=CacheAwareCompactionTimingInput(
            target=CacheAwareCompactionTarget.COLD_HISTORY,
            prefix_stability_status=PREFIX_STABILITY_STABLE,
            cache_hot=False,
            ttl_seconds_remaining=0,
            estimated_content_reduction_chars=800,
            estimated_cache_invalidation_cost_tokens=10,
        ),
        expected_decision=CacheAwareCompactionDecision.RUN,
        expected_reason=CacheAwareCompactionReason.COLD_HISTORY_SAFE_TO_COMPACT,
    ),
    CacheAwareCompactionCorpusCase(
        case_id="cache_aware_compaction.hot_stable_prefix_deferred",
        title="Hot stable prefix defers when invalidation cost exceeds benefit",
        timing_input=CacheAwareCompactionTimingInput(
            target=CacheAwareCompactionTarget.STABLE_PREFIX,
            prefix_stability_status=PREFIX_STABILITY_STABLE,
            cache_hot=True,
            ttl_seconds_remaining=900,
            estimated_content_reduction_chars=100,
            estimated_cache_invalidation_cost_tokens=500,
        ),
        expected_decision=CacheAwareCompactionDecision.DEFER,
        expected_reason=CacheAwareCompactionReason.CACHE_INVALIDATION_COST_TOO_HIGH,
    ),
    CacheAwareCompactionCorpusCase(
        case_id="cache_aware_compaction.near_expiry_stable_prefix_runs",
        title="Near-expiry stable prefix compaction may run",
        timing_input=CacheAwareCompactionTimingInput(
            target=CacheAwareCompactionTarget.STABLE_PREFIX,
            prefix_stability_status=PREFIX_STABILITY_STABLE,
            cache_hot=True,
            ttl_seconds_remaining=30,
            near_expiry_threshold_seconds=60,
            estimated_content_reduction_chars=200,
            estimated_cache_invalidation_cost_tokens=400,
        ),
        expected_decision=CacheAwareCompactionDecision.RUN,
        expected_reason=CacheAwareCompactionReason.CACHE_NEAR_EXPIRY,
    ),
    CacheAwareCompactionCorpusCase(
        case_id="cache_aware_compaction.prefix_not_stable_defers",
        title="Unstable prefix defers stable-prefix compaction",
        timing_input=CacheAwareCompactionTimingInput(
            target=CacheAwareCompactionTarget.STABLE_PREFIX,
            prefix_stability_status=PREFIX_STABILITY_INVALIDATED,
            invalidation_reason=PromptCacheInvalidationReason.PREFIX_CHANGED,
            cache_hot=True,
            ttl_seconds_remaining=500,
            estimated_content_reduction_chars=300,
            estimated_cache_invalidation_cost_tokens=50,
        ),
        expected_decision=CacheAwareCompactionDecision.DEFER,
        expected_reason=CacheAwareCompactionReason.PREFIX_NOT_STABLE,
    ),
    CacheAwareCompactionCorpusCase(
        case_id="cache_aware_compaction.low_benefit_bypasses",
        title="Zero content-reduction benefit bypasses compaction",
        timing_input=CacheAwareCompactionTimingInput(
            target=CacheAwareCompactionTarget.DYNAMIC_TAIL,
            prefix_stability_status=PREFIX_STABILITY_STABLE,
            cache_hot=False,
            ttl_seconds_remaining=120,
            estimated_content_reduction_chars=0,
            estimated_cache_invalidation_cost_tokens=0,
            dynamic_tail_reduction_available=False,
        ),
        expected_decision=CacheAwareCompactionDecision.BYPASS,
        expected_reason=CacheAwareCompactionReason.LOW_CONTENT_REDUCTION_BENEFIT,
    ),
    CacheAwareCompactionCorpusCase(
        case_id="cache_aware_compaction.full_thread_rewrite_requires_review",
        title="Full-thread rewrite requires manual review by default",
        timing_input=CacheAwareCompactionTimingInput(
            target=CacheAwareCompactionTarget.FULL_THREAD,
            prefix_stability_status=PREFIX_STABILITY_STABLE,
            cache_hot=False,
            ttl_seconds_remaining=300,
            estimated_content_reduction_chars=500,
            estimated_cache_invalidation_cost_tokens=100,
        ),
        expected_decision=CacheAwareCompactionDecision.REQUIRE_MANUAL_REVIEW,
        expected_reason=CacheAwareCompactionReason.FULL_THREAD_REWRITE_RISK,
    ),
    CacheAwareCompactionCorpusCase(
        case_id="cache_aware_compaction.protected_or_semantic_risk_requires_review",
        title="Protected or semantic risk requires manual review",
        timing_input=CacheAwareCompactionTimingInput(
            target=CacheAwareCompactionTarget.DYNAMIC_TAIL,
            prefix_stability_status=PREFIX_STABILITY_STABLE,
            cache_hot=False,
            ttl_seconds_remaining=200,
            estimated_content_reduction_chars=250,
            protected_or_semantic_risk=True,
            dynamic_tail_reduction_available=True,
        ),
        expected_decision=CacheAwareCompactionDecision.REQUIRE_MANUAL_REVIEW,
        expected_reason=CacheAwareCompactionReason.PROTECTED_OR_SEMANTIC_RISK,
    ),
    CacheAwareCompactionCorpusCase(
        case_id="cache_aware_compaction.unknown_target_requires_review",
        title="Unknown compaction target requires manual review",
        timing_input=CacheAwareCompactionTimingInput(
            target=CacheAwareCompactionTarget.UNKNOWN,
            prefix_stability_status=PREFIX_STABILITY_STABLE,
            cache_hot=True,
            ttl_seconds_remaining=400,
            estimated_content_reduction_chars=150,
        ),
        expected_decision=CacheAwareCompactionDecision.REQUIRE_MANUAL_REVIEW,
        expected_reason=CacheAwareCompactionReason.INSUFFICIENT_SIGNALS,
    ),
    CacheAwareCompactionCorpusCase(
        case_id="cache_aware_compaction.insufficient_hotness_signals_requires_review",
        title="Stable-prefix compaction without hotness/TTL signals requires review",
        timing_input=CacheAwareCompactionTimingInput(
            target=CacheAwareCompactionTarget.STABLE_PREFIX,
            prefix_stability_status=PREFIX_STABILITY_STABLE,
            cache_hot=None,
            ttl_seconds_remaining=None,
            estimated_content_reduction_chars=200,
            estimated_cache_invalidation_cost_tokens=50,
        ),
        expected_decision=CacheAwareCompactionDecision.REQUIRE_MANUAL_REVIEW,
        expected_reason=CacheAwareCompactionReason.INSUFFICIENT_SIGNALS,
    ),
)
