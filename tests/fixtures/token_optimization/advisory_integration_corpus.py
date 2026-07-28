# © Artur Czarnecki. All rights reserved.

"""Synthetic corpus for advisory policy-gated integration (TOKEN-7C)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.runtime.token_optimization.advisory_integration import (
    TokenOptimizationAdvisoryGateReason,
    TokenOptimizationAdvisoryIntegrationMode,
    TokenOptimizationAdvisoryIntegrationPolicy,
    TokenOptimizationAdvisoryIntegrationRequest,
    TokenOptimizationAdvisoryIntegrationStatus,
)
from intergrax.runtime.token_optimization.contracts import (
    TokenOptimizationAdvisorySignal,
    TokenOptimizationRecommendationAction,
    TokenOptimizationSourceType,
    TokenOptimizationStrategyKind,
)
from intergrax.runtime.token_optimization.prompt_cache import PREFIX_STABILITY_STABLE

ADVISORY_INTEGRATION_SYNTHETIC_CORPUS_MARKER = "SYNTHETIC_ADVISORY_INTEGRATION_CORPUS_V1"

_METADATA = {"synthetic_marker": ADVISORY_INTEGRATION_SYNTHETIC_CORPUS_MARKER}


@dataclass(frozen=True, slots=True)
class AdvisoryIntegrationCorpusCase:
    case_id: str
    title: str
    request: TokenOptimizationAdvisoryIntegrationRequest
    expected_status: TokenOptimizationAdvisoryIntegrationStatus
    expected_reason: TokenOptimizationAdvisoryGateReason
    expected_recommendation_action: TokenOptimizationRecommendationAction | None
    synthetic_marker: str = ADVISORY_INTEGRATION_SYNTHETIC_CORPUS_MARKER


_ENABLE_STRATEGY_SIGNAL = TokenOptimizationAdvisorySignal(
    source_type=TokenOptimizationSourceType.RAG_CONTEXT_PACK,
    strategy_kind=TokenOptimizationStrategyKind.DEDUPLICATION,
    sample_count=50,
    measured_saved_ratio=0.15,
    validation_pass_rate=0.98,
    fallback_rate=0.02,
    regression_gate_passed=True,
)

_DISABLE_STRATEGY_SIGNAL = TokenOptimizationAdvisorySignal(
    source_type=TokenOptimizationSourceType.TOOL_OUTPUT,
    strategy_kind=TokenOptimizationStrategyKind.EXTRACTIVE_FILTERING,
    sample_count=20,
    measured_saved_ratio=0.08,
    fallback_rate=0.30,
    regression_gate_passed=True,
)

ADVISORY_INTEGRATION_CORPUS: tuple[AdvisoryIntegrationCorpusCase, ...] = (
    AdvisoryIntegrationCorpusCase(
        case_id="advisory_integration.policy_disabled_blocks",
        title="Disabled policy blocks advisory integration",
        request=TokenOptimizationAdvisoryIntegrationRequest(
            signal=TokenOptimizationAdvisorySignal(
                source_type=TokenOptimizationSourceType.PROMPT,
                sample_count=10,
                measured_saved_ratio=0.12,
                regression_gate_passed=True,
            ),
            policy=TokenOptimizationAdvisoryIntegrationPolicy.disabled(),
            metadata=_METADATA,
        ),
        expected_status=TokenOptimizationAdvisoryIntegrationStatus.BLOCKED_BY_POLICY,
        expected_reason=TokenOptimizationAdvisoryGateReason.POLICY_DISABLED,
        expected_recommendation_action=None,
    ),
    AdvisoryIntegrationCorpusCase(
        case_id="advisory_integration.report_only_returns_report_only",
        title="Report-only mode returns report-only status",
        request=TokenOptimizationAdvisoryIntegrationRequest(
            signal=TokenOptimizationAdvisorySignal(
                source_type=TokenOptimizationSourceType.MEMORY,
                sample_count=25,
                measured_saved_ratio=0.01,
                regression_gate_passed=True,
            ),
            policy=TokenOptimizationAdvisoryIntegrationPolicy.report_only(),
            metadata=_METADATA,
        ),
        expected_status=TokenOptimizationAdvisoryIntegrationStatus.REPORT_ONLY,
        expected_reason=TokenOptimizationAdvisoryGateReason.REPORT_ONLY_MODE,
        expected_recommendation_action=TokenOptimizationRecommendationAction.KEEP_CURRENT,
    ),
    AdvisoryIntegrationCorpusCase(
        case_id="advisory_integration.dry_run_returns_dry_run",
        title="Dry-run mode returns dry-run status",
        request=TokenOptimizationAdvisoryIntegrationRequest(
            signal=TokenOptimizationAdvisorySignal(
                source_type=TokenOptimizationSourceType.MEMORY,
                sample_count=25,
                measured_saved_ratio=0.01,
                regression_gate_passed=True,
            ),
            policy=TokenOptimizationAdvisoryIntegrationPolicy.dry_run(),
            metadata=_METADATA,
        ),
        expected_status=TokenOptimizationAdvisoryIntegrationStatus.DRY_RUN,
        expected_reason=TokenOptimizationAdvisoryGateReason.DRY_RUN_MODE,
        expected_recommendation_action=TokenOptimizationRecommendationAction.KEEP_CURRENT,
    ),
    AdvisoryIntegrationCorpusCase(
        case_id="advisory_integration.review_only_requires_review",
        title="Review-only mode requires review",
        request=TokenOptimizationAdvisoryIntegrationRequest(
            signal=TokenOptimizationAdvisorySignal(
                source_type=TokenOptimizationSourceType.MEMORY,
                sample_count=25,
                measured_saved_ratio=0.01,
                regression_gate_passed=True,
            ),
            policy=TokenOptimizationAdvisoryIntegrationPolicy.review_only(),
            metadata=_METADATA,
        ),
        expected_status=TokenOptimizationAdvisoryIntegrationStatus.REVIEW_REQUIRED,
        expected_reason=TokenOptimizationAdvisoryGateReason.REVIEW_ONLY_MODE,
        expected_recommendation_action=TokenOptimizationRecommendationAction.KEEP_CURRENT,
    ),
    AdvisoryIntegrationCorpusCase(
        case_id="advisory_integration.advisory_allowed_returns_ready",
        title="Advisory-allowed mode returns recommendation-ready",
        request=TokenOptimizationAdvisoryIntegrationRequest(
            signal=TokenOptimizationAdvisorySignal(
                source_type=TokenOptimizationSourceType.MEMORY,
                sample_count=25,
                measured_saved_ratio=0.01,
                regression_gate_passed=True,
            ),
            policy=TokenOptimizationAdvisoryIntegrationPolicy(
                mode=TokenOptimizationAdvisoryIntegrationMode.ADVISORY_ALLOWED,
            ),
            metadata=_METADATA,
        ),
        expected_status=TokenOptimizationAdvisoryIntegrationStatus.RECOMMENDATION_READY,
        expected_reason=TokenOptimizationAdvisoryGateReason.ADVISORY_ALLOWED,
        expected_recommendation_action=TokenOptimizationRecommendationAction.KEEP_CURRENT,
    ),
    AdvisoryIntegrationCorpusCase(
        case_id="advisory_integration.insufficient_signals",
        title="Insufficient signals return insufficient status",
        request=TokenOptimizationAdvisoryIntegrationRequest(
            signal=TokenOptimizationAdvisorySignal(
                source_type=TokenOptimizationSourceType.PROMPT,
                sample_count=0,
            ),
            policy=TokenOptimizationAdvisoryIntegrationPolicy.dry_run(),
            metadata=_METADATA,
        ),
        expected_status=TokenOptimizationAdvisoryIntegrationStatus.INSUFFICIENT_SIGNALS,
        expected_reason=TokenOptimizationAdvisoryGateReason.INSUFFICIENT_SIGNALS,
        expected_recommendation_action=TokenOptimizationRecommendationAction.INSUFFICIENT_DATA,
    ),
    AdvisoryIntegrationCorpusCase(
        case_id="advisory_integration.strategy_enable_blocked_by_policy",
        title="Strategy enable blocked when policy disallows enable",
        request=TokenOptimizationAdvisoryIntegrationRequest(
            signal=_ENABLE_STRATEGY_SIGNAL,
            policy=TokenOptimizationAdvisoryIntegrationPolicy(
                mode=TokenOptimizationAdvisoryIntegrationMode.ADVISORY_ALLOWED,
                allow_strategy_enable=False,
            ),
            metadata=_METADATA,
        ),
        expected_status=TokenOptimizationAdvisoryIntegrationStatus.BLOCKED_BY_POLICY,
        expected_reason=TokenOptimizationAdvisoryGateReason.STRATEGY_ENABLE_NOT_ALLOWED,
        expected_recommendation_action=None,
    ),
    AdvisoryIntegrationCorpusCase(
        case_id="advisory_integration.strategy_enable_allowed",
        title="Strategy enable allowed when policy permits enable",
        request=TokenOptimizationAdvisoryIntegrationRequest(
            signal=_ENABLE_STRATEGY_SIGNAL,
            policy=TokenOptimizationAdvisoryIntegrationPolicy(
                mode=TokenOptimizationAdvisoryIntegrationMode.ADVISORY_ALLOWED,
                allow_strategy_enable=True,
            ),
            metadata=_METADATA,
        ),
        expected_status=TokenOptimizationAdvisoryIntegrationStatus.RECOMMENDATION_READY,
        expected_reason=TokenOptimizationAdvisoryGateReason.ADVISORY_ALLOWED,
        expected_recommendation_action=TokenOptimizationRecommendationAction.ENABLE_STRATEGY,
    ),
    AdvisoryIntegrationCorpusCase(
        case_id="advisory_integration.strategy_disable_blocked_by_policy",
        title="Strategy disable blocked when policy disallows disable",
        request=TokenOptimizationAdvisoryIntegrationRequest(
            signal=_DISABLE_STRATEGY_SIGNAL,
            policy=TokenOptimizationAdvisoryIntegrationPolicy(
                mode=TokenOptimizationAdvisoryIntegrationMode.ADVISORY_ALLOWED,
                allow_strategy_disable=False,
                require_review_for_risky_recommendations=False,
            ),
            metadata=_METADATA,
        ),
        expected_status=TokenOptimizationAdvisoryIntegrationStatus.BLOCKED_BY_POLICY,
        expected_reason=TokenOptimizationAdvisoryGateReason.STRATEGY_DISABLE_NOT_ALLOWED,
        expected_recommendation_action=None,
    ),
    AdvisoryIntegrationCorpusCase(
        case_id="advisory_integration.risky_disable_requires_review",
        title="Risky disable recommendation requires review",
        request=TokenOptimizationAdvisoryIntegrationRequest(
            signal=_DISABLE_STRATEGY_SIGNAL,
            policy=TokenOptimizationAdvisoryIntegrationPolicy.dry_run(),
            metadata=_METADATA,
        ),
        expected_status=TokenOptimizationAdvisoryIntegrationStatus.REVIEW_REQUIRED,
        expected_reason=TokenOptimizationAdvisoryGateReason.RISK_REQUIRES_REVIEW,
        expected_recommendation_action=TokenOptimizationRecommendationAction.DISABLE_STRATEGY,
    ),
    AdvisoryIntegrationCorpusCase(
        case_id="advisory_integration.hot_stable_cache_report_only",
        title="Hot stable cache recommendation in report-only mode",
        request=TokenOptimizationAdvisoryIntegrationRequest(
            signal=TokenOptimizationAdvisorySignal(
                source_type=TokenOptimizationSourceType.SYSTEM_POLICY,
                sample_count=15,
                measured_saved_ratio=0.06,
                fallback_rate=0.08,
                regression_gate_passed=True,
                cache_hot=True,
                cache_prefix_stability_status=PREFIX_STABILITY_STABLE,
            ),
            policy=TokenOptimizationAdvisoryIntegrationPolicy.report_only(),
            metadata=_METADATA,
        ),
        expected_status=TokenOptimizationAdvisoryIntegrationStatus.REPORT_ONLY,
        expected_reason=TokenOptimizationAdvisoryGateReason.REPORT_ONLY_MODE,
        expected_recommendation_action=(
            TokenOptimizationRecommendationAction.PRESERVE_CACHEABLE_PREFIX
        ),
    ),
    AdvisoryIntegrationCorpusCase(
        case_id="advisory_integration.dynamic_tail_dry_run",
        title="Dynamic tail recommendation in dry-run mode",
        request=TokenOptimizationAdvisoryIntegrationRequest(
            signal=TokenOptimizationAdvisorySignal(
                source_type=TokenOptimizationSourceType.CONVERSATION_HISTORY,
                sample_count=10,
                measured_saved_ratio=0.05,
                validation_pass_rate=0.90,
                fallback_rate=0.10,
                regression_gate_passed=True,
                dynamic_tail_reduction_available=True,
            ),
            policy=TokenOptimizationAdvisoryIntegrationPolicy.dry_run(),
            metadata=_METADATA,
        ),
        expected_status=TokenOptimizationAdvisoryIntegrationStatus.DRY_RUN,
        expected_reason=TokenOptimizationAdvisoryGateReason.DRY_RUN_MODE,
        expected_recommendation_action=(
            TokenOptimizationRecommendationAction.PREFER_DYNAMIC_TAIL_REDUCTION
        ),
    ),
)
