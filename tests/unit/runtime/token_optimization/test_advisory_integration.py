# © Artur Czarnecki. All rights reserved.

"""TOKEN-7C: policy-gated advisory integration tests."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from typing import Any
from unittest.mock import patch

import pytest

from intergrax.runtime.token_optimization.advisory_integration import (
    TokenOptimizationAdvisoryGateReason,
    TokenOptimizationAdvisoryIntegrationMode,
    TokenOptimizationAdvisoryIntegrationPolicy,
    TokenOptimizationAdvisoryIntegrationRequest,
    TokenOptimizationAdvisoryIntegrationResult,
    TokenOptimizationAdvisoryIntegrationStatus,
    evaluate_policy_gated_advisory_request,
    format_token_optimization_advisory_integration_result,
    token_optimization_advisory_integration_result_to_dict,
)
from intergrax.runtime.token_optimization.contracts import (
    TokenOptimizationAdvisoryRecommendation,
    TokenOptimizationAdvisorySignal,
    TokenOptimizationRecommendationAction,
    TokenOptimizationRecommendationConfidence,
    TokenOptimizationRecommendationReason,
    TokenOptimizationSourceType,
    TokenOptimizationStrategyKind,
)
from tests.fixtures.token_optimization.advisory_integration_corpus import (
    ADVISORY_INTEGRATION_CORPUS,
    ADVISORY_INTEGRATION_SYNTHETIC_CORPUS_MARKER,
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
        "source_text",
        "recommendation",
        "signal",
    }
)

_REQUIRED_CORPUS_CASE_IDS = frozenset(
    {
        "advisory_integration.policy_disabled_blocks",
        "advisory_integration.report_only_returns_report_only",
        "advisory_integration.dry_run_returns_dry_run",
        "advisory_integration.review_only_requires_review",
        "advisory_integration.advisory_allowed_returns_ready",
        "advisory_integration.insufficient_signals",
        "advisory_integration.strategy_enable_blocked_by_policy",
        "advisory_integration.strategy_enable_allowed",
        "advisory_integration.strategy_disable_blocked_by_policy",
        "advisory_integration.risky_disable_requires_review",
        "advisory_integration.hot_stable_cache_report_only",
        "advisory_integration.dynamic_tail_dry_run",
    }
)


def _enable_strategy_signal() -> TokenOptimizationAdvisorySignal:
    return TokenOptimizationAdvisorySignal(
        source_type=TokenOptimizationSourceType.RAG_CONTEXT_PACK,
        strategy_kind=TokenOptimizationStrategyKind.DEDUPLICATION,
        sample_count=50,
        measured_saved_ratio=0.15,
        validation_pass_rate=0.98,
        fallback_rate=0.02,
        regression_gate_passed=True,
    )


def _disable_strategy_signal() -> TokenOptimizationAdvisorySignal:
    return TokenOptimizationAdvisorySignal(
        source_type=TokenOptimizationSourceType.TOOL_OUTPUT,
        strategy_kind=TokenOptimizationStrategyKind.EXTRACTIVE_FILTERING,
        sample_count=20,
        measured_saved_ratio=0.08,
        fallback_rate=0.30,
        regression_gate_passed=True,
    )


def _collect_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, Mapping):
        for key, nested in value.items():
            keys.add(str(key))
            keys.update(_collect_keys(nested))
    elif isinstance(value, (list, tuple)):
        for item in value:
            keys.update(_collect_keys(item))
    return keys


def test_integration_mode_values_are_stable() -> None:
    assert [mode.value for mode in TokenOptimizationAdvisoryIntegrationMode] == [
        "disabled",
        "report_only",
        "dry_run",
        "review_only",
        "advisory_allowed",
    ]


def test_integration_status_values_are_stable() -> None:
    assert [status.value for status in TokenOptimizationAdvisoryIntegrationStatus] == [
        "blocked_by_policy",
        "report_only",
        "dry_run",
        "review_required",
        "recommendation_ready",
        "insufficient_signals",
    ]


def test_gate_reason_values_are_stable() -> None:
    assert [reason.value for reason in TokenOptimizationAdvisoryGateReason] == [
        "policy_disabled",
        "mode_disabled",
        "report_only_mode",
        "dry_run_mode",
        "review_only_mode",
        "advisory_allowed",
        "insufficient_signals",
        "strategy_enable_not_allowed",
        "strategy_disable_not_allowed",
        "risk_requires_review",
        "manual_review_recommended",
        "auto_apply_forbidden",
    ]


def test_policy_rejects_auto_apply() -> None:
    with pytest.raises(ValueError, match="allow_auto_apply must remain False"):
        TokenOptimizationAdvisoryIntegrationPolicy(allow_auto_apply=True)


def test_policy_requires_disabled_mode_when_disabled() -> None:
    with pytest.raises(ValueError, match="when enabled is False, mode must be DISABLED"):
        TokenOptimizationAdvisoryIntegrationPolicy(
            enabled=False,
            mode=TokenOptimizationAdvisoryIntegrationMode.DRY_RUN,
        )


def test_policy_rejects_disabled_mode_when_enabled() -> None:
    with pytest.raises(ValueError, match="when enabled is True, mode must not be DISABLED"):
        TokenOptimizationAdvisoryIntegrationPolicy(
            enabled=True,
            mode=TokenOptimizationAdvisoryIntegrationMode.DISABLED,
        )


def test_request_validates_request_id() -> None:
    with pytest.raises(ValueError, match="request_id must be non-empty"):
        TokenOptimizationAdvisoryIntegrationRequest(
            signal=TokenOptimizationAdvisorySignal(
                source_type=TokenOptimizationSourceType.PROMPT,
            ),
            request_id="   ",
        )


def test_request_rejects_raw_content_metadata_keys() -> None:
    for key in _FORBIDDEN_CONTENT_FIELD_NAMES:
        with pytest.raises(ValueError, match="metadata must not contain raw-content-like key"):
            TokenOptimizationAdvisoryIntegrationRequest(
                signal=TokenOptimizationAdvisorySignal(
                    source_type=TokenOptimizationSourceType.PROMPT,
                ),
                metadata={key: "value"},
            )


def test_result_rejects_auto_apply() -> None:
    with pytest.raises(ValueError, match="auto_apply_allowed must remain False"):
        TokenOptimizationAdvisoryIntegrationResult(
            status=TokenOptimizationAdvisoryIntegrationStatus.DRY_RUN,
            reason=TokenOptimizationAdvisoryGateReason.DRY_RUN_MODE,
            source_type=TokenOptimizationSourceType.PROMPT,
            auto_apply_allowed=True,
        )


def test_result_rejects_raw_content_flag() -> None:
    with pytest.raises(ValueError, match="raw_content_included must remain False"):
        TokenOptimizationAdvisoryIntegrationResult(
            status=TokenOptimizationAdvisoryIntegrationStatus.DRY_RUN,
            reason=TokenOptimizationAdvisoryGateReason.DRY_RUN_MODE,
            source_type=TokenOptimizationSourceType.PROMPT,
            raw_content_included=True,
        )


def test_blocked_result_cannot_include_recommendation() -> None:
    recommendation = TokenOptimizationAdvisoryRecommendation(
        action=TokenOptimizationRecommendationAction.KEEP_CURRENT,
        reason=TokenOptimizationRecommendationReason.REGRESSION_GATE_PASSED,
        confidence=TokenOptimizationRecommendationConfidence.MEDIUM,
        source_type=TokenOptimizationSourceType.PROMPT,
    )
    with pytest.raises(ValueError, match="blocked results must not include a recommendation"):
        TokenOptimizationAdvisoryIntegrationResult(
            status=TokenOptimizationAdvisoryIntegrationStatus.BLOCKED_BY_POLICY,
            reason=TokenOptimizationAdvisoryGateReason.POLICY_DISABLED,
            source_type=TokenOptimizationSourceType.PROMPT,
            recommendation=recommendation,
        )


def test_policy_disabled_blocks_without_recommendation() -> None:
    request = TokenOptimizationAdvisoryIntegrationRequest(
        signal=_enable_strategy_signal(),
        policy=TokenOptimizationAdvisoryIntegrationPolicy.disabled(),
    )
    result = evaluate_policy_gated_advisory_request(request)
    assert result.status is TokenOptimizationAdvisoryIntegrationStatus.BLOCKED_BY_POLICY
    assert result.reason is TokenOptimizationAdvisoryGateReason.POLICY_DISABLED
    assert result.recommendation is None


def test_policy_disabled_does_not_call_recommender() -> None:
    request = TokenOptimizationAdvisoryIntegrationRequest(
        signal=_enable_strategy_signal(),
        policy=TokenOptimizationAdvisoryIntegrationPolicy.disabled(),
    )
    with patch(
        "intergrax.runtime.token_optimization.advisory_integration.recommend_token_optimization_action"
    ) as mock_recommend:
        result = evaluate_policy_gated_advisory_request(request)
    mock_recommend.assert_not_called()
    assert result.recommendation is None


def test_mode_disabled_blocks_without_recommendation() -> None:
    policy = TokenOptimizationAdvisoryIntegrationPolicy.dry_run()
    object.__setattr__(policy, "mode", TokenOptimizationAdvisoryIntegrationMode.DISABLED)
    request = TokenOptimizationAdvisoryIntegrationRequest(
        signal=_enable_strategy_signal(),
        policy=policy,
    )
    result = evaluate_policy_gated_advisory_request(request)
    assert result.status is TokenOptimizationAdvisoryIntegrationStatus.BLOCKED_BY_POLICY
    assert result.reason is TokenOptimizationAdvisoryGateReason.MODE_DISABLED
    assert result.recommendation is None


def test_insufficient_signals_returns_insufficient_status() -> None:
    request = TokenOptimizationAdvisoryIntegrationRequest(
        signal=TokenOptimizationAdvisorySignal(
            source_type=TokenOptimizationSourceType.PROMPT,
            sample_count=0,
        ),
        policy=TokenOptimizationAdvisoryIntegrationPolicy.dry_run(),
    )
    result = evaluate_policy_gated_advisory_request(request)
    assert result.status is TokenOptimizationAdvisoryIntegrationStatus.INSUFFICIENT_SIGNALS
    assert result.reason is TokenOptimizationAdvisoryGateReason.INSUFFICIENT_SIGNALS
    assert result.recommendation is not None
    assert (
        result.recommendation.action
        is TokenOptimizationRecommendationAction.INSUFFICIENT_DATA
    )


def test_strategy_enable_blocked_when_not_allowed() -> None:
    request = TokenOptimizationAdvisoryIntegrationRequest(
        signal=_enable_strategy_signal(),
        policy=TokenOptimizationAdvisoryIntegrationPolicy(
            mode=TokenOptimizationAdvisoryIntegrationMode.ADVISORY_ALLOWED,
            allow_strategy_enable=False,
        ),
    )
    result = evaluate_policy_gated_advisory_request(request)
    assert result.status is TokenOptimizationAdvisoryIntegrationStatus.BLOCKED_BY_POLICY
    assert result.reason is TokenOptimizationAdvisoryGateReason.STRATEGY_ENABLE_NOT_ALLOWED
    assert result.recommendation is None


def test_strategy_enable_allowed_when_policy_allows() -> None:
    request = TokenOptimizationAdvisoryIntegrationRequest(
        signal=_enable_strategy_signal(),
        policy=TokenOptimizationAdvisoryIntegrationPolicy(
            mode=TokenOptimizationAdvisoryIntegrationMode.ADVISORY_ALLOWED,
            allow_strategy_enable=True,
        ),
    )
    result = evaluate_policy_gated_advisory_request(request)
    assert result.status is TokenOptimizationAdvisoryIntegrationStatus.RECOMMENDATION_READY
    assert result.reason is TokenOptimizationAdvisoryGateReason.ADVISORY_ALLOWED
    assert (
        result.recommendation is not None
        and result.recommendation.action is TokenOptimizationRecommendationAction.ENABLE_STRATEGY
    )


def test_strategy_disable_blocked_when_not_allowed() -> None:
    request = TokenOptimizationAdvisoryIntegrationRequest(
        signal=_disable_strategy_signal(),
        policy=TokenOptimizationAdvisoryIntegrationPolicy(
            mode=TokenOptimizationAdvisoryIntegrationMode.ADVISORY_ALLOWED,
            allow_strategy_disable=False,
            require_review_for_risky_recommendations=False,
        ),
    )
    result = evaluate_policy_gated_advisory_request(request)
    assert result.status is TokenOptimizationAdvisoryIntegrationStatus.BLOCKED_BY_POLICY
    assert result.reason is TokenOptimizationAdvisoryGateReason.STRATEGY_DISABLE_NOT_ALLOWED
    assert result.recommendation is None


def test_risky_recommendations_require_review() -> None:
    request = TokenOptimizationAdvisoryIntegrationRequest(
        signal=_disable_strategy_signal(),
        policy=TokenOptimizationAdvisoryIntegrationPolicy.dry_run(),
    )
    result = evaluate_policy_gated_advisory_request(request)
    assert result.status is TokenOptimizationAdvisoryIntegrationStatus.REVIEW_REQUIRED
    assert result.reason is TokenOptimizationAdvisoryGateReason.RISK_REQUIRES_REVIEW
    assert (
        result.recommendation is not None
        and result.recommendation.action is TokenOptimizationRecommendationAction.DISABLE_STRATEGY
    )


def test_report_only_mode_returns_report_only() -> None:
    request = TokenOptimizationAdvisoryIntegrationRequest(
        signal=TokenOptimizationAdvisorySignal(
            source_type=TokenOptimizationSourceType.MEMORY,
            sample_count=25,
            measured_saved_ratio=0.01,
            regression_gate_passed=True,
        ),
        policy=TokenOptimizationAdvisoryIntegrationPolicy.report_only(),
    )
    result = evaluate_policy_gated_advisory_request(request)
    assert result.status is TokenOptimizationAdvisoryIntegrationStatus.REPORT_ONLY
    assert result.reason is TokenOptimizationAdvisoryGateReason.REPORT_ONLY_MODE


def test_dry_run_mode_returns_dry_run() -> None:
    request = TokenOptimizationAdvisoryIntegrationRequest(
        signal=TokenOptimizationAdvisorySignal(
            source_type=TokenOptimizationSourceType.MEMORY,
            sample_count=25,
            measured_saved_ratio=0.01,
            regression_gate_passed=True,
        ),
        policy=TokenOptimizationAdvisoryIntegrationPolicy.dry_run(),
    )
    result = evaluate_policy_gated_advisory_request(request)
    assert result.status is TokenOptimizationAdvisoryIntegrationStatus.DRY_RUN
    assert result.reason is TokenOptimizationAdvisoryGateReason.DRY_RUN_MODE


def test_review_only_mode_requires_review() -> None:
    request = TokenOptimizationAdvisoryIntegrationRequest(
        signal=TokenOptimizationAdvisorySignal(
            source_type=TokenOptimizationSourceType.MEMORY,
            sample_count=25,
            measured_saved_ratio=0.01,
            regression_gate_passed=True,
        ),
        policy=TokenOptimizationAdvisoryIntegrationPolicy.review_only(),
    )
    result = evaluate_policy_gated_advisory_request(request)
    assert result.status is TokenOptimizationAdvisoryIntegrationStatus.REVIEW_REQUIRED
    assert result.reason is TokenOptimizationAdvisoryGateReason.REVIEW_ONLY_MODE


def test_advisory_allowed_mode_returns_recommendation_ready() -> None:
    request = TokenOptimizationAdvisoryIntegrationRequest(
        signal=TokenOptimizationAdvisorySignal(
            source_type=TokenOptimizationSourceType.MEMORY,
            sample_count=25,
            measured_saved_ratio=0.01,
            regression_gate_passed=True,
        ),
        policy=TokenOptimizationAdvisoryIntegrationPolicy(
            mode=TokenOptimizationAdvisoryIntegrationMode.ADVISORY_ALLOWED,
        ),
    )
    result = evaluate_policy_gated_advisory_request(request)
    assert result.status is TokenOptimizationAdvisoryIntegrationStatus.RECOMMENDATION_READY
    assert result.reason is TokenOptimizationAdvisoryGateReason.ADVISORY_ALLOWED


def test_result_to_dict_is_redaction_safe() -> None:
    request = TokenOptimizationAdvisoryIntegrationRequest(
        signal=_enable_strategy_signal(),
        policy=TokenOptimizationAdvisoryIntegrationPolicy(
            mode=TokenOptimizationAdvisoryIntegrationMode.ADVISORY_ALLOWED,
            allow_strategy_enable=True,
        ),
        request_id="req-1",
    )
    result = evaluate_policy_gated_advisory_request(request)
    payload = token_optimization_advisory_integration_result_to_dict(result)
    keys = _collect_keys(payload)
    assert keys.isdisjoint(_FORBIDDEN_CONTENT_FIELD_NAMES)
    assert keys.isdisjoint(_FORBIDDEN_SAVINGS_FIELD_NAMES)
    assert "recommendation" not in payload
    assert payload["auto_apply_allowed"] is False
    assert payload["raw_content_included"] is False
    json.dumps(payload)


def test_format_integration_result_is_deterministic() -> None:
    request = TokenOptimizationAdvisoryIntegrationRequest(
        signal=TokenOptimizationAdvisorySignal(
            source_type=TokenOptimizationSourceType.MEMORY,
            sample_count=25,
            measured_saved_ratio=0.01,
            regression_gate_passed=True,
        ),
        policy=TokenOptimizationAdvisoryIntegrationPolicy.dry_run(),
        request_id="req-1",
    )
    result = evaluate_policy_gated_advisory_request(request)
    first = format_token_optimization_advisory_integration_result(result)
    second = format_token_optimization_advisory_integration_result(result)
    assert first == second


def test_format_integration_result_is_redaction_safe() -> None:
    request = TokenOptimizationAdvisoryIntegrationRequest(
        signal=_enable_strategy_signal(),
        policy=TokenOptimizationAdvisoryIntegrationPolicy(
            mode=TokenOptimizationAdvisoryIntegrationMode.ADVISORY_ALLOWED,
            allow_strategy_enable=True,
        ),
        request_id="req-1",
    )
    result = evaluate_policy_gated_advisory_request(request)
    text = format_token_optimization_advisory_integration_result(result)
    lowered = text.lower()
    for field_name in _FORBIDDEN_CONTENT_FIELD_NAMES:
        assert f"{field_name}=" not in lowered
    for field_name in _FORBIDDEN_SAVINGS_FIELD_NAMES:
        assert field_name not in lowered
    assert "TokenOptimizationAdvisoryRecommendation" not in text
    assert "TokenOptimizationAdvisorySignal" not in text


def test_synthetic_integration_corpus_case_ids_are_unique() -> None:
    case_ids = [case.case_id for case in ADVISORY_INTEGRATION_CORPUS]
    assert len(case_ids) == len(set(case_ids))


def test_synthetic_integration_corpus_has_required_cases() -> None:
    case_ids = {case.case_id for case in ADVISORY_INTEGRATION_CORPUS}
    assert _REQUIRED_CORPUS_CASE_IDS.issubset(case_ids)


def test_synthetic_integration_corpus_has_marker() -> None:
    for case in ADVISORY_INTEGRATION_CORPUS:
        assert case.synthetic_marker == ADVISORY_INTEGRATION_SYNTHETIC_CORPUS_MARKER
        assert (
            case.request.metadata.get("synthetic_marker")
            == ADVISORY_INTEGRATION_SYNTHETIC_CORPUS_MARKER
        )


def test_synthetic_integration_corpus_all_cases_match_expected_status() -> None:
    for case in ADVISORY_INTEGRATION_CORPUS:
        result = evaluate_policy_gated_advisory_request(case.request)
        assert result.status is case.expected_status, case.case_id
        assert result.reason is case.expected_reason, case.case_id
        if case.expected_recommendation_action is None:
            assert result.recommendation is None, case.case_id
        else:
            assert result.recommendation is not None, case.case_id
            assert (
                result.recommendation.action is case.expected_recommendation_action
            ), case.case_id


def test_integration_results_never_allow_auto_apply() -> None:
    for case in ADVISORY_INTEGRATION_CORPUS:
        result = evaluate_policy_gated_advisory_request(case.request)
        assert not result.auto_apply_allowed
        if result.recommendation is not None:
            assert not result.recommendation.auto_apply_allowed


def test_integration_results_never_include_raw_content() -> None:
    for case in ADVISORY_INTEGRATION_CORPUS:
        result = evaluate_policy_gated_advisory_request(case.request)
        assert not result.raw_content_included
        if result.recommendation is not None:
            assert not result.recommendation.raw_content_included
        payload = token_optimization_advisory_integration_result_to_dict(result)
        text = format_token_optimization_advisory_integration_result(result)
        assert _collect_keys(payload).isdisjoint(_FORBIDDEN_CONTENT_FIELD_NAMES)
        assert _collect_keys(payload).isdisjoint(_FORBIDDEN_SAVINGS_FIELD_NAMES)
        for field_name in _FORBIDDEN_CONTENT_FIELD_NAMES:
            assert f"{field_name}=" not in text.lower()


def test_same_request_produces_same_result() -> None:
    request = TokenOptimizationAdvisoryIntegrationRequest(
        signal=_enable_strategy_signal(),
        policy=TokenOptimizationAdvisoryIntegrationPolicy.dry_run(),
        request_id="req-stable",
    )
    first = evaluate_policy_gated_advisory_request(request)
    second = evaluate_policy_gated_advisory_request(request)
    assert dataclasses.asdict(first) == dataclasses.asdict(second)
