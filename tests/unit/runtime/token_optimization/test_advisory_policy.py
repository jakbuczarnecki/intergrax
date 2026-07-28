# © Artur Czarnecki. All rights reserved.

"""TOKEN-7D: advisory policy presets and resolver tests."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import pytest

from intergrax.runtime.token_optimization.advisory_integration import (
    TokenOptimizationAdvisoryGateReason,
    TokenOptimizationAdvisoryIntegrationMode,
    TokenOptimizationAdvisoryIntegrationPolicy,
    TokenOptimizationAdvisoryIntegrationRequest,
    TokenOptimizationAdvisoryIntegrationStatus,
    evaluate_policy_gated_advisory_request,
)
from intergrax.runtime.token_optimization.advisory_policy import (
    TokenOptimizationAdvisoryPolicyOverrides,
    TokenOptimizationAdvisoryPolicyPreset,
    TokenOptimizationAdvisoryPolicyResolution,
    format_token_optimization_advisory_policy_resolution,
    resolve_token_optimization_advisory_policy,
    token_optimization_advisory_policy_resolution_to_dict,
)
from intergrax.runtime.token_optimization.contracts import (
    TokenOptimizationAdvisorySignal,
    TokenOptimizationRecommendationAction,
    TokenOptimizationSourceType,
    TokenOptimizationStrategyKind,
)
from tests.fixtures.token_optimization.advisory_policy_corpus import (
    ADVISORY_POLICY_CORPUS,
    ADVISORY_POLICY_SYNTHETIC_CORPUS_MARKER,
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
        "advisory_policy.disabled",
        "advisory_policy.report_only",
        "advisory_policy.dry_run_safe",
        "advisory_policy.review_first",
        "advisory_policy.advisory_allowed_safe",
        "advisory_policy.override_allows_strategy_enable",
        "advisory_policy.override_blocks_strategy_disable",
        "advisory_policy.override_disables_risky_review",
    }
)


def _keep_current_signal() -> TokenOptimizationAdvisorySignal:
    return TokenOptimizationAdvisorySignal(
        source_type=TokenOptimizationSourceType.MEMORY,
        sample_count=25,
        measured_saved_ratio=0.01,
        regression_gate_passed=True,
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


def test_policy_preset_values_are_stable() -> None:
    assert [preset.value for preset in TokenOptimizationAdvisoryPolicyPreset] == [
        "disabled",
        "report_only",
        "dry_run_safe",
        "review_first",
        "advisory_allowed_safe",
    ]


def test_policy_overrides_accept_bool_or_none() -> None:
    overrides = TokenOptimizationAdvisoryPolicyOverrides(
        allow_strategy_enable=True,
        allow_strategy_disable=False,
        require_review_for_risky_recommendations=None,
    )
    assert overrides.allow_strategy_enable is True
    assert overrides.allow_strategy_disable is False
    assert overrides.require_review_for_risky_recommendations is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"allow_strategy_enable": "yes"},
        {"allow_strategy_disable": 1},
        {"require_review_for_risky_recommendations": 0.0},
    ],
)
def test_policy_overrides_reject_non_bool_values(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="override must be bool or None"):
        TokenOptimizationAdvisoryPolicyOverrides(**kwargs)


def test_policy_resolution_rejects_auto_apply() -> None:
    policy = TokenOptimizationAdvisoryIntegrationPolicy.dry_run()
    with pytest.raises(ValueError, match="auto_apply_allowed must remain False"):
        TokenOptimizationAdvisoryPolicyResolution(
            preset=TokenOptimizationAdvisoryPolicyPreset.DRY_RUN_SAFE,
            policy=policy,
            auto_apply_allowed=True,
        )


def test_policy_resolution_rejects_raw_content_flag() -> None:
    policy = TokenOptimizationAdvisoryIntegrationPolicy.dry_run()
    with pytest.raises(ValueError, match="raw_content_included must remain False"):
        TokenOptimizationAdvisoryPolicyResolution(
            preset=TokenOptimizationAdvisoryPolicyPreset.DRY_RUN_SAFE,
            policy=policy,
            raw_content_included=True,
        )


def test_policy_resolution_rejects_policy_allow_auto_apply() -> None:
    policy = TokenOptimizationAdvisoryIntegrationPolicy.dry_run()
    object.__setattr__(policy, "allow_auto_apply", True)
    with pytest.raises(ValueError, match="policy.allow_auto_apply must remain False"):
        TokenOptimizationAdvisoryPolicyResolution(
            preset=TokenOptimizationAdvisoryPolicyPreset.DRY_RUN_SAFE,
            policy=policy,
        )


def test_disabled_preset_resolves_disabled_policy() -> None:
    resolution = resolve_token_optimization_advisory_policy(
        TokenOptimizationAdvisoryPolicyPreset.DISABLED,
    )
    assert resolution.policy.enabled is False
    assert resolution.policy.mode is TokenOptimizationAdvisoryIntegrationMode.DISABLED
    assert resolution.policy.allow_strategy_enable is False
    assert resolution.policy.allow_strategy_disable is True
    assert resolution.policy.require_review_for_risky_recommendations is True
    assert resolution.policy.allow_auto_apply is False


def test_report_only_preset_resolves_report_only_policy() -> None:
    resolution = resolve_token_optimization_advisory_policy(
        TokenOptimizationAdvisoryPolicyPreset.REPORT_ONLY,
    )
    assert resolution.policy.enabled is True
    assert resolution.policy.mode is TokenOptimizationAdvisoryIntegrationMode.REPORT_ONLY
    assert resolution.policy.allow_auto_apply is False


def test_dry_run_safe_preset_resolves_dry_run_policy() -> None:
    resolution = resolve_token_optimization_advisory_policy(
        TokenOptimizationAdvisoryPolicyPreset.DRY_RUN_SAFE,
    )
    assert resolution.policy.enabled is True
    assert resolution.policy.mode is TokenOptimizationAdvisoryIntegrationMode.DRY_RUN
    assert resolution.policy.allow_auto_apply is False


def test_review_first_preset_resolves_review_only_policy() -> None:
    resolution = resolve_token_optimization_advisory_policy(
        TokenOptimizationAdvisoryPolicyPreset.REVIEW_FIRST,
    )
    assert resolution.policy.enabled is True
    assert resolution.policy.mode is TokenOptimizationAdvisoryIntegrationMode.REVIEW_ONLY
    assert resolution.policy.allow_auto_apply is False


def test_advisory_allowed_safe_preset_resolves_advisory_allowed_policy() -> None:
    resolution = resolve_token_optimization_advisory_policy(
        TokenOptimizationAdvisoryPolicyPreset.ADVISORY_ALLOWED_SAFE,
    )
    assert resolution.policy.enabled is True
    assert resolution.policy.mode is TokenOptimizationAdvisoryIntegrationMode.ADVISORY_ALLOWED
    assert resolution.policy.allow_strategy_enable is False
    assert resolution.policy.allow_auto_apply is False


def test_disabled_preset_rejects_overrides() -> None:
    with pytest.raises(ValueError, match="DISABLED preset does not accept overrides"):
        resolve_token_optimization_advisory_policy(
            TokenOptimizationAdvisoryPolicyPreset.DISABLED,
            overrides=TokenOptimizationAdvisoryPolicyOverrides(allow_strategy_enable=True),
        )


def test_overrides_are_applied_in_stable_order() -> None:
    overrides = TokenOptimizationAdvisoryPolicyOverrides(
        require_review_for_risky_recommendations=False,
        allow_strategy_disable=False,
        allow_strategy_enable=True,
    )
    resolution = resolve_token_optimization_advisory_policy(
        TokenOptimizationAdvisoryPolicyPreset.ADVISORY_ALLOWED_SAFE,
        overrides=overrides,
    )
    assert resolution.overrides_applied == (
        "allow_strategy_enable",
        "allow_strategy_disable",
        "require_review_for_risky_recommendations",
    )
    assert resolution.policy.allow_strategy_enable is True
    assert resolution.policy.allow_strategy_disable is False
    assert resolution.policy.require_review_for_risky_recommendations is False


def test_resolution_to_dict_is_redaction_safe() -> None:
    resolution = resolve_token_optimization_advisory_policy(
        TokenOptimizationAdvisoryPolicyPreset.DRY_RUN_SAFE,
    )
    payload = token_optimization_advisory_policy_resolution_to_dict(resolution)
    keys = _collect_keys(payload)
    assert keys.isdisjoint(_FORBIDDEN_CONTENT_FIELD_NAMES)
    assert keys.isdisjoint(_FORBIDDEN_SAVINGS_FIELD_NAMES)
    assert payload["auto_apply_allowed"] is False
    assert payload["raw_content_included"] is False
    assert payload["policy_allow_auto_apply"] is False
    json.dumps(payload)


def test_format_policy_resolution_is_deterministic() -> None:
    resolution = resolve_token_optimization_advisory_policy(
        TokenOptimizationAdvisoryPolicyPreset.REVIEW_FIRST,
    )
    first = format_token_optimization_advisory_policy_resolution(resolution)
    second = format_token_optimization_advisory_policy_resolution(resolution)
    assert first == second
    assert "preset=review_first" in first
    assert "policy_mode=review_only" in first


def test_format_policy_resolution_is_redaction_safe() -> None:
    resolution = resolve_token_optimization_advisory_policy(
        TokenOptimizationAdvisoryPolicyPreset.ADVISORY_ALLOWED_SAFE,
        overrides=TokenOptimizationAdvisoryPolicyOverrides(allow_strategy_enable=True),
    )
    text = format_token_optimization_advisory_policy_resolution(resolution)
    lowered = text.lower()
    for forbidden in _FORBIDDEN_CONTENT_FIELD_NAMES:
        assert f"{forbidden}=" not in lowered
    for forbidden in _FORBIDDEN_SAVINGS_FIELD_NAMES:
        assert forbidden not in lowered
    assert "auto_apply_allowed=False" in text
    assert "raw_content_included=False" in text


def test_synthetic_policy_corpus_case_ids_are_unique() -> None:
    case_ids = [case.case_id for case in ADVISORY_POLICY_CORPUS]
    assert len(case_ids) == len(set(case_ids))


def test_synthetic_policy_corpus_has_required_cases() -> None:
    case_ids = {case.case_id for case in ADVISORY_POLICY_CORPUS}
    assert _REQUIRED_CORPUS_CASE_IDS.issubset(case_ids)


def test_synthetic_policy_corpus_has_marker() -> None:
    for case in ADVISORY_POLICY_CORPUS:
        assert case.synthetic_marker == ADVISORY_POLICY_SYNTHETIC_CORPUS_MARKER


@pytest.mark.parametrize("case", ADVISORY_POLICY_CORPUS, ids=lambda case: case.case_id)
def test_synthetic_policy_corpus_all_cases_resolve_as_expected(
    case: object,
) -> None:
    from tests.fixtures.token_optimization.advisory_policy_corpus import (
        AdvisoryPolicyCorpusCase,
    )

    assert isinstance(case, AdvisoryPolicyCorpusCase)
    resolution = resolve_token_optimization_advisory_policy(case.preset, case.overrides)
    assert resolution.policy.enabled is case.expected_enabled
    assert resolution.policy.mode is case.expected_mode
    assert resolution.policy.allow_strategy_enable is case.expected_allow_strategy_enable
    assert resolution.policy.allow_strategy_disable is case.expected_allow_strategy_disable
    assert (
        resolution.policy.require_review_for_risky_recommendations
        is case.expected_require_review_for_risky_recommendations
    )
    assert resolution.overrides_applied == case.expected_overrides_applied
    assert resolution.auto_apply_allowed is False
    assert resolution.raw_content_included is False


def test_resolved_disabled_policy_blocks_integration_gate() -> None:
    resolution = resolve_token_optimization_advisory_policy(
        TokenOptimizationAdvisoryPolicyPreset.DISABLED,
    )
    request = TokenOptimizationAdvisoryIntegrationRequest(
        signal=_keep_current_signal(),
        policy=resolution.policy,
    )
    result = evaluate_policy_gated_advisory_request(request)
    assert result.status is TokenOptimizationAdvisoryIntegrationStatus.BLOCKED_BY_POLICY
    assert result.reason is TokenOptimizationAdvisoryGateReason.POLICY_DISABLED


def test_resolved_dry_run_policy_drives_dry_run_gate() -> None:
    resolution = resolve_token_optimization_advisory_policy(
        TokenOptimizationAdvisoryPolicyPreset.DRY_RUN_SAFE,
    )
    request = TokenOptimizationAdvisoryIntegrationRequest(
        signal=_keep_current_signal(),
        policy=resolution.policy,
    )
    result = evaluate_policy_gated_advisory_request(request)
    assert result.status is TokenOptimizationAdvisoryIntegrationStatus.DRY_RUN
    assert result.reason is TokenOptimizationAdvisoryGateReason.DRY_RUN_MODE


def test_resolved_review_first_policy_drives_review_required_gate() -> None:
    resolution = resolve_token_optimization_advisory_policy(
        TokenOptimizationAdvisoryPolicyPreset.REVIEW_FIRST,
    )
    request = TokenOptimizationAdvisoryIntegrationRequest(
        signal=_keep_current_signal(),
        policy=resolution.policy,
    )
    result = evaluate_policy_gated_advisory_request(request)
    assert result.status is TokenOptimizationAdvisoryIntegrationStatus.REVIEW_REQUIRED
    assert result.reason is TokenOptimizationAdvisoryGateReason.REVIEW_ONLY_MODE


def test_resolved_advisory_allowed_safe_policy_blocks_strategy_enable_by_default() -> None:
    resolution = resolve_token_optimization_advisory_policy(
        TokenOptimizationAdvisoryPolicyPreset.ADVISORY_ALLOWED_SAFE,
    )
    request = TokenOptimizationAdvisoryIntegrationRequest(
        signal=_enable_strategy_signal(),
        policy=resolution.policy,
    )
    result = evaluate_policy_gated_advisory_request(request)
    assert result.status is TokenOptimizationAdvisoryIntegrationStatus.BLOCKED_BY_POLICY
    assert result.reason is TokenOptimizationAdvisoryGateReason.STRATEGY_ENABLE_NOT_ALLOWED


def test_resolved_override_can_allow_strategy_enable_without_auto_apply() -> None:
    resolution = resolve_token_optimization_advisory_policy(
        TokenOptimizationAdvisoryPolicyPreset.ADVISORY_ALLOWED_SAFE,
        overrides=TokenOptimizationAdvisoryPolicyOverrides(allow_strategy_enable=True),
    )
    request = TokenOptimizationAdvisoryIntegrationRequest(
        signal=_enable_strategy_signal(),
        policy=resolution.policy,
    )
    result = evaluate_policy_gated_advisory_request(request)
    assert result.status is TokenOptimizationAdvisoryIntegrationStatus.RECOMMENDATION_READY
    assert result.reason is TokenOptimizationAdvisoryGateReason.ADVISORY_ALLOWED
    assert (
        result.recommendation is not None
        and result.recommendation.action is TokenOptimizationRecommendationAction.ENABLE_STRATEGY
    )
    assert result.auto_apply_allowed is False


@pytest.mark.parametrize(
    "preset",
    list(TokenOptimizationAdvisoryPolicyPreset),
)
def test_resolutions_never_allow_auto_apply(
    preset: TokenOptimizationAdvisoryPolicyPreset,
) -> None:
    resolution = resolve_token_optimization_advisory_policy(preset)
    assert resolution.auto_apply_allowed is False
    assert resolution.policy.allow_auto_apply is False
    payload = token_optimization_advisory_policy_resolution_to_dict(resolution)
    assert payload["auto_apply_allowed"] is False
    assert payload["policy_allow_auto_apply"] is False


@pytest.mark.parametrize(
    "preset",
    list(TokenOptimizationAdvisoryPolicyPreset),
)
def test_resolutions_never_include_raw_content(
    preset: TokenOptimizationAdvisoryPolicyPreset,
) -> None:
    resolution = resolve_token_optimization_advisory_policy(preset)
    assert resolution.raw_content_included is False
    payload = token_optimization_advisory_policy_resolution_to_dict(resolution)
    assert payload["raw_content_included"] is False
    keys = _collect_keys(payload)
    assert keys.isdisjoint(_FORBIDDEN_CONTENT_FIELD_NAMES)
    text = format_token_optimization_advisory_policy_resolution(resolution)
    lowered = text.lower()
    for forbidden in _FORBIDDEN_CONTENT_FIELD_NAMES:
        assert f"{forbidden}=" not in lowered
