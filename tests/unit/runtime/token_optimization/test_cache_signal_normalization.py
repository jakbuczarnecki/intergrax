# © Artur Czarnecki. All rights reserved.

"""TOKEN-10D-2: cache signal normalization unit tests."""

from __future__ import annotations

import inspect

import pytest

from intergrax.llm_adapters._shared.adapter_response_builders import build_adapter_response
from intergrax.llm_adapters.contracts.provider_extensions import (
    LLMProviderExtensions,
    VllmProviderExtensions,
)
from intergrax.llm_adapters.contracts.token_usage import LLMTokenUsage
from intergrax.runtime.token_optimization.cache_signal_normalization import (
    cache_signal_normalization_result_to_safe_dict,
    normalize_cache_aware_compaction_signals,
    prompt_cache_usage_snapshot_from_adapter_response,
)
from intergrax.runtime.token_optimization.cache_signal_normalization_contracts import (
    CacheAwareCompactionSignalNormalizationReason,
    CacheAwareCompactionSignalNormalizationRequest,
    CacheAwareCompactionSignalNormalizationResult,
    CacheAwareCompactionSignalNormalizationStatus,
    CacheSignalValueSource,
)
from intergrax.runtime.token_optimization.contracts import (
    CacheAwareCompactionTarget,
    CacheAwareCompactionTimingInput,
    PromptCacheAttribution,
    PromptCacheInvalidationReason,
    PromptCacheMode,
    PromptCachePolicy,
    PromptCacheProviderCapabilities,
    PromptCacheUsageSnapshot,
)
from intergrax.runtime.token_optimization.prompt_cache import PREFIX_STABILITY_STABLE

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _vllm_response(
    *,
    input_tokens: int = 1000,
    cached_input_tokens: int = 0,
    details_reported: bool = True,
    usage_source: str = "sdk",
    provider_extensions: LLMProviderExtensions | None = None,
):
    if provider_extensions is None:
        provider_extensions = LLMProviderExtensions(
            usage_source=usage_source,  # type: ignore[arg-type]
            vllm=VllmProviderExtensions(
                prompt_tokens_details_reported=details_reported,
            ),
        )
    return build_adapter_response(
        content="ok",
        provider="vllm",
        model="vllm-test",
        usage=LLMTokenUsage.from_counts(
            input_tokens=input_tokens,
            output_tokens=10,
            cached_input_tokens=cached_input_tokens,
        ),
        provider_extensions=provider_extensions,
    )


def _generic_response(
    *,
    provider: str = "openai",
    input_tokens: int = 500,
    cached_input_tokens: int = 0,
    usage_source: str = "sdk",
):
    return build_adapter_response(
        content="ok",
        provider=provider,
        model="generic-test",
        usage=LLMTokenUsage.from_counts(
            input_tokens=input_tokens,
            output_tokens=5,
            cached_input_tokens=cached_input_tokens,
        ),
        provider_extensions=LLMProviderExtensions(usage_source=usage_source),  # type: ignore[arg-type]
    )


def _attribution(
    usage: PromptCacheUsageSnapshot | None,
    *,
    capabilities: PromptCacheProviderCapabilities | None = None,
    prefix_stability_status: str | None = PREFIX_STABILITY_STABLE,
    invalidation_reason: PromptCacheInvalidationReason = PromptCacheInvalidationReason.NONE,
) -> PromptCacheAttribution:
    return PromptCacheAttribution(
        policy=PromptCachePolicy(enabled=True, mode=PromptCacheMode.PROVIDER_DEFAULT),
        provider_capabilities=capabilities,
        usage=usage,
        prefix_stability_status=prefix_stability_status,
        invalidation_reason=invalidation_reason,
    )


def _vllm_capabilities() -> PromptCacheProviderCapabilities:
    return PromptCacheProviderCapabilities(
        provider="vllm",
        supports_prompt_caching=True,
        supports_cache_usage_tokens=True,
    )


# --- adapter extraction ---


def test_vllm_reported_cache_hit_extraction() -> None:
    response = _vllm_response(input_tokens=1000, cached_input_tokens=800)
    snapshot = prompt_cache_usage_snapshot_from_adapter_response(response)
    assert snapshot is not None
    assert snapshot.cached_input_tokens == 800
    assert snapshot.uncached_input_tokens == 200
    assert snapshot.cache_hit_ratio == pytest.approx(0.8)


def test_vllm_explicit_cache_miss_extraction() -> None:
    response = _vllm_response(input_tokens=1000, cached_input_tokens=0)
    snapshot = prompt_cache_usage_snapshot_from_adapter_response(response)
    assert snapshot is not None
    assert snapshot.cached_input_tokens == 0
    assert snapshot.uncached_input_tokens == 1000
    assert snapshot.cache_hit_ratio == 0.0


def test_vllm_details_not_reported_extraction() -> None:
    response = _vllm_response(
        input_tokens=1000,
        cached_input_tokens=0,
        details_reported=False,
    )
    snapshot = prompt_cache_usage_snapshot_from_adapter_response(response)
    assert snapshot is not None
    assert snapshot.cached_input_tokens is None
    assert snapshot.uncached_input_tokens is None
    assert snapshot.cache_hit_ratio is None


def test_vllm_extension_missing_zero_not_miss() -> None:
    response = build_adapter_response(
        content="ok",
        provider="vllm",
        model="vllm-test",
        usage=LLMTokenUsage.from_counts(
            input_tokens=1000,
            output_tokens=1,
            cached_input_tokens=0,
        ),
        provider_extensions=LLMProviderExtensions(usage_source="sdk"),
    )
    snapshot = prompt_cache_usage_snapshot_from_adapter_response(response)
    assert snapshot is not None
    assert snapshot.cached_input_tokens is None
    assert snapshot.uncached_input_tokens is None


def test_vllm_extension_missing_positive_rejected() -> None:
    response = build_adapter_response(
        content="ok",
        provider="vllm",
        model="vllm-test",
        usage=LLMTokenUsage.from_counts(
            input_tokens=1000,
            output_tokens=1,
            cached_input_tokens=50,
        ),
        provider_extensions=LLMProviderExtensions(usage_source="sdk"),
    )
    with pytest.raises(ValueError, match="prompt_tokens_details extension"):
        prompt_cache_usage_snapshot_from_adapter_response(response)


def test_missing_usage_extraction() -> None:
    response = build_adapter_response(content="ok", provider="vllm", model="m", usage=None)
    assert prompt_cache_usage_snapshot_from_adapter_response(response) is None


def test_estimated_usage_extraction_has_no_cache_evidence() -> None:
    response = _vllm_response(
        input_tokens=1000,
        cached_input_tokens=800,
        usage_source="estimate",
    )
    assert prompt_cache_usage_snapshot_from_adapter_response(response) is None


def test_cached_tokens_exceed_input_tokens_rejected() -> None:
    response = _vllm_response(input_tokens=100, cached_input_tokens=200)
    with pytest.raises(ValueError, match="cached_input_tokens exceeds input_tokens"):
        prompt_cache_usage_snapshot_from_adapter_response(response)


def test_missing_provider_rejected() -> None:
    response = build_adapter_response(
        content="ok",
        provider=None,
        usage=LLMTokenUsage.from_counts(input_tokens=10, output_tokens=1),
    )
    with pytest.raises(ValueError, match="provider must be explicit"):
        prompt_cache_usage_snapshot_from_adapter_response(response)


def test_generic_provider_positive_cached_usage() -> None:
    response = _generic_response(cached_input_tokens=120, input_tokens=500)
    snapshot = prompt_cache_usage_snapshot_from_adapter_response(response)
    assert snapshot is not None
    assert snapshot.cached_input_tokens == 120
    assert snapshot.uncached_input_tokens == 380
    assert snapshot.cache_hit_ratio == pytest.approx(0.24)


def test_generic_provider_zero_without_reported_flag_is_unknown() -> None:
    response = _generic_response(cached_input_tokens=0, input_tokens=500)
    snapshot = prompt_cache_usage_snapshot_from_adapter_response(response)
    assert snapshot is not None
    assert snapshot.cached_input_tokens is None
    assert snapshot.uncached_input_tokens is None
    assert snapshot.cache_hit_ratio is None


# --- normalization ---


def test_positive_cached_tokens_normalization() -> None:
    usage = PromptCacheUsageSnapshot(
        provider="vllm",
        cached_input_tokens=800,
        uncached_input_tokens=200,
        cache_hit_ratio=0.8,
    )
    result = normalize_cache_aware_compaction_signals(
        CacheAwareCompactionSignalNormalizationRequest(
            target=CacheAwareCompactionTarget.STABLE_PREFIX,
            cache_attribution=_attribution(usage, capabilities=_vllm_capabilities()),
            estimated_content_reduction_chars=50,
        )
    )
    assert result.status is CacheAwareCompactionSignalNormalizationStatus.NORMALIZED
    assert result.timing_input is not None
    assert result.timing_input.cache_hot is True
    assert result.timing_input.estimated_cache_invalidation_cost_tokens == 800
    assert result.cache_hot_source is CacheSignalValueSource.CACHED_INPUT_TOKENS


def test_positive_cache_read_tokens_normalization() -> None:
    usage = PromptCacheUsageSnapshot(provider="anthropic", cache_read_tokens=300)
    caps = PromptCacheProviderCapabilities(
        provider="anthropic",
        supports_prompt_caching=True,
        supports_cache_read_tokens=True,
    )
    result = normalize_cache_aware_compaction_signals(
        CacheAwareCompactionSignalNormalizationRequest(
            target=CacheAwareCompactionTarget.STABLE_PREFIX,
            cache_attribution=_attribution(usage, capabilities=caps),
        )
    )
    assert result.timing_input is not None
    assert result.timing_input.cache_hot is True
    assert result.timing_input.estimated_cache_invalidation_cost_tokens == 300
    assert result.invalidation_cost_source is CacheSignalValueSource.CACHE_READ_TOKENS


def test_both_token_fields_use_max_not_sum() -> None:
    usage = PromptCacheUsageSnapshot(
        provider="vllm",
        cached_input_tokens=400,
        cache_read_tokens=700,
    )
    caps = PromptCacheProviderCapabilities(
        provider="vllm",
        supports_prompt_caching=True,
        supports_cache_usage_tokens=True,
        supports_cache_read_tokens=True,
    )
    result = normalize_cache_aware_compaction_signals(
        CacheAwareCompactionSignalNormalizationRequest(
            target=CacheAwareCompactionTarget.STABLE_PREFIX,
            cache_attribution=_attribution(usage, capabilities=caps),
        )
    )
    assert result.timing_input is not None
    assert result.timing_input.estimated_cache_invalidation_cost_tokens == 700


def test_explicit_miss_normalization() -> None:
    usage = PromptCacheUsageSnapshot(
        provider="vllm",
        cached_input_tokens=0,
        uncached_input_tokens=1000,
        cache_hit_ratio=0.0,
    )
    result = normalize_cache_aware_compaction_signals(
        CacheAwareCompactionSignalNormalizationRequest(
            target=CacheAwareCompactionTarget.COLD_HISTORY,
            cache_attribution=_attribution(usage, capabilities=_vllm_capabilities()),
        )
    )
    assert result.status is CacheAwareCompactionSignalNormalizationStatus.NORMALIZED
    assert result.timing_input is not None
    assert result.timing_input.cache_hot is False


def test_missing_usage_partial_normalization() -> None:
    result = normalize_cache_aware_compaction_signals(
        CacheAwareCompactionSignalNormalizationRequest(
            target=CacheAwareCompactionTarget.STABLE_PREFIX,
            cache_attribution=_attribution(None, capabilities=_vllm_capabilities()),
        )
    )
    assert result.status is CacheAwareCompactionSignalNormalizationStatus.PARTIAL
    assert result.timing_input is not None
    assert result.timing_input.cache_hot is None
    assert result.timing_input.estimated_cache_invalidation_cost_tokens is None


def test_estimated_usage_only_partial_normalization() -> None:
    result = normalize_cache_aware_compaction_signals(
        CacheAwareCompactionSignalNormalizationRequest(
            target=CacheAwareCompactionTarget.STABLE_PREFIX,
            cache_attribution=None,
        )
    )
    assert result.status is CacheAwareCompactionSignalNormalizationStatus.PARTIAL
    assert result.timing_input is not None
    assert result.timing_input.cache_hot is None


def test_contradictory_hit_evidence_rejected() -> None:
    usage = PromptCacheUsageSnapshot(
        provider="vllm",
        cached_input_tokens=500,
        cache_hit_ratio=0.0,
    )
    result = normalize_cache_aware_compaction_signals(
        CacheAwareCompactionSignalNormalizationRequest(
            target=CacheAwareCompactionTarget.STABLE_PREFIX,
            cache_attribution=_attribution(usage, capabilities=_vllm_capabilities()),
        )
    )
    assert result.status is CacheAwareCompactionSignalNormalizationStatus.REJECTED
    assert result.timing_input is None


def test_capability_contradiction_rejected() -> None:
    usage = PromptCacheUsageSnapshot(provider="vllm", cached_input_tokens=100)
    caps = PromptCacheProviderCapabilities(provider="vllm", supports_prompt_caching=False)
    result = normalize_cache_aware_compaction_signals(
        CacheAwareCompactionSignalNormalizationRequest(
            target=CacheAwareCompactionTarget.STABLE_PREFIX,
            cache_attribution=_attribution(usage, capabilities=caps),
        )
    )
    assert result.status is CacheAwareCompactionSignalNormalizationStatus.REJECTED
    assert result.timing_input is None


def test_capabilities_missing_partial() -> None:
    usage = PromptCacheUsageSnapshot(provider="vllm", cached_input_tokens=100)
    result = normalize_cache_aware_compaction_signals(
        CacheAwareCompactionSignalNormalizationRequest(
            target=CacheAwareCompactionTarget.STABLE_PREFIX,
            cache_attribution=_attribution(usage, capabilities=None),
        )
    )
    assert result.status is CacheAwareCompactionSignalNormalizationStatus.PARTIAL
    assert result.timing_input is not None
    assert result.timing_input.cache_hot is True


def test_prefix_state_pass_through() -> None:
    usage = PromptCacheUsageSnapshot(provider="vllm", cached_input_tokens=10)
    result = normalize_cache_aware_compaction_signals(
        CacheAwareCompactionSignalNormalizationRequest(
            target=CacheAwareCompactionTarget.STABLE_PREFIX,
            cache_attribution=_attribution(
                usage,
                capabilities=_vllm_capabilities(),
                prefix_stability_status="invalidated",
                invalidation_reason=PromptCacheInvalidationReason.PREFIX_CHANGED,
            ),
        )
    )
    assert result.timing_input is not None
    assert result.timing_input.prefix_stability_status == "invalidated"
    assert (
        result.timing_input.invalidation_reason
        is PromptCacheInvalidationReason.PREFIX_CHANGED
    )


def test_attribution_missing_unknown_invalidation() -> None:
    result = normalize_cache_aware_compaction_signals(
        CacheAwareCompactionSignalNormalizationRequest(
            target=CacheAwareCompactionTarget.STABLE_PREFIX,
        )
    )
    assert result.timing_input is not None
    assert result.timing_input.prefix_stability_status is None
    assert (
        result.timing_input.invalidation_reason is PromptCacheInvalidationReason.UNKNOWN
    )


def test_explicit_ttl_passed_through() -> None:
    usage = PromptCacheUsageSnapshot(provider="vllm", cached_input_tokens=50)
    result = normalize_cache_aware_compaction_signals(
        CacheAwareCompactionSignalNormalizationRequest(
            target=CacheAwareCompactionTarget.STABLE_PREFIX,
            cache_attribution=_attribution(usage, capabilities=_vllm_capabilities()),
            ttl_seconds_remaining=120,
        )
    )
    assert result.timing_input is not None
    assert result.timing_input.ttl_seconds_remaining == 120
    assert result.ttl_source is CacheSignalValueSource.RUNTIME_TTL


def test_requested_ttl_not_used_as_remaining_ttl() -> None:
    policy = PromptCachePolicy(
        enabled=True,
        mode=PromptCacheMode.PROVIDER_DEFAULT,
        requested_ttl_seconds=3600,
    )
    usage = PromptCacheUsageSnapshot(provider="vllm", cached_input_tokens=10)
    attribution = PromptCacheAttribution(
        policy=policy,
        provider_capabilities=_vllm_capabilities(),
        usage=usage,
    )
    result = normalize_cache_aware_compaction_signals(
        CacheAwareCompactionSignalNormalizationRequest(
            target=CacheAwareCompactionTarget.STABLE_PREFIX,
            cache_attribution=attribution,
        )
    )
    assert result.timing_input is not None
    assert result.timing_input.ttl_seconds_remaining is None


def test_default_ttl_not_used_as_remaining_ttl() -> None:
    caps = PromptCacheProviderCapabilities(
        provider="vllm",
        supports_prompt_caching=True,
        supports_cache_usage_tokens=True,
        default_ttl_seconds=600,
    )
    usage = PromptCacheUsageSnapshot(provider="vllm", cached_input_tokens=10)
    result = normalize_cache_aware_compaction_signals(
        CacheAwareCompactionSignalNormalizationRequest(
            target=CacheAwareCompactionTarget.STABLE_PREFIX,
            cache_attribution=_attribution(usage, capabilities=caps),
        )
    )
    assert result.timing_input is not None
    assert result.timing_input.ttl_seconds_remaining is None


def test_content_reduction_chars_passed_through() -> None:
    usage = PromptCacheUsageSnapshot(provider="vllm", cached_input_tokens=10)
    result = normalize_cache_aware_compaction_signals(
        CacheAwareCompactionSignalNormalizationRequest(
            target=CacheAwareCompactionTarget.STABLE_PREFIX,
            cache_attribution=_attribution(usage, capabilities=_vllm_capabilities()),
            estimated_content_reduction_chars=250,
        )
    )
    assert result.timing_input is not None
    assert result.timing_input.estimated_content_reduction_chars == 250


def test_no_char_to_token_conversion() -> None:
    source = inspect.getsource(normalize_cache_aware_compaction_signals)
    assert "estimated_content_reduction_chars" in source
    usage = PromptCacheUsageSnapshot(provider="vllm")
    result = normalize_cache_aware_compaction_signals(
        CacheAwareCompactionSignalNormalizationRequest(
            target=CacheAwareCompactionTarget.STABLE_PREFIX,
            cache_attribution=_attribution(usage, capabilities=_vllm_capabilities()),
            estimated_content_reduction_chars=9999,
        )
    )
    assert result.timing_input is not None
    assert result.timing_input.estimated_cache_invalidation_cost_tokens is None


def test_dynamic_tail_signal_passed_through() -> None:
    usage = PromptCacheUsageSnapshot(provider="vllm", cached_input_tokens=1)
    result = normalize_cache_aware_compaction_signals(
        CacheAwareCompactionSignalNormalizationRequest(
            target=CacheAwareCompactionTarget.DYNAMIC_TAIL,
            cache_attribution=_attribution(usage, capabilities=_vllm_capabilities()),
            dynamic_tail_reduction_available=True,
        )
    )
    assert result.timing_input is not None
    assert result.timing_input.dynamic_tail_reduction_available is True


def test_protected_risk_passed_through() -> None:
    result = normalize_cache_aware_compaction_signals(
        CacheAwareCompactionSignalNormalizationRequest(
            target=CacheAwareCompactionTarget.STABLE_PREFIX,
            protected_or_semantic_risk=True,
        )
    )
    assert result.timing_input is not None
    assert result.timing_input.protected_or_semantic_risk is True


def test_global_kv_usage_not_imported() -> None:
    import intergrax.runtime.token_optimization.cache_signal_normalization as module

    source = inspect.getsource(module)
    assert "httpx" not in source
    assert "collect_vllm_diagnostics" not in source
    assert "fetch_vllm_metrics" not in source
    assert "parse_prometheus" not in source
    assert "kv_cache_usage_perc" not in source


def test_safe_report_allowlist() -> None:
    usage = PromptCacheUsageSnapshot(provider="vllm", cached_input_tokens=50)
    result = normalize_cache_aware_compaction_signals(
        CacheAwareCompactionSignalNormalizationRequest(
            target=CacheAwareCompactionTarget.STABLE_PREFIX,
            cache_attribution=_attribution(usage, capabilities=_vllm_capabilities()),
            estimated_content_reduction_chars=10,
            ttl_seconds_remaining=30,
        )
    )
    safe = cache_signal_normalization_result_to_safe_dict(result)
    assert safe["raw_content_included"] is False
    assert "content" not in safe
    assert "response" not in safe
    assert "metadata" not in safe
    assert safe["timing_input_present"] is True


def test_contract_invariants_rejected_without_timing_input() -> None:
    with pytest.raises(ValueError, match="REJECTED requires timing_input=None"):
        CacheAwareCompactionSignalNormalizationResult(
            status=CacheAwareCompactionSignalNormalizationStatus.REJECTED,
            timing_input=CacheAwareCompactionTimingInput(
                target=CacheAwareCompactionTarget.STABLE_PREFIX,
            ),
            reason_codes=(
                CacheAwareCompactionSignalNormalizationReason.CONTRADICTORY_CACHE_EVIDENCE,
            ),
            cache_hot_source=CacheSignalValueSource.NOT_AVAILABLE,
            invalidation_cost_source=CacheSignalValueSource.NOT_AVAILABLE,
            ttl_source=CacheSignalValueSource.NOT_AVAILABLE,
        )


def test_contract_invariants_partial_requires_timing_input() -> None:
    with pytest.raises(ValueError, match="partial requires timing_input"):
        CacheAwareCompactionSignalNormalizationResult(
            status=CacheAwareCompactionSignalNormalizationStatus.PARTIAL,
            timing_input=None,
            reason_codes=(CacheAwareCompactionSignalNormalizationReason.CACHE_HOT_UNKNOWN,),
            cache_hot_source=CacheSignalValueSource.NOT_AVAILABLE,
            invalidation_cost_source=CacheSignalValueSource.NOT_AVAILABLE,
            ttl_source=CacheSignalValueSource.NOT_AVAILABLE,
        )
