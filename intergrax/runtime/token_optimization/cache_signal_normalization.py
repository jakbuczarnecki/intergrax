# © Artur Czarnecki. All rights reserved.

"""Provider-neutral cache signal normalization (TOKEN-10D-2)."""

from __future__ import annotations

from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.runtime.token_optimization.cache_signal_normalization_contracts import (
    CacheAwareCompactionSignalNormalizationReason,
    CacheAwareCompactionSignalNormalizationRequest,
    CacheAwareCompactionSignalNormalizationResult,
    CacheAwareCompactionSignalNormalizationStatus,
    CacheSignalValueSource,
)
from intergrax.runtime.token_optimization.contracts import (
    CacheAwareCompactionTimingInput,
    PromptCacheAttribution,
    PromptCacheInvalidationReason,
    PromptCacheProviderCapabilities,
    PromptCacheUsageSnapshot,
)


def prompt_cache_usage_snapshot_from_adapter_response(
    response: LLMAdapterResponse,
) -> PromptCacheUsageSnapshot | None:
    """Extract typed provider-reported cache usage from an adapter response."""
    if response.provider is None:
        raise ValueError("provider must be explicit for cache usage extraction")
    if response.usage is None:
        return None

    usage = response.usage
    extensions = response.provider_extensions
    usage_source = extensions.usage_source if extensions is not None else "sdk"
    if usage_source == "estimate":
        return None

    if usage.input_tokens < 0 or usage.output_tokens < 0 or usage.cached_input_tokens < 0:
        raise ValueError("token counts cannot be negative")
    if usage.cached_input_tokens > usage.input_tokens:
        raise ValueError("cached_input_tokens exceeds input_tokens")

    provider = response.provider
    model = response.model
    if provider == "vllm":
        return _vllm_usage_snapshot(response, provider=provider, model=model)
    return _generic_usage_snapshot(response, provider=provider, model=model)


def normalize_cache_aware_compaction_signals(
    request: CacheAwareCompactionSignalNormalizationRequest,
) -> CacheAwareCompactionSignalNormalizationResult:
    """Compile provider-neutral cache signals into timing input (no provider I/O)."""
    reason_codes: list[CacheAwareCompactionSignalNormalizationReason] = []
    attribution = request.cache_attribution
    usage = attribution.usage if attribution is not None else None
    capabilities = (
        attribution.provider_capabilities if attribution is not None else None
    )
    provider = usage.provider if usage is not None else None
    model = usage.model if usage is not None else None

    if attribution is None:
        reason_codes.append(CacheAwareCompactionSignalNormalizationReason.ATTRIBUTION_MISSING)
    elif usage is None:
        reason_codes.append(CacheAwareCompactionSignalNormalizationReason.USAGE_NOT_REPORTED)

    rejection = _validate_usage_against_capabilities(usage, capabilities)
    if rejection is not None:
        reason_codes.append(rejection)
        return _rejected_result(
            reason_codes=reason_codes,
            provider=provider,
            model=model,
        )

    if usage is not None and _usage_is_contradictory(usage):
        reason_codes.append(
            CacheAwareCompactionSignalNormalizationReason.CONTRADICTORY_CACHE_EVIDENCE
        )
        return _rejected_result(
            reason_codes=reason_codes,
            provider=provider,
            model=model,
        )

    if capabilities is None and usage is not None:
        reason_codes.append(
            CacheAwareCompactionSignalNormalizationReason.CAPABILITIES_NOT_DECLARED
        )

    cache_hot, cache_hot_source = _derive_cache_hot(usage)
    invalidation_cost, invalidation_cost_source = _derive_invalidation_cost(usage)
    ttl_source = (
        CacheSignalValueSource.RUNTIME_TTL
        if request.ttl_seconds_remaining is not None
        else CacheSignalValueSource.NOT_AVAILABLE
    )
    if request.ttl_seconds_remaining is not None:
        reason_codes.append(CacheAwareCompactionSignalNormalizationReason.TTL_EXPLICIT_RUNTIME)

    if attribution is not None:
        reason_codes.append(
            CacheAwareCompactionSignalNormalizationReason.PREFIX_ATTRIBUTION_PASSTHROUGH
        )
        prefix_stability_status = attribution.prefix_stability_status
        invalidation_reason = attribution.invalidation_reason
    else:
        prefix_stability_status = None
        invalidation_reason = PromptCacheInvalidationReason.UNKNOWN

    if cache_hot is True:
        reason_codes.append(
            CacheAwareCompactionSignalNormalizationReason.CACHE_HOT_POSITIVE_EVIDENCE
        )
    elif cache_hot is False:
        reason_codes.append(CacheAwareCompactionSignalNormalizationReason.CACHE_HOT_EXPLICIT_MISS)
    else:
        reason_codes.append(CacheAwareCompactionSignalNormalizationReason.CACHE_HOT_UNKNOWN)

    status = _derive_status(
        cache_hot=cache_hot,
        capabilities=capabilities,
        usage=usage,
        reason_codes=reason_codes,
    )
    if status is CacheAwareCompactionSignalNormalizationStatus.NORMALIZED:
        reason_codes.append(CacheAwareCompactionSignalNormalizationReason.SIGNALS_COMPLETE)

    timing_input = CacheAwareCompactionTimingInput(
        target=request.target,
        prefix_stability_status=prefix_stability_status,
        invalidation_reason=invalidation_reason,
        cache_hot=cache_hot,
        ttl_seconds_remaining=request.ttl_seconds_remaining,
        near_expiry_threshold_seconds=request.near_expiry_threshold_seconds,
        estimated_content_reduction_chars=request.estimated_content_reduction_chars,
        estimated_cache_invalidation_cost_tokens=invalidation_cost,
        protected_or_semantic_risk=request.protected_or_semantic_risk,
        dynamic_tail_reduction_available=request.dynamic_tail_reduction_available,
    )

    return CacheAwareCompactionSignalNormalizationResult(
        status=status,
        timing_input=timing_input,
        reason_codes=_dedupe_reason_codes(reason_codes),
        cache_hot_source=cache_hot_source,
        invalidation_cost_source=invalidation_cost_source,
        ttl_source=ttl_source,
        provider=provider,
        model=model,
        raw_content_included=False,
    )


_ALLOWED_REPORT_FIELDS = frozenset(
    {
        "status",
        "reason_codes",
        "target",
        "provider",
        "model",
        "cache_hot",
        "cache_hot_source",
        "ttl_seconds_remaining",
        "ttl_source",
        "estimated_content_reduction_chars",
        "estimated_cache_invalidation_cost_tokens",
        "invalidation_cost_source",
        "prefix_stability_status",
        "invalidation_reason",
        "protected_or_semantic_risk",
        "dynamic_tail_reduction_available",
        "timing_input_present",
        "raw_content_included",
    }
)


def cache_signal_normalization_result_to_safe_dict(
    result: CacheAwareCompactionSignalNormalizationResult,
) -> dict[str, object]:
    """Serialize normalization outcome using an allowlist (no raw content)."""
    timing = result.timing_input
    payload: dict[str, object] = {
        "status": result.status.value,
        "reason_codes": [reason.value for reason in result.reason_codes],
        "target": timing.target.value if timing is not None else None,
        "provider": result.provider,
        "model": result.model,
        "cache_hot": timing.cache_hot if timing is not None else None,
        "cache_hot_source": result.cache_hot_source.value,
        "ttl_seconds_remaining": (
            timing.ttl_seconds_remaining if timing is not None else None
        ),
        "ttl_source": result.ttl_source.value,
        "estimated_content_reduction_chars": (
            timing.estimated_content_reduction_chars if timing is not None else None
        ),
        "estimated_cache_invalidation_cost_tokens": (
            timing.estimated_cache_invalidation_cost_tokens
            if timing is not None
            else None
        ),
        "invalidation_cost_source": result.invalidation_cost_source.value,
        "prefix_stability_status": (
            timing.prefix_stability_status if timing is not None else None
        ),
        "invalidation_reason": (
            timing.invalidation_reason.value if timing is not None else None
        ),
        "protected_or_semantic_risk": (
            timing.protected_or_semantic_risk if timing is not None else None
        ),
        "dynamic_tail_reduction_available": (
            timing.dynamic_tail_reduction_available if timing is not None else None
        ),
        "timing_input_present": timing is not None,
        "raw_content_included": False,
    }
    for key in payload:
        if key not in _ALLOWED_REPORT_FIELDS:
            raise ValueError(f"unexpected report field: {key}")
    return payload


def _vllm_usage_snapshot(
    response: LLMAdapterResponse,
    *,
    provider: str,
    model: str | None,
) -> PromptCacheUsageSnapshot:
    extensions = response.provider_extensions
    usage = response.usage
    assert usage is not None

    if extensions is None or extensions.vllm is None:
        if usage.cached_input_tokens > 0:
            raise ValueError(
                "vLLM cached_input_tokens without prompt_tokens_details extension"
            )
        return PromptCacheUsageSnapshot(
            provider=provider,
            model=model,
            cached_input_tokens=None,
            uncached_input_tokens=None,
            cache_hit_ratio=None,
        )

    if not extensions.vllm.prompt_tokens_details_reported:
        return PromptCacheUsageSnapshot(
            provider=provider,
            model=model,
            cached_input_tokens=None,
            uncached_input_tokens=None,
            cache_hit_ratio=None,
        )

    cached = usage.cached_input_tokens
    uncached = usage.uncached_input_tokens
    ratio = _cache_hit_ratio(cached, usage.input_tokens)
    return PromptCacheUsageSnapshot(
        provider=provider,
        model=model,
        cached_input_tokens=cached,
        uncached_input_tokens=uncached,
        cache_hit_ratio=ratio,
    )


def _generic_usage_snapshot(
    response: LLMAdapterResponse,
    *,
    provider: str,
    model: str | None,
) -> PromptCacheUsageSnapshot:
    usage = response.usage
    assert usage is not None
    cached = usage.cached_input_tokens

    if cached > 0:
        uncached = usage.uncached_input_tokens
        ratio = _cache_hit_ratio(cached, usage.input_tokens)
        return PromptCacheUsageSnapshot(
            provider=provider,
            model=model,
            cached_input_tokens=cached,
            uncached_input_tokens=uncached,
            cache_hit_ratio=ratio,
        )

    return PromptCacheUsageSnapshot(
        provider=provider,
        model=model,
        cached_input_tokens=None,
        uncached_input_tokens=None,
        cache_hit_ratio=None,
    )


def _cache_hit_ratio(cached_input_tokens: int, input_tokens: int) -> float | None:
    if input_tokens <= 0:
        return None
    return cached_input_tokens / input_tokens


def _has_positive_cache_evidence(usage: PromptCacheUsageSnapshot) -> bool:
    return (
        (usage.cached_input_tokens is not None and usage.cached_input_tokens > 0)
        or (usage.cache_read_tokens is not None and usage.cache_read_tokens > 0)
        or (usage.cache_hit_ratio is not None and usage.cache_hit_ratio > 0.0)
    )


def _has_explicit_cache_miss(usage: PromptCacheUsageSnapshot) -> bool:
    if usage.cache_hit_ratio == 0.0:
        return True
    if (
        usage.cached_input_tokens == 0
        and usage.uncached_input_tokens is not None
        and usage.uncached_input_tokens > 0
    ):
        return True
    if (
        usage.cache_read_tokens == 0
        and usage.cache_creation_tokens is not None
        and usage.cache_creation_tokens > 0
    ):
        return True
    return False


def _usage_is_contradictory(usage: PromptCacheUsageSnapshot) -> bool:
    if usage.cache_hit_ratio != 0.0:
        return False
    return (
        (usage.cached_input_tokens is not None and usage.cached_input_tokens > 0)
        or (usage.cache_read_tokens is not None and usage.cache_read_tokens > 0)
    )


def _has_positive_cache_fields(usage: PromptCacheUsageSnapshot) -> bool:
    return any(
        value is not None and value > 0
        for value in (
            usage.cache_read_tokens,
            usage.cache_creation_tokens,
            usage.cached_input_tokens,
            usage.uncached_input_tokens,
        )
    )


def _validate_usage_against_capabilities(
    usage: PromptCacheUsageSnapshot | None,
    capabilities: PromptCacheProviderCapabilities | None,
) -> CacheAwareCompactionSignalNormalizationReason | None:
    if usage is None or capabilities is None:
        return None

    if not capabilities.supports_prompt_caching and _has_positive_cache_fields(usage):
        return CacheAwareCompactionSignalNormalizationReason.CAPABILITY_CONTRADICTION

    uses_cached_input = (
        usage.cached_input_tokens is not None and usage.cached_input_tokens > 0
    )
    uses_cache_read = usage.cache_read_tokens is not None and usage.cache_read_tokens > 0
    if uses_cached_input or uses_cache_read:
        if (
            not capabilities.supports_cache_usage_tokens
            and not capabilities.supports_cache_read_tokens
        ):
            return CacheAwareCompactionSignalNormalizationReason.CAPABILITY_CONTRADICTION
        if uses_cached_input and not capabilities.supports_cache_usage_tokens:
            return CacheAwareCompactionSignalNormalizationReason.CAPABILITY_CONTRADICTION
        if uses_cache_read and not capabilities.supports_cache_read_tokens:
            return CacheAwareCompactionSignalNormalizationReason.CAPABILITY_CONTRADICTION

    return None


def _derive_cache_hot(
    usage: PromptCacheUsageSnapshot | None,
) -> tuple[bool | None, CacheSignalValueSource]:
    if usage is None:
        return None, CacheSignalValueSource.NOT_AVAILABLE

    if _has_positive_cache_evidence(usage):
        if usage.cached_input_tokens is not None and usage.cached_input_tokens > 0:
            return True, CacheSignalValueSource.CACHED_INPUT_TOKENS
        if usage.cache_read_tokens is not None and usage.cache_read_tokens > 0:
            return True, CacheSignalValueSource.CACHE_READ_TOKENS
        return True, CacheSignalValueSource.CACHE_HIT_RATIO

    if _has_explicit_cache_miss(usage):
        return False, CacheSignalValueSource.EXPLICIT_MISS

    return None, CacheSignalValueSource.NOT_AVAILABLE


def _derive_invalidation_cost(
    usage: PromptCacheUsageSnapshot | None,
) -> tuple[int | None, CacheSignalValueSource]:
    if usage is None:
        return None, CacheSignalValueSource.NOT_AVAILABLE

    cached = usage.cached_input_tokens
    cache_read = usage.cache_read_tokens
    candidates: list[tuple[int, CacheSignalValueSource]] = []
    if cached is not None and cached > 0:
        candidates.append((cached, CacheSignalValueSource.CACHED_INPUT_TOKENS))
    if cache_read is not None and cache_read > 0:
        candidates.append((cache_read, CacheSignalValueSource.CACHE_READ_TOKENS))

    if not candidates:
        return None, CacheSignalValueSource.NOT_AVAILABLE

    return max(candidates, key=lambda item: item[0])


def _derive_status(
    *,
    cache_hot: bool | None,
    capabilities: PromptCacheProviderCapabilities | None,
    usage: PromptCacheUsageSnapshot | None,
    reason_codes: list[CacheAwareCompactionSignalNormalizationReason],
) -> CacheAwareCompactionSignalNormalizationStatus:
    if (
        CacheAwareCompactionSignalNormalizationReason.CAPABILITIES_NOT_DECLARED
        in reason_codes
    ):
        return CacheAwareCompactionSignalNormalizationStatus.PARTIAL
    if usage is None or cache_hot is None:
        return CacheAwareCompactionSignalNormalizationStatus.PARTIAL
    return CacheAwareCompactionSignalNormalizationStatus.NORMALIZED


def _dedupe_reason_codes(
    reason_codes: list[CacheAwareCompactionSignalNormalizationReason],
) -> tuple[CacheAwareCompactionSignalNormalizationReason, ...]:
    seen: set[CacheAwareCompactionSignalNormalizationReason] = set()
    ordered: list[CacheAwareCompactionSignalNormalizationReason] = []
    for reason in reason_codes:
        if reason in seen:
            continue
        seen.add(reason)
        ordered.append(reason)
    return tuple(ordered)


def _rejected_result(
    *,
    reason_codes: list[CacheAwareCompactionSignalNormalizationReason],
    provider: str | None,
    model: str | None,
) -> CacheAwareCompactionSignalNormalizationResult:
    return CacheAwareCompactionSignalNormalizationResult(
        status=CacheAwareCompactionSignalNormalizationStatus.REJECTED,
        timing_input=None,
        reason_codes=_dedupe_reason_codes(reason_codes),
        cache_hot_source=CacheSignalValueSource.NOT_AVAILABLE,
        invalidation_cost_source=CacheSignalValueSource.NOT_AVAILABLE,
        ttl_source=CacheSignalValueSource.NOT_AVAILABLE,
        provider=provider,
        model=model,
        raw_content_included=False,
    )
