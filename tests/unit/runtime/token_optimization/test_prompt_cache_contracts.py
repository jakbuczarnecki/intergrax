# © Artur Czarnecki. All rights reserved.

"""TOKEN-OPT-5B: provider prompt-cache contract tests."""

from __future__ import annotations

import dataclasses

import pytest

from intergrax.runtime.token_optimization.contracts import (
    PromptCacheAttribution,
    PromptCacheInvalidationReason,
    PromptCacheMode,
    PromptCachePolicy,
    PromptCacheProviderCapabilities,
    PromptCacheUsageSnapshot,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_FORBIDDEN_USAGE_FIELD_NAMES = frozenset(
    {
        "saved_tokens",
        "optimized_tokens",
        "baseline_tokens",
        "compressed_tokens",
    }
)


def test_prompt_cache_mode_values_are_stable() -> None:
    assert PromptCacheMode.OFF == "off"
    assert PromptCacheMode.PROVIDER_DEFAULT == "provider_default"
    assert PromptCacheMode.EXPLICIT_BREAKPOINTS == "explicit_breakpoints"
    assert PromptCacheMode.CACHE_KEY == "cache_key"
    assert PromptCacheMode.SESSION_AFFINITY == "session_affinity"
    assert [mode.value for mode in PromptCacheMode] == [
        "off",
        "provider_default",
        "explicit_breakpoints",
        "cache_key",
        "session_affinity",
    ]


def test_prompt_cache_invalidation_reason_values_are_stable() -> None:
    assert [reason.value for reason in PromptCacheInvalidationReason] == [
        "none",
        "disabled",
        "unsupported_provider",
        "prefix_changed",
        "append_only_violation",
        "tool_envelope_changed",
        "dynamic_data_in_prefix",
        "ttl_expired",
        "cache_key_changed",
        "session_changed",
        "provider_not_reported",
        "unknown",
    ]


def test_prompt_cache_provider_capabilities_validate_provider() -> None:
    with pytest.raises(ValueError, match="provider cannot be empty"):
        PromptCacheProviderCapabilities(provider="   ")
    caps = PromptCacheProviderCapabilities(provider="synth-provider")
    assert caps.provider == "synth-provider"
    assert caps.supports_prompt_caching is False


def test_prompt_cache_provider_capabilities_reject_feature_flags_when_prompt_cache_unsupported() -> None:
    with pytest.raises(ValueError, match="must not claim specific prompt-cache features"):
        PromptCacheProviderCapabilities(
            provider="synth-provider",
            supports_prompt_caching=False,
            supports_automatic_caching=True,
        )


def test_prompt_cache_provider_capabilities_validate_non_negative_limits() -> None:
    with pytest.raises(ValueError, match="max_cache_breakpoints cannot be negative"):
        PromptCacheProviderCapabilities(
            provider="synth-provider",
            supports_prompt_caching=True,
            max_cache_breakpoints=-1,
        )


def test_prompt_cache_provider_capabilities_validate_ttl_order() -> None:
    with pytest.raises(ValueError, match="default_ttl_seconds cannot exceed max_ttl_seconds"):
        PromptCacheProviderCapabilities(
            provider="synth-provider",
            supports_prompt_caching=True,
            supports_cache_retention_ttl=True,
            default_ttl_seconds=3600,
            max_ttl_seconds=60,
        )


def test_prompt_cache_policy_disabled_requires_off_mode() -> None:
    policy = PromptCachePolicy.disabled()
    assert policy.enabled is False
    assert policy.mode is PromptCacheMode.OFF
    with pytest.raises(ValueError, match="mode must be OFF when enabled is False"):
        PromptCachePolicy(enabled=False, mode=PromptCacheMode.PROVIDER_DEFAULT)


def test_prompt_cache_policy_enabled_requires_non_off_mode() -> None:
    with pytest.raises(ValueError, match="mode must not be OFF when enabled is True"):
        PromptCachePolicy(enabled=True, mode=PromptCacheMode.OFF)
    policy = PromptCachePolicy(enabled=True, mode=PromptCacheMode.PROVIDER_DEFAULT)
    assert policy.mode is PromptCacheMode.PROVIDER_DEFAULT


def test_prompt_cache_policy_explicit_breakpoints_requires_allow_flag() -> None:
    with pytest.raises(ValueError, match="allow_explicit_breakpoints"):
        PromptCachePolicy(
            enabled=True,
            mode=PromptCacheMode.EXPLICIT_BREAKPOINTS,
            allow_explicit_breakpoints=False,
        )


def test_prompt_cache_policy_cache_key_requires_scope() -> None:
    with pytest.raises(ValueError, match="allow_cache_key"):
        PromptCachePolicy(
            enabled=True,
            mode=PromptCacheMode.CACHE_KEY,
            allow_cache_key=False,
            cache_key_scope="tenant",
        )
    with pytest.raises(ValueError, match="cache_key_scope"):
        PromptCachePolicy(
            enabled=True,
            mode=PromptCacheMode.CACHE_KEY,
            allow_cache_key=True,
            cache_key_scope=None,
        )
    with pytest.raises(ValueError, match="cache_key_scope cannot be empty"):
        PromptCachePolicy(
            enabled=True,
            mode=PromptCacheMode.CACHE_KEY,
            allow_cache_key=True,
            cache_key_scope="  ",
        )


def test_prompt_cache_policy_session_affinity_requires_allow_flag() -> None:
    with pytest.raises(ValueError, match="allow_session_affinity"):
        PromptCachePolicy(
            enabled=True,
            mode=PromptCacheMode.SESSION_AFFINITY,
            allow_session_affinity=False,
        )


def test_prompt_cache_usage_snapshot_validates_provider() -> None:
    with pytest.raises(ValueError, match="provider cannot be empty"):
        PromptCacheUsageSnapshot(provider="")


def test_prompt_cache_usage_snapshot_validates_non_negative_token_fields() -> None:
    with pytest.raises(ValueError, match="cache_read_tokens cannot be negative"):
        PromptCacheUsageSnapshot(provider="synth-provider", cache_read_tokens=-1)


def test_prompt_cache_usage_snapshot_validates_hit_ratio() -> None:
    with pytest.raises(ValueError, match="cache_hit_ratio must be between"):
        PromptCacheUsageSnapshot(provider="synth-provider", cache_hit_ratio=1.5)


def test_prompt_cache_attribution_keeps_cache_and_content_reduction_separate() -> None:
    usage = PromptCacheUsageSnapshot(
        provider="synth-provider",
        cache_read_tokens=100,
        cached_input_tokens=100,
    )
    attribution = PromptCacheAttribution(
        policy=PromptCachePolicy.disabled(),
        usage=usage,
        content_reduction_strategy="extractive_filtering",
        content_saved_chars=40,
        content_saved_tokens=10,
    )
    assert attribution.has_provider_cache_usage() is True
    assert attribution.has_content_reduction() is True
    assert attribution.usage is not None
    assert attribution.usage.cache_read_tokens == 100
    assert attribution.content_saved_tokens == 10
    field_names = {field.name for field in dataclasses.fields(PromptCacheUsageSnapshot)}
    assert field_names.isdisjoint(_FORBIDDEN_USAGE_FIELD_NAMES)


def test_prompt_cache_attribution_validates_provider_match() -> None:
    with pytest.raises(ValueError, match="must match"):
        PromptCacheAttribution(
            policy=PromptCachePolicy.disabled(),
            provider_capabilities=PromptCacheProviderCapabilities(provider="alpha"),
            usage=PromptCacheUsageSnapshot(provider="beta", cache_read_tokens=1),
        )


def test_prompt_cache_attribution_does_not_compute_token_savings_from_cache_usage() -> None:
    attribution = PromptCacheAttribution(
        policy=PromptCachePolicy.disabled(),
        usage=PromptCacheUsageSnapshot(
            provider="synth-provider",
            cache_read_tokens=250,
            cached_input_tokens=250,
        ),
    )
    assert attribution.content_saved_tokens is None
    assert attribution.content_saved_chars is None
    assert attribution.has_provider_cache_usage() is True
    assert attribution.has_content_reduction() is False
    assert not hasattr(attribution, "saved_tokens")
    assert not hasattr(attribution.usage, "saved_tokens")
