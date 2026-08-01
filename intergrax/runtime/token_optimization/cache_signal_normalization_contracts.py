# © Artur Czarnecki. All rights reserved.

"""Cache signal normalization contracts (TOKEN-10D-2)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from intergrax.runtime.token_optimization.contracts import (
    CacheAwareCompactionTarget,
    CacheAwareCompactionTimingInput,
    PromptCacheAttribution,
    PromptCacheInvalidationReason,
)


class CacheAwareCompactionSignalNormalizationStatus(StrEnum):
    """Outcome of provider-neutral cache signal normalization."""

    NORMALIZED = "normalized"
    PARTIAL = "partial"
    REJECTED = "rejected"


class CacheAwareCompactionSignalNormalizationReason(StrEnum):
    """Deterministic reason codes for cache signal normalization."""

    CACHE_HOT_POSITIVE_EVIDENCE = "cache_hot_positive_evidence"
    CACHE_HOT_EXPLICIT_MISS = "cache_hot_explicit_miss"
    CACHE_HOT_UNKNOWN = "cache_hot_unknown"
    USAGE_NOT_REPORTED = "usage_not_reported"
    CONTRADICTORY_CACHE_EVIDENCE = "contradictory_cache_evidence"
    CAPABILITY_CONTRADICTION = "capability_contradiction"
    CAPABILITIES_NOT_DECLARED = "capabilities_not_declared"
    TTL_EXPLICIT_RUNTIME = "ttl_explicit_runtime"
    PREFIX_ATTRIBUTION_PASSTHROUGH = "prefix_attribution_passthrough"
    ATTRIBUTION_MISSING = "attribution_missing"
    SIGNALS_COMPLETE = "signals_complete"


class CacheSignalValueSource(StrEnum):
    """Provenance for normalized cache timing fields."""

    CACHED_INPUT_TOKENS = "cached_input_tokens"
    CACHE_READ_TOKENS = "cache_read_tokens"
    CACHE_HIT_RATIO = "cache_hit_ratio"
    EXPLICIT_MISS = "explicit_miss"
    RUNTIME_TTL = "runtime_ttl"
    NOT_AVAILABLE = "not_available"


@dataclass(frozen=True, slots=True)
class CacheAwareCompactionSignalNormalizationRequest:
    """Caller-supplied cache signals for timing-input compilation."""

    target: CacheAwareCompactionTarget
    cache_attribution: PromptCacheAttribution | None = None
    ttl_seconds_remaining: int | None = None
    near_expiry_threshold_seconds: int = 60
    estimated_content_reduction_chars: int | None = None
    protected_or_semantic_risk: bool = False
    dynamic_tail_reduction_available: bool = False

    def __post_init__(self) -> None:
        if self.ttl_seconds_remaining is not None and self.ttl_seconds_remaining < 0:
            raise ValueError("ttl_seconds_remaining cannot be negative")
        if self.near_expiry_threshold_seconds < 0:
            raise ValueError("near_expiry_threshold_seconds cannot be negative")
        if (
            self.estimated_content_reduction_chars is not None
            and self.estimated_content_reduction_chars < 0
        ):
            raise ValueError("estimated_content_reduction_chars cannot be negative")


@dataclass(frozen=True, slots=True)
class CacheAwareCompactionSignalNormalizationResult:
    """Provider-neutral timing-input compilation outcome (no raw content)."""

    status: CacheAwareCompactionSignalNormalizationStatus
    timing_input: CacheAwareCompactionTimingInput | None
    reason_codes: tuple[CacheAwareCompactionSignalNormalizationReason, ...]
    cache_hot_source: CacheSignalValueSource
    invalidation_cost_source: CacheSignalValueSource
    ttl_source: CacheSignalValueSource
    provider: str | None = None
    model: str | None = None
    raw_content_included: bool = False

    def __post_init__(self) -> None:
        if self.raw_content_included:
            raise ValueError("raw_content_included must remain False")
        if (
            self.status
            in {
                CacheAwareCompactionSignalNormalizationStatus.NORMALIZED,
                CacheAwareCompactionSignalNormalizationStatus.PARTIAL,
            }
            and self.timing_input is None
        ):
            raise ValueError(f"{self.status.value} requires timing_input")
        if (
            self.status is CacheAwareCompactionSignalNormalizationStatus.REJECTED
            and self.timing_input is not None
        ):
            raise ValueError("REJECTED requires timing_input=None")
