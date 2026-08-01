# © Artur Czarnecki. All rights reserved.

"""Cache-aware runtime composition contracts (TOKEN-10D-3)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.runtime.token_optimization.cache_aware_orchestration_contracts import (
    CacheAwareTokenOptimizationOrchestrationResult,
    CacheAwareTokenOptimizationOrchestrationStatus,
)
from intergrax.runtime.token_optimization.cache_signal_normalization_contracts import (
    CacheAwareCompactionSignalNormalizationRequest,
    CacheAwareCompactionSignalNormalizationResult,
    CacheAwareCompactionSignalNormalizationStatus,
)
from intergrax.runtime.token_optimization.contracts import (
    CacheAwareCompactionTarget,
    PromptCacheAttribution,
)
from intergrax.runtime.token_optimization.llm_router_contracts import (
    TokenOptimizationLLMRouterRequest,
)


class CacheAwareTokenOptimizationRuntimeStatus(StrEnum):
    """Outcome of cache-aware runtime composition."""

    SIGNALS_REJECTED = "signals_rejected"
    EXECUTED = "executed"
    DEFERRED = "deferred"
    BYPASSED = "bypassed"
    REVIEW_REQUIRED = "review_required"
    ROUTER_TERMINAL = "router_terminal"


class CacheAwareTokenOptimizationEvidenceReconciliationReason(StrEnum):
    """Deterministic reason for cache evidence reconciliation."""

    ATTRIBUTION_ONLY = "attribution_only"
    ADAPTER_FILLED_MISSING_USAGE = "adapter_filled_missing_usage"
    IDENTICAL_EVIDENCE = "identical_evidence"
    COMPLEMENTARY_MERGE = "complementary_merge"
    CONFLICTING_CACHE_EVIDENCE = "conflicting_cache_evidence"
    PROVIDER_MISMATCH = "provider_mismatch"
    MODEL_MISMATCH = "model_mismatch"
    REQUEST_ATTRIBUTION_MISMATCH = "request_attribution_mismatch"
    EXTRACTION_ERROR = "extraction_error"


@dataclass(frozen=True, slots=True)
class CacheAwareTokenOptimizationRuntimeRequest:
    """Caller-supplied cache-aware runtime inputs (no raw content duplication)."""

    router_request: TokenOptimizationLLMRouterRequest
    cache_attribution: PromptCacheAttribution
    target: CacheAwareCompactionTarget
    adapter_response: LLMAdapterResponse | None = None
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
        if self.adapter_response is not None:
            provider = self.adapter_response.provider
            if provider is not None and not provider.strip():
                raise ValueError("adapter_response.provider cannot be empty when explicit")


@dataclass(frozen=True, slots=True)
class CacheAwareTokenOptimizationRuntimeResult:
    """Cache-aware runtime composition outcome (no raw content)."""

    status: CacheAwareTokenOptimizationRuntimeStatus
    normalization_result: CacheAwareCompactionSignalNormalizationResult | None
    orchestration_result: CacheAwareTokenOptimizationOrchestrationResult | None
    reconciled_cache_attribution: PromptCacheAttribution
    adapter_cache_evidence_reported: bool
    evidence_reconciliation_reason: CacheAwareTokenOptimizationEvidenceReconciliationReason | None
    executed: bool
    review_required: bool
    raw_content_included: bool = False

    def __post_init__(self) -> None:
        if self.raw_content_included:
            raise ValueError("raw_content_included must remain False")

        status = self.status
        normalization = self.normalization_result
        orchestration = self.orchestration_result

        if status is CacheAwareTokenOptimizationRuntimeStatus.SIGNALS_REJECTED:
            if orchestration is not None:
                raise ValueError("SIGNALS_REJECTED requires orchestration_result=None")
            if self.executed:
                raise ValueError("SIGNALS_REJECTED requires executed=False")
            if self.review_required:
                raise ValueError("SIGNALS_REJECTED requires review_required=False")
            return

        if normalization is None:
            raise ValueError(f"{status.value} requires normalization_result")
        if (
            normalization.status
            is CacheAwareCompactionSignalNormalizationStatus.REJECTED
        ):
            raise ValueError(f"{status.value} requires normalization status != REJECTED")
        if orchestration is None:
            raise ValueError(f"{status.value} requires orchestration_result")

        if status is CacheAwareTokenOptimizationRuntimeStatus.EXECUTED:
            if normalization.status not in {
                CacheAwareCompactionSignalNormalizationStatus.NORMALIZED,
                CacheAwareCompactionSignalNormalizationStatus.PARTIAL,
            }:
                raise ValueError("EXECUTED requires normalization status NORMALIZED or PARTIAL")
            if (
                orchestration.orchestration_status
                is not CacheAwareTokenOptimizationOrchestrationStatus.EXECUTED
            ):
                raise ValueError("EXECUTED requires orchestration status EXECUTED")
            if orchestration.pipeline_result is None:
                raise ValueError("EXECUTED requires pipeline_result")
            if not self.executed:
                raise ValueError("EXECUTED requires executed=True")
            if self.review_required:
                raise ValueError("EXECUTED requires review_required=False")
            return

        if status is CacheAwareTokenOptimizationRuntimeStatus.DEFERRED:
            if (
                orchestration.orchestration_status
                is not CacheAwareTokenOptimizationOrchestrationStatus.DEFERRED
            ):
                raise ValueError("DEFERRED requires orchestration status DEFERRED")
            if self.executed:
                raise ValueError("DEFERRED requires executed=False")
            if self.review_required:
                raise ValueError("DEFERRED requires review_required=False")
            return

        if status is CacheAwareTokenOptimizationRuntimeStatus.BYPASSED:
            if (
                orchestration.orchestration_status
                is not CacheAwareTokenOptimizationOrchestrationStatus.BYPASSED
            ):
                raise ValueError("BYPASSED requires orchestration status BYPASSED")
            if self.executed:
                raise ValueError("BYPASSED requires executed=False")
            if self.review_required:
                raise ValueError("BYPASSED requires review_required=False")
            return

        if status is CacheAwareTokenOptimizationRuntimeStatus.REVIEW_REQUIRED:
            if (
                orchestration.orchestration_status
                is not CacheAwareTokenOptimizationOrchestrationStatus.REVIEW_REQUIRED
            ):
                raise ValueError("REVIEW_REQUIRED requires orchestration status REVIEW_REQUIRED")
            if self.executed:
                raise ValueError("REVIEW_REQUIRED requires executed=False")
            if not self.review_required:
                raise ValueError("REVIEW_REQUIRED requires review_required=True")
            return

        if status is CacheAwareTokenOptimizationRuntimeStatus.ROUTER_TERMINAL:
            if (
                orchestration.orchestration_status
                is not CacheAwareTokenOptimizationOrchestrationStatus.ROUTER_TERMINAL
            ):
                raise ValueError("ROUTER_TERMINAL requires orchestration status ROUTER_TERMINAL")
            if self.executed:
                raise ValueError("ROUTER_TERMINAL requires executed=False")
            if self.review_required:
                raise ValueError("ROUTER_TERMINAL requires review_required=False")
