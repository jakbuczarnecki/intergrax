# © Artur Czarnecki. All rights reserved.

"""Cache-aware runtime composition entrypoint (TOKEN-10D-3)."""

from __future__ import annotations

from dataclasses import dataclass, replace

from intergrax.runtime.token_optimization.cache_aware_orchestration import (
    CacheAwareTokenOptimizationOrchestrator,
    cache_aware_orchestration_result_to_safe_dict,
)
from intergrax.runtime.token_optimization.cache_aware_orchestration_contracts import (
    CacheAwareTokenOptimizationOrchestrationRequest,
    CacheAwareTokenOptimizationOrchestrationStatus,
)
from intergrax.runtime.token_optimization.cache_aware_runtime_contracts import (
    CacheAwareTokenOptimizationEvidenceReconciliationReason,
    CacheAwareTokenOptimizationRuntimeRequest,
    CacheAwareTokenOptimizationRuntimeResult,
    CacheAwareTokenOptimizationRuntimeStatus,
)
from intergrax.runtime.token_optimization.cache_signal_normalization import (
    cache_signal_normalization_result_to_safe_dict,
    normalize_cache_aware_compaction_signals,
    prompt_cache_usage_snapshot_from_adapter_response,
)
from intergrax.runtime.token_optimization.cache_signal_normalization_contracts import (
    CacheAwareCompactionSignalNormalizationRequest,
    CacheAwareCompactionSignalNormalizationStatus,
)
from intergrax.runtime.token_optimization.contracts import (
    PromptCacheAttribution,
    PromptCacheUsageSnapshot,
    TokenOptimizationAttribution,
)

_USAGE_COMPARABLE_FIELDS = (
    "cache_read_tokens",
    "cache_creation_tokens",
    "cached_input_tokens",
    "uncached_input_tokens",
    "cache_hit_ratio",
)


class CacheAwareTokenOptimizationRuntime:
    """Compose cache evidence reconciliation, normalization, and orchestration."""

    def __init__(
        self,
        *,
        orchestrator: CacheAwareTokenOptimizationOrchestrator,
    ) -> None:
        self._orchestrator = orchestrator

    def run(
        self,
        request: CacheAwareTokenOptimizationRuntimeRequest,
    ) -> CacheAwareTokenOptimizationRuntimeResult:
        adapter_snapshot: PromptCacheUsageSnapshot | None = None
        adapter_cache_evidence_reported = False

        if request.adapter_response is not None:
            try:
                adapter_snapshot = prompt_cache_usage_snapshot_from_adapter_response(
                    request.adapter_response
                )
            except ValueError:
                return _signals_rejected(
                    cache_attribution=request.cache_attribution,
                    adapter_cache_evidence_reported=False,
                    evidence_reconciliation_reason=(
                        CacheAwareTokenOptimizationEvidenceReconciliationReason.EXTRACTION_ERROR
                    ),
                )
            adapter_cache_evidence_reported = adapter_snapshot is not None

        reconciliation = _reconcile_cache_evidence(
            cache_attribution=request.cache_attribution,
            adapter_response=request.adapter_response,
            adapter_snapshot=adapter_snapshot,
            router_attribution=request.router_request.request.attribution,
        )
        if reconciliation.rejection_reason is not None:
            return _signals_rejected(
                cache_attribution=request.cache_attribution,
                adapter_cache_evidence_reported=adapter_cache_evidence_reported,
                evidence_reconciliation_reason=reconciliation.rejection_reason,
            )

        normalization_request = CacheAwareCompactionSignalNormalizationRequest(
            target=request.target,
            cache_attribution=reconciliation.reconciled_attribution,
            ttl_seconds_remaining=request.ttl_seconds_remaining,
            near_expiry_threshold_seconds=request.near_expiry_threshold_seconds,
            estimated_content_reduction_chars=request.estimated_content_reduction_chars,
            protected_or_semantic_risk=request.protected_or_semantic_risk,
            dynamic_tail_reduction_available=request.dynamic_tail_reduction_available,
        )
        normalization_result = normalize_cache_aware_compaction_signals(
            normalization_request
        )

        if (
            normalization_result.status
            is CacheAwareCompactionSignalNormalizationStatus.REJECTED
        ):
            return CacheAwareTokenOptimizationRuntimeResult(
                status=CacheAwareTokenOptimizationRuntimeStatus.SIGNALS_REJECTED,
                normalization_result=normalization_result,
                orchestration_result=None,
                reconciled_cache_attribution=reconciliation.reconciled_attribution,
                adapter_cache_evidence_reported=adapter_cache_evidence_reported,
                evidence_reconciliation_reason=reconciliation.success_reason,
                executed=False,
                review_required=False,
                raw_content_included=False,
            )

        orchestration_result = self._orchestrator.orchestrate(
            CacheAwareTokenOptimizationOrchestrationRequest(
                router_request=request.router_request,
                timing_input=normalization_result.timing_input,
            )
        )
        runtime_status = _map_orchestration_status(
            orchestration_result.orchestration_status
        )

        return CacheAwareTokenOptimizationRuntimeResult(
            status=runtime_status,
            normalization_result=normalization_result,
            orchestration_result=orchestration_result,
            reconciled_cache_attribution=reconciliation.reconciled_attribution,
            adapter_cache_evidence_reported=adapter_cache_evidence_reported,
            evidence_reconciliation_reason=reconciliation.success_reason,
            executed=orchestration_result.executed,
            review_required=orchestration_result.review_required,
            raw_content_included=False,
        )


_ALLOWED_REPORT_FIELDS = frozenset(
    {
        "request_id",
        "runtime_status",
        "evidence_reconciliation_reason",
        "adapter_cache_evidence_reported",
        "normalization_status",
        "normalization_reason_codes",
        "orchestration_status",
        "router_status",
        "router_reason",
        "configuration_id",
        "timing_decision",
        "timing_reason",
        "timing_target",
        "cache_hot",
        "ttl_seconds_remaining",
        "estimated_content_reduction_chars",
        "estimated_cache_invalidation_cost_tokens",
        "executed",
        "review_required",
        "pipeline_id",
        "applied_layer_ids",
        "bypassed_layer_ids",
        "failed_layer_ids",
        "fallback_used",
        "completed",
        "raw_content_included",
    }
)


def cache_aware_runtime_result_to_safe_dict(
    result: CacheAwareTokenOptimizationRuntimeResult,
) -> dict[str, object]:
    """Serialize runtime outcome using an allowlist (no raw content)."""
    request_id: str | None = None
    orchestration_status: str | None = None
    router_status: str | None = None
    router_reason: str | None = None
    configuration_id: str | None = None
    timing_decision: str | None = None
    timing_reason: str | None = None
    timing_target: str | None = None
    cache_hot: bool | None = None
    ttl_seconds_remaining: int | None = None
    estimated_content_reduction_chars: int | None = None
    estimated_cache_invalidation_cost_tokens: int | None = None
    pipeline_id: str | None = None
    applied_layer_ids: list[str] = []
    bypassed_layer_ids: list[str] = []
    failed_layer_ids: list[str] = []
    fallback_used = False
    completed = False

    orchestration = result.orchestration_result
    if orchestration is not None:
        orchestration_payload = cache_aware_orchestration_result_to_safe_dict(
            orchestration
        )
        request_id = _as_optional_str(orchestration_payload.get("request_id"))
        orchestration_status = _as_optional_str(
            orchestration_payload.get("orchestration_status")
        )
        router_status = _as_optional_str(orchestration_payload.get("router_status"))
        router_reason = _as_optional_str(orchestration_payload.get("router_reason"))
        configuration_id = _as_optional_str(orchestration_payload.get("configuration_id"))
        timing_decision = _as_optional_str(orchestration_payload.get("timing_decision"))
        timing_reason = _as_optional_str(orchestration_payload.get("timing_reason"))
        timing_target = _as_optional_str(orchestration_payload.get("timing_target"))
        cache_hot = orchestration_payload.get("cache_hot")
        ttl_seconds_remaining = orchestration_payload.get("ttl_seconds_remaining")
        estimated_content_reduction_chars = orchestration_payload.get(
            "estimated_content_reduction_chars"
        )
        estimated_cache_invalidation_cost_tokens = orchestration_payload.get(
            "estimated_cache_invalidation_cost_tokens"
        )
        pipeline_id = _as_optional_str(orchestration_payload.get("pipeline_id"))
        applied_layer_ids = _as_string_list(orchestration_payload.get("applied_layer_ids"))
        bypassed_layer_ids = _as_string_list(orchestration_payload.get("bypassed_layer_ids"))
        failed_layer_ids = _as_string_list(orchestration_payload.get("failed_layer_ids"))
        fallback_used = bool(orchestration_payload.get("fallback_used"))
        completed = bool(orchestration_payload.get("completed"))

    normalization_status: str | None = None
    normalization_reason_codes: list[str] = []
    if result.normalization_result is not None:
        normalization_payload = cache_signal_normalization_result_to_safe_dict(
            result.normalization_result
        )
        normalization_status = _as_optional_str(normalization_payload.get("status"))
        reason_codes = normalization_payload.get("reason_codes")
        normalization_reason_codes = _as_string_list(reason_codes)
        if request_id is None:
            request_id = None
        if cache_hot is None:
            cache_hot = normalization_payload.get("cache_hot")
        if ttl_seconds_remaining is None:
            ttl_seconds_remaining = normalization_payload.get("ttl_seconds_remaining")
        if estimated_content_reduction_chars is None:
            estimated_content_reduction_chars = normalization_payload.get(
                "estimated_content_reduction_chars"
            )
        if estimated_cache_invalidation_cost_tokens is None:
            estimated_cache_invalidation_cost_tokens = normalization_payload.get(
                "estimated_cache_invalidation_cost_tokens"
            )

    payload: dict[str, object] = {
        "request_id": request_id,
        "runtime_status": result.status.value,
        "evidence_reconciliation_reason": (
            result.evidence_reconciliation_reason.value
            if result.evidence_reconciliation_reason is not None
            else None
        ),
        "adapter_cache_evidence_reported": result.adapter_cache_evidence_reported,
        "normalization_status": normalization_status,
        "normalization_reason_codes": normalization_reason_codes,
        "orchestration_status": orchestration_status,
        "router_status": router_status,
        "router_reason": router_reason,
        "configuration_id": configuration_id,
        "timing_decision": timing_decision,
        "timing_reason": timing_reason,
        "timing_target": timing_target,
        "cache_hot": cache_hot,
        "ttl_seconds_remaining": ttl_seconds_remaining,
        "estimated_content_reduction_chars": estimated_content_reduction_chars,
        "estimated_cache_invalidation_cost_tokens": estimated_cache_invalidation_cost_tokens,
        "executed": result.executed,
        "review_required": result.review_required,
        "pipeline_id": pipeline_id,
        "applied_layer_ids": applied_layer_ids,
        "bypassed_layer_ids": bypassed_layer_ids,
        "failed_layer_ids": failed_layer_ids,
        "fallback_used": fallback_used,
        "completed": completed,
        "raw_content_included": False,
    }

    for key in payload:
        if key not in _ALLOWED_REPORT_FIELDS:
            raise ValueError(f"unexpected report field: {key}")
    return payload


def _signals_rejected(
    *,
    cache_attribution: PromptCacheAttribution,
    adapter_cache_evidence_reported: bool,
    evidence_reconciliation_reason: CacheAwareTokenOptimizationEvidenceReconciliationReason,
) -> CacheAwareTokenOptimizationRuntimeResult:
    return CacheAwareTokenOptimizationRuntimeResult(
        status=CacheAwareTokenOptimizationRuntimeStatus.SIGNALS_REJECTED,
        normalization_result=None,
        orchestration_result=None,
        reconciled_cache_attribution=cache_attribution,
        adapter_cache_evidence_reported=adapter_cache_evidence_reported,
        evidence_reconciliation_reason=evidence_reconciliation_reason,
        executed=False,
        review_required=False,
        raw_content_included=False,
    )


def _map_orchestration_status(
    status: CacheAwareTokenOptimizationOrchestrationStatus,
) -> CacheAwareTokenOptimizationRuntimeStatus:
    mapping = {
        CacheAwareTokenOptimizationOrchestrationStatus.EXECUTED: (
            CacheAwareTokenOptimizationRuntimeStatus.EXECUTED
        ),
        CacheAwareTokenOptimizationOrchestrationStatus.DEFERRED: (
            CacheAwareTokenOptimizationRuntimeStatus.DEFERRED
        ),
        CacheAwareTokenOptimizationOrchestrationStatus.BYPASSED: (
            CacheAwareTokenOptimizationRuntimeStatus.BYPASSED
        ),
        CacheAwareTokenOptimizationOrchestrationStatus.REVIEW_REQUIRED: (
            CacheAwareTokenOptimizationRuntimeStatus.REVIEW_REQUIRED
        ),
        CacheAwareTokenOptimizationOrchestrationStatus.ROUTER_TERMINAL: (
            CacheAwareTokenOptimizationRuntimeStatus.ROUTER_TERMINAL
        ),
    }
    return mapping[status]


@dataclass(frozen=True, slots=True)
class _ReconciliationOutcome:
    reconciled_attribution: PromptCacheAttribution
    success_reason: CacheAwareTokenOptimizationEvidenceReconciliationReason | None
    rejection_reason: CacheAwareTokenOptimizationEvidenceReconciliationReason | None


def _reconcile_cache_evidence(
    *,
    cache_attribution: PromptCacheAttribution,
    adapter_response: object | None,
    adapter_snapshot: PromptCacheUsageSnapshot | None,
    router_attribution: TokenOptimizationAttribution | None,
) -> _ReconciliationOutcome:
    if adapter_response is None:
        rejection = _check_identity_mismatches(
            cache_attribution=cache_attribution,
            adapter_provider=None,
            adapter_model=None,
            usage=cache_attribution.usage,
            router_attribution=router_attribution,
        )
        if rejection is not None:
            return _ReconciliationOutcome(
                reconciled_attribution=cache_attribution,
                success_reason=None,
                rejection_reason=rejection,
            )
        return _ReconciliationOutcome(
            reconciled_attribution=cache_attribution,
            success_reason=CacheAwareTokenOptimizationEvidenceReconciliationReason.ATTRIBUTION_ONLY,
            rejection_reason=None,
        )

    from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse

    response = adapter_response
    assert isinstance(response, LLMAdapterResponse)

    attribution_usage = cache_attribution.usage
    if attribution_usage is None:
        if adapter_snapshot is None:
            rejection = _check_identity_mismatches(
                cache_attribution=cache_attribution,
                adapter_provider=response.provider,
                adapter_model=response.model,
                usage=None,
                router_attribution=router_attribution,
            )
            if rejection is not None:
                return _ReconciliationOutcome(
                    reconciled_attribution=cache_attribution,
                    success_reason=None,
                    rejection_reason=rejection,
                )
            return _ReconciliationOutcome(
                reconciled_attribution=cache_attribution,
                success_reason=CacheAwareTokenOptimizationEvidenceReconciliationReason.ATTRIBUTION_ONLY,
                rejection_reason=None,
            )

        rejection = _check_identity_mismatches(
            cache_attribution=cache_attribution,
            adapter_provider=response.provider,
            adapter_model=response.model,
            usage=adapter_snapshot,
            router_attribution=router_attribution,
        )
        if rejection is not None:
            return _ReconciliationOutcome(
                reconciled_attribution=cache_attribution,
                success_reason=None,
                rejection_reason=rejection,
            )
        reconciled = replace(cache_attribution, usage=adapter_snapshot)
        return _ReconciliationOutcome(
            reconciled_attribution=reconciled,
            success_reason=(
                CacheAwareTokenOptimizationEvidenceReconciliationReason.ADAPTER_FILLED_MISSING_USAGE
            ),
            rejection_reason=None,
        )

    if adapter_snapshot is None:
        rejection = _check_identity_mismatches(
            cache_attribution=cache_attribution,
            adapter_provider=response.provider,
            adapter_model=response.model,
            usage=attribution_usage,
            router_attribution=router_attribution,
        )
        if rejection is not None:
            return _ReconciliationOutcome(
                reconciled_attribution=cache_attribution,
                success_reason=None,
                rejection_reason=rejection,
            )
        return _ReconciliationOutcome(
            reconciled_attribution=cache_attribution,
            success_reason=CacheAwareTokenOptimizationEvidenceReconciliationReason.ATTRIBUTION_ONLY,
            rejection_reason=None,
        )

    rejection = _check_identity_mismatches(
        cache_attribution=cache_attribution,
        adapter_provider=response.provider,
        adapter_model=response.model,
        usage=attribution_usage,
        router_attribution=router_attribution,
    )
    if rejection is not None:
        return _ReconciliationOutcome(
            reconciled_attribution=cache_attribution,
            success_reason=None,
            rejection_reason=rejection,
        )

    if _usage_values_conflict(attribution_usage, adapter_snapshot):
        return _ReconciliationOutcome(
            reconciled_attribution=cache_attribution,
            success_reason=None,
            rejection_reason=(
                CacheAwareTokenOptimizationEvidenceReconciliationReason.CONFLICTING_CACHE_EVIDENCE
            ),
        )

    if _usage_snapshots_identical(attribution_usage, adapter_snapshot):
        return _ReconciliationOutcome(
            reconciled_attribution=cache_attribution,
            success_reason=CacheAwareTokenOptimizationEvidenceReconciliationReason.IDENTICAL_EVIDENCE,
            rejection_reason=None,
        )

    merged_usage = _merge_complementary_usage(attribution_usage, adapter_snapshot)
    reconciled = replace(cache_attribution, usage=merged_usage)
    return _ReconciliationOutcome(
        reconciled_attribution=reconciled,
        success_reason=CacheAwareTokenOptimizationEvidenceReconciliationReason.COMPLEMENTARY_MERGE,
        rejection_reason=None,
    )


def _check_identity_mismatches(
    *,
    cache_attribution: PromptCacheAttribution,
    adapter_provider: str | None,
    adapter_model: str | None,
    usage: PromptCacheUsageSnapshot | None,
    router_attribution: TokenOptimizationAttribution | None,
) -> CacheAwareTokenOptimizationEvidenceReconciliationReason | None:
    providers: list[str] = []
    if adapter_provider is not None:
        providers.append(adapter_provider)
    if usage is not None:
        providers.append(usage.provider)
    capabilities = cache_attribution.provider_capabilities
    if capabilities is not None:
        providers.append(capabilities.provider)
    if router_attribution is not None and router_attribution.provider is not None:
        providers.append(router_attribution.provider)

    if len(set(providers)) > 1:
        return CacheAwareTokenOptimizationEvidenceReconciliationReason.PROVIDER_MISMATCH

    models: list[str] = []
    if adapter_model is not None:
        models.append(adapter_model)
    if usage is not None and usage.model is not None:
        models.append(usage.model)
    if router_attribution is not None and router_attribution.model is not None:
        models.append(router_attribution.model)

    if len(set(models)) > 1:
        return CacheAwareTokenOptimizationEvidenceReconciliationReason.MODEL_MISMATCH

    if router_attribution is not None:
        evidence_providers = set(providers)
        evidence_models = set(models)
        if (
            router_attribution.provider is not None
            and evidence_providers
            and router_attribution.provider not in evidence_providers
        ):
            return (
                CacheAwareTokenOptimizationEvidenceReconciliationReason.REQUEST_ATTRIBUTION_MISMATCH
            )
        if (
            router_attribution.model is not None
            and evidence_models
            and router_attribution.model not in evidence_models
        ):
            return (
                CacheAwareTokenOptimizationEvidenceReconciliationReason.REQUEST_ATTRIBUTION_MISMATCH
            )

    return None


def _usage_values_conflict(
    left: PromptCacheUsageSnapshot,
    right: PromptCacheUsageSnapshot,
) -> bool:
    for field_name in _USAGE_COMPARABLE_FIELDS:
        left_value = object.__getattribute__(left, field_name)
        right_value = object.__getattribute__(right, field_name)
        if left_value is None or right_value is None:
            continue
        if left_value != right_value:
            return True
    return False


def _usage_snapshots_identical(
    left: PromptCacheUsageSnapshot,
    right: PromptCacheUsageSnapshot,
) -> bool:
    for field_name in _USAGE_COMPARABLE_FIELDS:
        if object.__getattribute__(left, field_name) != object.__getattribute__(right, field_name):
            return False
    return True


def _merge_complementary_usage(
    attribution_usage: PromptCacheUsageSnapshot,
    adapter_usage: PromptCacheUsageSnapshot,
) -> PromptCacheUsageSnapshot:
    return PromptCacheUsageSnapshot(
        provider=attribution_usage.provider,
        model=(
            attribution_usage.model
            if attribution_usage.model is not None
            else adapter_usage.model
        ),
        cache_read_tokens=(
            attribution_usage.cache_read_tokens
            if attribution_usage.cache_read_tokens is not None
            else adapter_usage.cache_read_tokens
        ),
        cache_creation_tokens=(
            attribution_usage.cache_creation_tokens
            if attribution_usage.cache_creation_tokens is not None
            else adapter_usage.cache_creation_tokens
        ),
        cached_input_tokens=(
            attribution_usage.cached_input_tokens
            if attribution_usage.cached_input_tokens is not None
            else adapter_usage.cached_input_tokens
        ),
        uncached_input_tokens=(
            attribution_usage.uncached_input_tokens
            if attribution_usage.uncached_input_tokens is not None
            else adapter_usage.uncached_input_tokens
        ),
        cache_hit_ratio=(
            attribution_usage.cache_hit_ratio
            if attribution_usage.cache_hit_ratio is not None
            else adapter_usage.cache_hit_ratio
        ),
        cache_latency_delta_estimate_ms=attribution_usage.cache_latency_delta_estimate_ms,
        cache_discount_estimate=attribution_usage.cache_discount_estimate,
    )


def _as_optional_str(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    return value


def _as_string_list(value: object) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    normalized: list[str] = []
    for item in value:
        if isinstance(item, str):
            normalized.append(item)
    return normalized
