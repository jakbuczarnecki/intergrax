# © Artur Czarnecki. All rights reserved.

"""Cache-aware orchestration gate between router selection and pipeline execution (TOKEN-10D-1)."""

from __future__ import annotations

from collections.abc import Mapping

from intergrax.runtime.token_optimization.cache_aware_orchestration_contracts import (
    CacheAwareTokenOptimizationOrchestrationRequest,
    CacheAwareTokenOptimizationOrchestrationResult,
    CacheAwareTokenOptimizationOrchestrationStatus,
)
from intergrax.runtime.token_optimization.contracts import CacheAwareCompactionDecision
from intergrax.runtime.token_optimization.llm_router import TokenOptimizationLLMRouter
from intergrax.runtime.token_optimization.llm_router_contracts import (
    TokenOptimizationRouterStatus,
)
from intergrax.runtime.token_optimization.prompt_cache import decide_cache_aware_compaction_timing


class CacheAwareTokenOptimizationOrchestrator:
    """Route configuration selection, evaluate cache-aware timing, execute pipeline on RUN."""

    def __init__(self, router: TokenOptimizationLLMRouter) -> None:
        self._router = router

    def orchestrate(
        self,
        request: CacheAwareTokenOptimizationOrchestrationRequest,
    ) -> CacheAwareTokenOptimizationOrchestrationResult:
        router_result = self._router.route(request.router_request)

        if router_result.status is not TokenOptimizationRouterStatus.ROUTED:
            return CacheAwareTokenOptimizationOrchestrationResult(
                router_result=router_result,
                timing_decision=None,
                orchestration_status=CacheAwareTokenOptimizationOrchestrationStatus.ROUTER_TERMINAL,
                pipeline_result=None,
                executed=False,
                review_required=False,
            )

        timing_decision = decide_cache_aware_compaction_timing(request.timing_input)
        decision = timing_decision.decision

        if decision is CacheAwareCompactionDecision.RUN:
            executed_router = self._router.execute_routed(
                request.router_request,
                router_result,
            )
            return CacheAwareTokenOptimizationOrchestrationResult(
                router_result=executed_router,
                timing_decision=timing_decision,
                orchestration_status=CacheAwareTokenOptimizationOrchestrationStatus.EXECUTED,
                pipeline_result=executed_router.pipeline_result,
                executed=True,
                review_required=False,
            )

        if decision is CacheAwareCompactionDecision.DEFER:
            return CacheAwareTokenOptimizationOrchestrationResult(
                router_result=router_result,
                timing_decision=timing_decision,
                orchestration_status=CacheAwareTokenOptimizationOrchestrationStatus.DEFERRED,
                pipeline_result=None,
                executed=False,
                review_required=False,
            )

        if decision is CacheAwareCompactionDecision.BYPASS:
            return CacheAwareTokenOptimizationOrchestrationResult(
                router_result=router_result,
                timing_decision=timing_decision,
                orchestration_status=CacheAwareTokenOptimizationOrchestrationStatus.BYPASSED,
                pipeline_result=None,
                executed=False,
                review_required=False,
            )

        return CacheAwareTokenOptimizationOrchestrationResult(
            router_result=router_result,
            timing_decision=timing_decision,
            orchestration_status=CacheAwareTokenOptimizationOrchestrationStatus.REVIEW_REQUIRED,
            pipeline_result=None,
            executed=False,
            review_required=True,
        )


_ALLOWED_REPORT_FIELDS = frozenset(
    {
        "request_id",
        "router_status",
        "router_reason",
        "configuration_id",
        "orchestration_status",
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


def _safe_string_list(value: object) -> list[str] | None:
    if not isinstance(value, (list, tuple)):
        return None
    normalized: list[str] = []
    for item in value:
        if not isinstance(item, str):
            return None
        normalized.append(item)
    return normalized


def _safe_bool(value: object) -> bool | None:
    if type(value) is not bool:
        return None
    return value


def _safe_receipt_fields(
    receipt: Mapping[str, object],
) -> tuple[list[str], bool, str | None] | None:
    executed_layer_ids = _safe_string_list(receipt.get("executed_layer_ids"))
    completed = _safe_bool(receipt.get("completed"))
    failure_id_raw = receipt.get("required_failure_layer_id")
    if failure_id_raw is None:
        required_failure_layer_id: str | None = None
    elif isinstance(failure_id_raw, str):
        required_failure_layer_id = failure_id_raw
    else:
        return None
    if executed_layer_ids is None or completed is None:
        return None
    return executed_layer_ids, completed, required_failure_layer_id


def cache_aware_orchestration_result_to_safe_dict(
    result: CacheAwareTokenOptimizationOrchestrationResult,
) -> dict[str, object]:
    """Serialize orchestration outcome without raw content or LLM payloads."""
    router = result.router_result
    payload: dict[str, object] = {
        "request_id": router.request_id,
        "router_status": router.status.value,
        "router_reason": router.reason.value if router.reason is not None else None,
        "configuration_id": (
            router.configuration_id.value if router.configuration_id is not None else None
        ),
        "orchestration_status": result.orchestration_status.value,
        "timing_decision": (
            result.timing_decision.decision.value if result.timing_decision is not None else None
        ),
        "timing_reason": (
            result.timing_decision.reason.value if result.timing_decision is not None else None
        ),
        "timing_target": (
            result.timing_decision.target.value if result.timing_decision is not None else None
        ),
        "cache_hot": (
            result.timing_decision.cache_hot if result.timing_decision is not None else None
        ),
        "ttl_seconds_remaining": (
            result.timing_decision.ttl_seconds_remaining
            if result.timing_decision is not None
            else None
        ),
        "estimated_content_reduction_chars": (
            result.timing_decision.estimated_content_reduction_chars
            if result.timing_decision is not None
            else None
        ),
        "estimated_cache_invalidation_cost_tokens": (
            result.timing_decision.estimated_cache_invalidation_cost_tokens
            if result.timing_decision is not None
            else None
        ),
        "executed": result.executed,
        "review_required": result.review_required,
        "raw_content_included": False,
    }

    pipeline_result = result.pipeline_result
    if pipeline_result is not None:
        receipt_fields = _safe_receipt_fields(pipeline_result.receipt_metadata)
        if receipt_fields is None:
            completed = False
        else:
            _, completed, _ = receipt_fields

        payload["pipeline_id"] = pipeline_result.pipeline_id
        payload["applied_layer_ids"] = list(pipeline_result.applied_layer_ids)
        payload["bypassed_layer_ids"] = list(pipeline_result.bypassed_layer_ids)
        payload["failed_layer_ids"] = list(pipeline_result.failed_layer_ids)
        payload["fallback_used"] = pipeline_result.fallback_used
        payload["completed"] = completed
    else:
        payload["pipeline_id"] = None
        payload["applied_layer_ids"] = []
        payload["bypassed_layer_ids"] = []
        payload["failed_layer_ids"] = []
        payload["fallback_used"] = False
        payload["completed"] = False

    for key in payload:
        if key not in _ALLOWED_REPORT_FIELDS:
            raise ValueError(f"unexpected report field: {key}")
    return payload
