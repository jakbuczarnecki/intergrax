# © Artur Czarnecki. All rights reserved.

"""Unit tests for cache-aware orchestration gate (TOKEN-10D-1)."""

from __future__ import annotations

import json
from typing import Any, Sequence
from unittest.mock import patch

import pytest

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters._shared.adapter_response_builders import build_adapter_response
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.contracts.structured_result import LLMStructuredResult
from intergrax.llm_adapters.contracts.tool_call import LLMToolCall
from intergrax.runtime.token_optimization.cache_aware_orchestration import (
    CacheAwareTokenOptimizationOrchestrator,
    cache_aware_orchestration_result_to_safe_dict,
)
from intergrax.runtime.token_optimization.cache_aware_orchestration_contracts import (
    CacheAwareTokenOptimizationOrchestrationRequest,
    CacheAwareTokenOptimizationOrchestrationResult,
    CacheAwareTokenOptimizationOrchestrationStatus,
)
from intergrax.runtime.token_optimization.contracts import (
    CacheAwareCompactionDecision,
    CacheAwareCompactionReason,
    CacheAwareCompactionTarget,
    CacheAwareCompactionTimingDecision,
    CacheAwareCompactionTimingInput,
    PromptCacheInvalidationReason,
    TokenOptimizationPolicy,
    TokenOptimizationProfile,
    TokenOptimizationRequest,
    TokenOptimizationSourceType,
)
from intergrax.runtime.token_optimization.llm_router import (
    ROUTER_TOOL_ID,
    TokenOptimizationLLMRouter,
)
from intergrax.runtime.token_optimization.llm_router_contracts import (
    TokenOptimizationLLMRouterPolicy,
    TokenOptimizationLLMRouterRequest,
    TokenOptimizationLLMRouterResult,
    TokenOptimizationRouterConfigurationId,
    TokenOptimizationRouterReason,
    TokenOptimizationRouterReasonCode,
    TokenOptimizationRouterRisk,
    TokenOptimizationRouterStatus,
    TokenOptimizationRouterToolInput,
    TokenOptimizationRouterTransport,
)
from intergrax.runtime.token_optimization.pipeline import TokenOptimizationPipelineRunner
from intergrax.runtime.token_optimization.prompt_cache import (
    PREFIX_STABILITY_INVALIDATED,
    PREFIX_STABILITY_STABLE,
)
from tests.fixtures.token_optimization.cache_aware_compaction_corpus import (
    CACHE_AWARE_COMPACTION_CORPUS,
)

pytestmark = pytest.mark.unit


def _decision(
    configuration_id: TokenOptimizationRouterConfigurationId,
    *,
    review_required: bool = False,
    confidence: float = 0.9,
    reason_code: TokenOptimizationRouterReasonCode = TokenOptimizationRouterReasonCode.EXACT_DUPLICATES,
    risk: TokenOptimizationRouterRisk = TokenOptimizationRouterRisk.LOW,
) -> TokenOptimizationRouterToolInput:
    return TokenOptimizationRouterToolInput(
        configuration_id=configuration_id,
        reason_code=reason_code,
        risk=risk,
        review_required=review_required,
        confidence=confidence,
    )


def _router_request(
    *,
    content: str = "SYNTH-ALPHA\nSYNTH-ALPHA\n",
    request_id: str = "orchestration-test-1",
) -> TokenOptimizationLLMRouterRequest:
    return TokenOptimizationLLMRouterRequest(
        request=TokenOptimizationRequest(
            content=content,
            source_type=TokenOptimizationSourceType.RAG_CONTEXT_PACK,
            policy=TokenOptimizationPolicy(
                enabled=True,
                profile=TokenOptimizationProfile.CONSERVATIVE,
                allow_lossy=True,
            ),
        ),
        policy=TokenOptimizationLLMRouterPolicy(),
        request_id=request_id,
    )


class _NativeToolsAdapter(LLMAdapter):
    provider = "fake-native"
    model = "fake-native"

    def __init__(
        self,
        *,
        decision: TokenOptimizationRouterToolInput | None = None,
    ) -> None:
        super().__init__()
        self._decision = decision
        self.route_calls = 0

    @property
    def context_window_tokens(self) -> int:
        return 8192

    def supports_tools(self) -> bool:
        return True

    def supports_structured_output(self) -> bool:
        return True

    def generate_messages(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        run_id: str | None = None,
    ) -> LLMAdapterResponse:
        return build_adapter_response(content="unused")

    def generate_with_tools(
        self,
        messages: Sequence[ChatMessage],
        tools_schema: list[dict[str, Any]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        run_id: str | None = None,
    ) -> LLMAdapterResponse:
        self.route_calls += 1
        if self._decision is None:
            return build_adapter_response(content="")
        return build_adapter_response(
            content="",
            tool_calls=[
                LLMToolCall(
                    id="call-1",
                    name=ROUTER_TOOL_ID,
                    arguments_json=self._decision.model_dump_json(),
                )
            ],
        )

    def generate_structured(
        self,
        messages: Sequence[ChatMessage],
        output_model: type,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        run_id: str | None = None,
    ) -> LLMStructuredResult[Any]:
        raise AssertionError("structured output must not be called")


def _timing_input_for_case(case_id: str) -> CacheAwareCompactionTimingInput:
    case = next(item for item in CACHE_AWARE_COMPACTION_CORPUS if item.case_id == case_id)
    return case.timing_input


def _orchestrate(
    *,
    adapter: _NativeToolsAdapter,
    timing_input: CacheAwareCompactionTimingInput,
    router_request: TokenOptimizationLLMRouterRequest | None = None,
) -> CacheAwareTokenOptimizationOrchestrationResult:
    router = TokenOptimizationLLMRouter(adapter=adapter)
    orchestrator = CacheAwareTokenOptimizationOrchestrator(router=router)
    request = CacheAwareTokenOptimizationOrchestrationRequest(
        router_request=router_request or _router_request(),
        timing_input=timing_input,
    )
    return orchestrator.orchestrate(request)


def test_run_executes_pipeline_exactly_once() -> None:
    adapter = _NativeToolsAdapter(
        decision=_decision(TokenOptimizationRouterConfigurationId.EXACT_ONLY)
    )
    timing_input = _timing_input_for_case("cache_aware_compaction.dynamic_tail_safe_to_reduce")
    pipeline_calls = 0
    original_run = TokenOptimizationPipelineRunner.run

    def tracked_run(self, **kwargs: object) -> object:
        nonlocal pipeline_calls
        pipeline_calls += 1
        return original_run(self, **kwargs)

    with patch.object(TokenOptimizationPipelineRunner, "run", tracked_run):
        result = _orchestrate(adapter=adapter, timing_input=timing_input)

    assert result.orchestration_status is CacheAwareTokenOptimizationOrchestrationStatus.EXECUTED
    assert result.executed is True
    assert result.pipeline_result is not None
    assert pipeline_calls == 1
    assert adapter.route_calls == 1


def test_defer_does_not_execute_pipeline() -> None:
    adapter = _NativeToolsAdapter(
        decision=_decision(TokenOptimizationRouterConfigurationId.EXACT_ONLY)
    )
    timing_input = _timing_input_for_case("cache_aware_compaction.hot_stable_prefix_deferred")

    with patch.object(TokenOptimizationPipelineRunner, "run") as pipeline_run:
        result = _orchestrate(adapter=adapter, timing_input=timing_input)

    assert result.orchestration_status is CacheAwareTokenOptimizationOrchestrationStatus.DEFERRED
    assert result.executed is False
    assert result.pipeline_result is None
    pipeline_run.assert_not_called()
    assert result.timing_decision is not None
    assert result.timing_decision.reason is CacheAwareCompactionReason.CACHE_INVALIDATION_COST_TOO_HIGH


def test_bypass_does_not_execute_pipeline() -> None:
    adapter = _NativeToolsAdapter(
        decision=_decision(TokenOptimizationRouterConfigurationId.EXACT_ONLY)
    )
    timing_input = _timing_input_for_case("cache_aware_compaction.low_benefit_bypasses")

    with patch.object(TokenOptimizationPipelineRunner, "run") as pipeline_run:
        result = _orchestrate(adapter=adapter, timing_input=timing_input)

    assert result.orchestration_status is CacheAwareTokenOptimizationOrchestrationStatus.BYPASSED
    assert result.executed is False
    assert result.pipeline_result is None
    pipeline_run.assert_not_called()
    assert result.timing_decision is not None
    assert result.timing_decision.reason is CacheAwareCompactionReason.LOW_CONTENT_REDUCTION_BENEFIT


def test_require_manual_review_does_not_execute_pipeline() -> None:
    adapter = _NativeToolsAdapter(
        decision=_decision(TokenOptimizationRouterConfigurationId.EXACT_ONLY)
    )
    timing_input = _timing_input_for_case(
        "cache_aware_compaction.protected_or_semantic_risk_requires_review"
    )

    with patch.object(TokenOptimizationPipelineRunner, "run") as pipeline_run:
        result = _orchestrate(adapter=adapter, timing_input=timing_input)

    assert (
        result.orchestration_status
        is CacheAwareTokenOptimizationOrchestrationStatus.REVIEW_REQUIRED
    )
    assert result.review_required is True
    assert result.executed is False
    pipeline_run.assert_not_called()


def test_router_no_optimization_is_terminal() -> None:
    adapter = _NativeToolsAdapter(
        decision=_decision(
            TokenOptimizationRouterConfigurationId.NO_OPTIMIZATION,
            reason_code=TokenOptimizationRouterReasonCode.CLEAN_NO_OP,
        )
    )
    timing_input = _timing_input_for_case("cache_aware_compaction.dynamic_tail_safe_to_reduce")

    with patch(
        "intergrax.runtime.token_optimization.cache_aware_orchestration.decide_cache_aware_compaction_timing"
    ) as timing_helper:
        with patch.object(TokenOptimizationPipelineRunner, "run") as pipeline_run:
            result = _orchestrate(adapter=adapter, timing_input=timing_input)

    timing_helper.assert_not_called()
    pipeline_run.assert_not_called()
    assert (
        result.orchestration_status
        is CacheAwareTokenOptimizationOrchestrationStatus.ROUTER_TERMINAL
    )
    assert result.router_result.status is TokenOptimizationRouterStatus.NO_OPTIMIZATION


def test_router_review_required_is_terminal() -> None:
    adapter = _NativeToolsAdapter(
        decision=_decision(
            TokenOptimizationRouterConfigurationId.EXACT_ONLY,
            review_required=True,
        )
    )
    timing_input = _timing_input_for_case("cache_aware_compaction.dynamic_tail_safe_to_reduce")

    with patch(
        "intergrax.runtime.token_optimization.cache_aware_orchestration.decide_cache_aware_compaction_timing"
    ) as timing_helper:
        with patch.object(TokenOptimizationPipelineRunner, "run") as pipeline_run:
            result = _orchestrate(adapter=adapter, timing_input=timing_input)

    timing_helper.assert_not_called()
    pipeline_run.assert_not_called()
    assert (
        result.orchestration_status
        is CacheAwareTokenOptimizationOrchestrationStatus.ROUTER_TERMINAL
    )
    assert result.router_result.status is TokenOptimizationRouterStatus.REVIEW_REQUIRED
    assert result.review_required is False


def test_router_blocked_is_terminal() -> None:
    adapter = _NativeToolsAdapter(
        decision=_decision(
            TokenOptimizationRouterConfigurationId.EXACT_ONLY,
            confidence=0.1,
        )
    )
    timing_input = _timing_input_for_case("cache_aware_compaction.dynamic_tail_safe_to_reduce")

    with patch(
        "intergrax.runtime.token_optimization.cache_aware_orchestration.decide_cache_aware_compaction_timing"
    ) as timing_helper:
        with patch.object(TokenOptimizationPipelineRunner, "run") as pipeline_run:
            result = _orchestrate(adapter=adapter, timing_input=timing_input)

    timing_helper.assert_not_called()
    pipeline_run.assert_not_called()
    assert (
        result.orchestration_status
        is CacheAwareTokenOptimizationOrchestrationStatus.ROUTER_TERMINAL
    )
    assert result.router_result.status is TokenOptimizationRouterStatus.BLOCKED


def test_protected_semantic_risk_requires_review() -> None:
    adapter = _NativeToolsAdapter(
        decision=_decision(TokenOptimizationRouterConfigurationId.EXACT_ONLY)
    )
    timing_input = CacheAwareCompactionTimingInput(
        target=CacheAwareCompactionTarget.DYNAMIC_TAIL,
        prefix_stability_status=PREFIX_STABILITY_STABLE,
        protected_or_semantic_risk=True,
        dynamic_tail_reduction_available=True,
        estimated_content_reduction_chars=250,
    )

    with patch.object(TokenOptimizationPipelineRunner, "run") as pipeline_run:
        result = _orchestrate(adapter=adapter, timing_input=timing_input)

    pipeline_run.assert_not_called()
    assert (
        result.orchestration_status
        is CacheAwareTokenOptimizationOrchestrationStatus.REVIEW_REQUIRED
    )
    assert result.timing_decision is not None
    assert result.timing_decision.reason is CacheAwareCompactionReason.PROTECTED_OR_SEMANTIC_RISK


def test_unstable_prefix_defers() -> None:
    adapter = _NativeToolsAdapter(
        decision=_decision(TokenOptimizationRouterConfigurationId.EXACT_ONLY)
    )
    timing_input = _timing_input_for_case("cache_aware_compaction.prefix_not_stable_defers")

    with patch.object(TokenOptimizationPipelineRunner, "run") as pipeline_run:
        result = _orchestrate(adapter=adapter, timing_input=timing_input)

    pipeline_run.assert_not_called()
    assert result.orchestration_status is CacheAwareTokenOptimizationOrchestrationStatus.DEFERRED
    assert result.timing_decision is not None
    assert result.timing_decision.reason is CacheAwareCompactionReason.PREFIX_NOT_STABLE


def test_dynamic_tail_run() -> None:
    adapter = _NativeToolsAdapter(
        decision=_decision(TokenOptimizationRouterConfigurationId.EXACT_ONLY)
    )
    timing_input = _timing_input_for_case("cache_aware_compaction.dynamic_tail_safe_to_reduce")

    result = _orchestrate(adapter=adapter, timing_input=timing_input)

    assert result.orchestration_status is CacheAwareTokenOptimizationOrchestrationStatus.EXECUTED
    assert result.timing_decision is not None
    assert result.timing_decision.reason is CacheAwareCompactionReason.DYNAMIC_TAIL_SAFE_TO_REDUCE


def test_low_benefit_bypass() -> None:
    adapter = _NativeToolsAdapter(
        decision=_decision(TokenOptimizationRouterConfigurationId.EXACT_ONLY)
    )
    timing_input = _timing_input_for_case("cache_aware_compaction.low_benefit_bypasses")

    result = _orchestrate(adapter=adapter, timing_input=timing_input)

    assert result.orchestration_status is CacheAwareTokenOptimizationOrchestrationStatus.BYPASSED
    assert result.timing_decision is not None
    assert result.timing_decision.reason is CacheAwareCompactionReason.LOW_CONTENT_REDUCTION_BENEFIT


def test_high_invalidation_cost_defers() -> None:
    adapter = _NativeToolsAdapter(
        decision=_decision(TokenOptimizationRouterConfigurationId.EXACT_ONLY)
    )
    timing_input = _timing_input_for_case("cache_aware_compaction.hot_stable_prefix_deferred")

    result = _orchestrate(adapter=adapter, timing_input=timing_input)

    assert result.orchestration_status is CacheAwareTokenOptimizationOrchestrationStatus.DEFERRED
    assert result.timing_decision is not None
    assert result.timing_decision.reason is CacheAwareCompactionReason.CACHE_INVALIDATION_COST_TOO_HIGH


def test_full_thread_rewrite_requires_review() -> None:
    adapter = _NativeToolsAdapter(
        decision=_decision(TokenOptimizationRouterConfigurationId.EXACT_ONLY)
    )
    timing_input = _timing_input_for_case("cache_aware_compaction.full_thread_rewrite_requires_review")

    with patch.object(TokenOptimizationPipelineRunner, "run") as pipeline_run:
        result = _orchestrate(adapter=adapter, timing_input=timing_input)

    pipeline_run.assert_not_called()
    assert (
        result.orchestration_status
        is CacheAwareTokenOptimizationOrchestrationStatus.REVIEW_REQUIRED
    )
    assert result.timing_decision is not None
    assert result.timing_decision.reason is CacheAwareCompactionReason.FULL_THREAD_REWRITE_RISK


def test_safe_report_allowlist_and_no_raw_content() -> None:
    adapter = _NativeToolsAdapter(
        decision=_decision(TokenOptimizationRouterConfigurationId.EXACT_ONLY)
    )
    timing_input = _timing_input_for_case("cache_aware_compaction.dynamic_tail_safe_to_reduce")
    result = _orchestrate(adapter=adapter, timing_input=timing_input)

    safe = cache_aware_orchestration_result_to_safe_dict(result)
    dumped = json.dumps(safe)

    assert safe["raw_content_included"] is False
    assert "SYNTH-ALPHA" not in dumped
    assert set(safe.keys()) <= {
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


def test_invalid_result_invariants_rejected() -> None:
    router_result = TokenOptimizationLLMRouter(
        adapter=_NativeToolsAdapter(decision=_decision(TokenOptimizationRouterConfigurationId.EXACT_ONLY))
    ).route(_router_request())

    with pytest.raises(ValueError, match="EXECUTED requires pipeline_result"):
        CacheAwareTokenOptimizationOrchestrationResult(
            router_result=router_result,
            timing_decision=CacheAwareCompactionTimingDecision(
                decision=CacheAwareCompactionDecision.RUN,
                reason=CacheAwareCompactionReason.DYNAMIC_TAIL_SAFE_TO_REDUCE,
                target=CacheAwareCompactionTarget.DYNAMIC_TAIL,
                cache_hot=True,
                ttl_seconds_remaining=600,
                estimated_content_reduction_chars=400,
                estimated_cache_invalidation_cost_tokens=50,
            ),
            orchestration_status=CacheAwareTokenOptimizationOrchestrationStatus.EXECUTED,
            pipeline_result=None,
            executed=True,
            review_required=False,
        )


def test_execute_routed_does_not_call_llm() -> None:
    adapter = _NativeToolsAdapter(
        decision=_decision(TokenOptimizationRouterConfigurationId.EXACT_ONLY)
    )
    router = TokenOptimizationLLMRouter(adapter=adapter)
    router_request = _router_request()
    routed = router.route(router_request)
    pipeline_calls = 0
    original_run = TokenOptimizationPipelineRunner.run

    def tracked_run(self, **kwargs: object) -> object:
        nonlocal pipeline_calls
        pipeline_calls += 1
        return original_run(self, **kwargs)

    with patch.object(router, "route") as route_mock:
        with patch.object(TokenOptimizationPipelineRunner, "run", tracked_run):
            result = router.execute_routed(router_request, routed)

    route_mock.assert_not_called()
    assert adapter.route_calls == 1
    assert pipeline_calls == 1
    assert result.executed is True
    assert result.pipeline_result is not None


def test_execute_routed_rejects_non_routed_status() -> None:
    adapter = _NativeToolsAdapter(
        decision=_decision(
            TokenOptimizationRouterConfigurationId.NO_OPTIMIZATION,
            reason_code=TokenOptimizationRouterReasonCode.CLEAN_NO_OP,
        )
    )
    router = TokenOptimizationLLMRouter(adapter=adapter)
    router_request = _router_request()
    routed = router.route(router_request)

    with pytest.raises(ValueError, match="execute_routed requires router status ROUTED"):
        router.execute_routed(router_request, routed)


def test_execute_routed_rejects_missing_pipeline_config() -> None:
    routed = TokenOptimizationLLMRouterResult(
        request_id="orchestration-test-1",
        status=TokenOptimizationRouterStatus.ROUTED,
        reason=None,
        transport=TokenOptimizationRouterTransport.NATIVE_TOOLS,
        configuration_id=TokenOptimizationRouterConfigurationId.EXACT_ONLY,
        reason_code=TokenOptimizationRouterReasonCode.EXACT_DUPLICATES,
        risk=TokenOptimizationRouterRisk.LOW,
        review_required=False,
        confidence=0.9,
        provider="fake",
        model="fake",
        tool_call_id="call-1",
        pipeline_config=None,
        pipeline_result=None,
        executed=False,
    )
    router = TokenOptimizationLLMRouter(adapter=_NativeToolsAdapter())

    with pytest.raises(ValueError, match="execute_routed requires pipeline_config"):
        router.execute_routed(_router_request(), routed)


def test_execute_routed_rejects_missing_configuration_id() -> None:
    routed_result = TokenOptimizationLLMRouter(
        adapter=_NativeToolsAdapter(decision=_decision(TokenOptimizationRouterConfigurationId.EXACT_ONLY))
    ).route(_router_request())
    routed = TokenOptimizationLLMRouterResult(
        request_id=routed_result.request_id,
        status=routed_result.status,
        reason=routed_result.reason,
        transport=routed_result.transport,
        configuration_id=None,
        reason_code=routed_result.reason_code,
        risk=routed_result.risk,
        review_required=routed_result.review_required,
        confidence=routed_result.confidence,
        provider=routed_result.provider,
        model=routed_result.model,
        tool_call_id=routed_result.tool_call_id,
        pipeline_config=routed_result.pipeline_config,
        pipeline_result=None,
        executed=False,
    )
    router = TokenOptimizationLLMRouter(adapter=_NativeToolsAdapter())

    with pytest.raises(ValueError, match="execute_routed requires configuration_id"):
        router.execute_routed(_router_request(), routed)


def test_execute_routed_rejects_request_id_mismatch() -> None:
    routed = TokenOptimizationLLMRouter(
        adapter=_NativeToolsAdapter(decision=_decision(TokenOptimizationRouterConfigurationId.EXACT_ONLY))
    ).route(_router_request())
    router = TokenOptimizationLLMRouter(adapter=_NativeToolsAdapter())

    with pytest.raises(ValueError, match="request_id mismatch"):
        router.execute_routed(_router_request(request_id="different-id"), routed)


def test_route_and_execute_regression() -> None:
    adapter = _NativeToolsAdapter(
        decision=_decision(TokenOptimizationRouterConfigurationId.EXACT_ONLY)
    )
    router = TokenOptimizationLLMRouter(adapter=adapter)

    with patch.object(router, "route", wraps=router.route) as route_mock:
        with patch.object(router, "execute_routed", wraps=router.execute_routed) as execute_mock:
            result = router.route_and_execute(_router_request())

    route_mock.assert_called_once()
    execute_mock.assert_called_once()
    assert result.executed is True
    assert result.pipeline_result is not None
    assert "builtin.exact_deduplication" in result.pipeline_result.applied_layer_ids
