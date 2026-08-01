# © Artur Czarnecki. All rights reserved.

"""Integration tests for cache-aware runtime composition (TOKEN-10D-3)."""

from __future__ import annotations

from typing import Any, Sequence

import pytest

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters._shared.adapter_response_builders import build_adapter_response
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.contracts.provider_extensions import (
    LLMProviderExtensions,
    VllmProviderExtensions,
)
from intergrax.llm_adapters.contracts.token_usage import LLMTokenUsage
from intergrax.llm_adapters.contracts.tool_call import LLMToolCall
from intergrax.runtime.token_optimization.cache_aware_orchestration import (
    CacheAwareTokenOptimizationOrchestrator,
)
from intergrax.runtime.token_optimization.cache_aware_runtime import (
    CacheAwareTokenOptimizationRuntime,
)
from intergrax.runtime.token_optimization.cache_aware_runtime_contracts import (
    CacheAwareTokenOptimizationRuntimeRequest,
    CacheAwareTokenOptimizationRuntimeStatus,
)
from intergrax.runtime.token_optimization.cache_signal_normalization_contracts import (
    CacheAwareCompactionSignalNormalizationStatus,
)
from intergrax.runtime.token_optimization.contracts import (
    CacheAwareCompactionDecision,
    CacheAwareCompactionReason,
    CacheAwareCompactionTarget,
    PromptCacheAttribution,
    PromptCacheMode,
    PromptCachePolicy,
    PromptCacheProviderCapabilities,
    PromptCacheUsageSnapshot,
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
    TokenOptimizationRouterConfigurationId,
    TokenOptimizationRouterReasonCode,
    TokenOptimizationRouterRisk,
    TokenOptimizationRouterStatus,
    TokenOptimizationRouterToolInput,
)
from intergrax.runtime.token_optimization.prompt_cache import PREFIX_STABILITY_STABLE
from tests.fixtures.token_optimization.llm_router_corpus import LLM_ROUTER_CORPUS

pytestmark = pytest.mark.integration


def _decision(
    configuration_id: TokenOptimizationRouterConfigurationId,
    *,
    reason_code: TokenOptimizationRouterReasonCode = TokenOptimizationRouterReasonCode.EXACT_DUPLICATES,
) -> TokenOptimizationRouterToolInput:
    return TokenOptimizationRouterToolInput(
        configuration_id=configuration_id,
        reason_code=reason_code,
        risk=TokenOptimizationRouterRisk.LOW,
        review_required=False,
        confidence=0.9,
    )


class FakeNativeToolsAdapter(LLMAdapter):
    provider = "fake-integration"
    model = "fake-integration"

    def __init__(self, *, decision: TokenOptimizationRouterToolInput) -> None:
        super().__init__()
        self._decision = decision
        self.route_calls = 0
        self.pipeline_execution_count = 0

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
        self.pipeline_execution_count += 1
        return build_adapter_response(
            content="",
            tool_calls=[
                LLMToolCall(
                    id="integration-call-1",
                    name=ROUTER_TOOL_ID,
                    arguments_json=self._decision.model_dump_json(),
                )
            ],
        )


def _router_request() -> TokenOptimizationLLMRouterRequest:
    case = next(item for item in LLM_ROUTER_CORPUS if item.case_id == "router.rag_exact_duplicates")
    return TokenOptimizationLLMRouterRequest(
        request=TokenOptimizationRequest(
            content=case.content,
            source_type=case.source_type,
            policy=case.policy,
            protected_regions=case.protected_regions,
            metadata=dict(case.metadata),
        ),
        policy=TokenOptimizationLLMRouterPolicy(),
        request_id="integration-cache-aware-runtime",
    )


def _vllm_capabilities() -> PromptCacheProviderCapabilities:
    return PromptCacheProviderCapabilities(
        provider="vllm",
        supports_prompt_caching=True,
        supports_cache_usage_tokens=True,
    )


def _vllm_response(
    *,
    cached_input_tokens: int,
    details_reported: bool,
) -> LLMAdapterResponse:
    return build_adapter_response(
        content="assistant",
        provider="vllm",
        model="vllm-integration",
        usage=LLMTokenUsage.from_counts(
            input_tokens=1000,
            output_tokens=20,
            cached_input_tokens=cached_input_tokens,
        ),
        provider_extensions=LLMProviderExtensions(
            usage_source="sdk",
            vllm=VllmProviderExtensions(
                prompt_tokens_details_reported=details_reported,
            ),
        ),
    )


def _attribution_from_response(response: LLMAdapterResponse) -> PromptCacheAttribution:
    from intergrax.runtime.token_optimization.cache_signal_normalization import (
        prompt_cache_usage_snapshot_from_adapter_response,
    )

    usage = prompt_cache_usage_snapshot_from_adapter_response(response)
    return PromptCacheAttribution(
        policy=PromptCachePolicy(enabled=True, mode=PromptCacheMode.PROVIDER_DEFAULT),
        provider_capabilities=_vllm_capabilities(),
        usage=usage,
        prefix_stability_status=PREFIX_STABILITY_STABLE,
    )


def _runtime(
    *,
    decision: TokenOptimizationRouterToolInput,
) -> tuple[CacheAwareTokenOptimizationRuntime, FakeNativeToolsAdapter]:
    adapter = FakeNativeToolsAdapter(decision=decision)
    router = TokenOptimizationLLMRouter(adapter=adapter)
    orchestrator = CacheAwareTokenOptimizationOrchestrator(router=router)
    return CacheAwareTokenOptimizationRuntime(orchestrator=orchestrator), adapter


def test_cache_hit_defers_without_pipeline() -> None:
    runtime, adapter = _runtime(decision=_decision(TokenOptimizationRouterConfigurationId.EXACT_ONLY))
    response = _vllm_response(cached_input_tokens=800, details_reported=True)

    result = runtime.run(
        CacheAwareTokenOptimizationRuntimeRequest(
            router_request=_router_request(),
            cache_attribution=_attribution_from_response(response),
            adapter_response=response,
            target=CacheAwareCompactionTarget.STABLE_PREFIX,
            ttl_seconds_remaining=900,
            estimated_content_reduction_chars=50,
        )
    )

    assert result.normalization_result is not None
    assert (
        result.normalization_result.status
        is CacheAwareCompactionSignalNormalizationStatus.NORMALIZED
    )
    assert result.normalization_result.timing_input is not None
    assert result.normalization_result.timing_input.cache_hot is True
    assert result.status is CacheAwareTokenOptimizationRuntimeStatus.DEFERRED
    assert result.executed is False
    assert result.orchestration_result is not None
    assert result.orchestration_result.pipeline_result is None
    assert adapter.route_calls == 1
    assert adapter.pipeline_execution_count == 1


def test_explicit_cache_miss_runs_pipeline_once() -> None:
    runtime, adapter = _runtime(decision=_decision(TokenOptimizationRouterConfigurationId.EXACT_ONLY))
    response = _vllm_response(cached_input_tokens=0, details_reported=True)

    result = runtime.run(
        CacheAwareTokenOptimizationRuntimeRequest(
            router_request=_router_request(),
            cache_attribution=_attribution_from_response(response),
            adapter_response=response,
            target=CacheAwareCompactionTarget.COLD_HISTORY,
            estimated_content_reduction_chars=200,
        )
    )

    assert result.normalization_result is not None
    assert (
        result.normalization_result.status
        is CacheAwareCompactionSignalNormalizationStatus.NORMALIZED
    )
    assert result.normalization_result.timing_input is not None
    assert result.normalization_result.timing_input.cache_hot is False
    assert result.status is CacheAwareTokenOptimizationRuntimeStatus.EXECUTED
    assert result.executed is True
    assert result.orchestration_result is not None
    assert result.orchestration_result.pipeline_result is not None
    assert result.orchestration_result.timing_decision is not None
    assert result.orchestration_result.timing_decision.decision is CacheAwareCompactionDecision.RUN
    assert adapter.route_calls == 1
    assert adapter.pipeline_execution_count == 1


def test_unknown_cache_state_partial_without_invented_miss() -> None:
    runtime, adapter = _runtime(decision=_decision(TokenOptimizationRouterConfigurationId.EXACT_ONLY))
    response = _vllm_response(cached_input_tokens=0, details_reported=False)

    result = runtime.run(
        CacheAwareTokenOptimizationRuntimeRequest(
            router_request=_router_request(),
            cache_attribution=_attribution_from_response(response),
            adapter_response=response,
            target=CacheAwareCompactionTarget.STABLE_PREFIX,
            estimated_content_reduction_chars=100,
        )
    )

    assert result.normalization_result is not None
    assert (
        result.normalization_result.status
        is CacheAwareCompactionSignalNormalizationStatus.PARTIAL
    )
    assert result.normalization_result.timing_input is not None
    assert result.normalization_result.timing_input.cache_hot is None
    assert result.status is CacheAwareTokenOptimizationRuntimeStatus.REVIEW_REQUIRED
    assert result.executed is False
    assert result.orchestration_result is not None
    assert result.orchestration_result.timing_decision is not None
    assert (
        result.orchestration_result.timing_decision.decision
        is CacheAwareCompactionDecision.REQUIRE_MANUAL_REVIEW
    )
    assert (
        result.orchestration_result.timing_decision.reason
        is CacheAwareCompactionReason.INSUFFICIENT_SIGNALS
    )
    assert adapter.route_calls == 1
    assert adapter.pipeline_execution_count == 1


def test_conflicting_evidence_rejected_without_router_or_pipeline() -> None:
    runtime, adapter = _runtime(decision=_decision(TokenOptimizationRouterConfigurationId.EXACT_ONLY))
    response = _vllm_response(cached_input_tokens=800, details_reported=True)
    attribution = _attribution_from_response(response)
    conflicting_usage = PromptCacheUsageSnapshot(
        provider="vllm",
        model="vllm-integration",
        cached_input_tokens=500,
        uncached_input_tokens=500,
        cache_hit_ratio=0.5,
    )
    attribution = PromptCacheAttribution(
        policy=attribution.policy,
        provider_capabilities=attribution.provider_capabilities,
        usage=conflicting_usage,
        prefix_stability_status=attribution.prefix_stability_status,
        invalidation_reason=attribution.invalidation_reason,
    )

    result = runtime.run(
        CacheAwareTokenOptimizationRuntimeRequest(
            router_request=_router_request(),
            cache_attribution=attribution,
            adapter_response=response,
            target=CacheAwareCompactionTarget.STABLE_PREFIX,
            estimated_content_reduction_chars=50,
        )
    )

    assert result.status is CacheAwareTokenOptimizationRuntimeStatus.SIGNALS_REJECTED
    assert result.orchestration_result is None
    assert adapter.route_calls == 0
    assert adapter.pipeline_execution_count == 0


def test_router_terminal_without_pipeline() -> None:
    runtime, adapter = _runtime(
        decision=_decision(
            TokenOptimizationRouterConfigurationId.NO_OPTIMIZATION,
            reason_code=TokenOptimizationRouterReasonCode.CLEAN_NO_OP,
        )
    )
    response = _vllm_response(cached_input_tokens=0, details_reported=True)

    result = runtime.run(
        CacheAwareTokenOptimizationRuntimeRequest(
            router_request=_router_request(),
            cache_attribution=_attribution_from_response(response),
            adapter_response=response,
            target=CacheAwareCompactionTarget.COLD_HISTORY,
            estimated_content_reduction_chars=200,
        )
    )

    assert result.status is CacheAwareTokenOptimizationRuntimeStatus.ROUTER_TERMINAL
    assert result.executed is False
    assert result.orchestration_result is not None
    assert (
        result.orchestration_result.router_result.status
        is TokenOptimizationRouterStatus.NO_OPTIMIZATION
    )
    assert result.orchestration_result.pipeline_result is None
    assert adapter.route_calls == 1
    assert adapter.pipeline_execution_count == 1
