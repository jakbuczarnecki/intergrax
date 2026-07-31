# © Artur Czarnecki. All rights reserved.

"""Integration tests for cache signal normalization wiring (TOKEN-10D-2)."""

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
from intergrax.runtime.token_optimization.cache_aware_orchestration_contracts import (
    CacheAwareTokenOptimizationOrchestrationRequest,
    CacheAwareTokenOptimizationOrchestrationStatus,
)
from intergrax.runtime.token_optimization.cache_signal_normalization import (
    normalize_cache_aware_compaction_signals,
    prompt_cache_usage_snapshot_from_adapter_response,
)
from intergrax.runtime.token_optimization.cache_signal_normalization_contracts import (
    CacheAwareCompactionSignalNormalizationRequest,
)
from intergrax.runtime.token_optimization.contracts import (
    CacheAwareCompactionDecision,
    CacheAwareCompactionReason,
    CacheAwareCompactionTarget,
    PromptCacheAttribution,
    PromptCacheMode,
    PromptCachePolicy,
    PromptCacheProviderCapabilities,
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
    TokenOptimizationRouterToolInput,
)
from intergrax.runtime.token_optimization.prompt_cache import PREFIX_STABILITY_STABLE
from tests.fixtures.token_optimization.llm_router_corpus import LLM_ROUTER_CORPUS

pytestmark = pytest.mark.integration


def _decision() -> TokenOptimizationRouterToolInput:
    return TokenOptimizationRouterToolInput(
        configuration_id=TokenOptimizationRouterConfigurationId.EXACT_ONLY,
        reason_code=TokenOptimizationRouterReasonCode.EXACT_DUPLICATES,
        risk=TokenOptimizationRouterRisk.LOW,
        review_required=False,
        confidence=0.9,
    )


class FakeNativeToolsAdapter(LLMAdapter):
    provider = "fake-integration"
    model = "fake-integration"

    def __init__(self) -> None:
        super().__init__()
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
        self.pipeline_execution_count += 1
        return build_adapter_response(
            content="",
            tool_calls=[
                LLMToolCall(
                    id="integration-call-1",
                    name=ROUTER_TOOL_ID,
                    arguments_json=_decision().model_dump_json(),
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
        request_id="integration-cache-signal-normalization",
    )


def _vllm_capabilities() -> PromptCacheProviderCapabilities:
    return PromptCacheProviderCapabilities(
        provider="vllm",
        supports_prompt_caching=True,
        supports_cache_usage_tokens=True,
    )


def _vllm_adapter_response(
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
    usage = prompt_cache_usage_snapshot_from_adapter_response(response)
    return PromptCacheAttribution(
        policy=PromptCachePolicy(enabled=True, mode=PromptCacheMode.PROVIDER_DEFAULT),
        provider_capabilities=_vllm_capabilities(),
        usage=usage,
        prefix_stability_status=PREFIX_STABILITY_STABLE,
    )


def test_adapter_cache_hit_defers_pipeline() -> None:
    adapter = FakeNativeToolsAdapter()
    router = TokenOptimizationLLMRouter(adapter=adapter)
    orchestrator = CacheAwareTokenOptimizationOrchestrator(router=router)

    llm_response = _vllm_adapter_response(cached_input_tokens=800, details_reported=True)
    normalization = normalize_cache_aware_compaction_signals(
        CacheAwareCompactionSignalNormalizationRequest(
            target=CacheAwareCompactionTarget.STABLE_PREFIX,
            cache_attribution=_attribution_from_response(llm_response),
            ttl_seconds_remaining=900,
            estimated_content_reduction_chars=50,
        )
    )
    assert normalization.timing_input is not None
    assert normalization.timing_input.cache_hot is True
    assert normalization.timing_input.estimated_cache_invalidation_cost_tokens == 800

    result = orchestrator.orchestrate(
        CacheAwareTokenOptimizationOrchestrationRequest(
            router_request=_router_request(),
            timing_input=normalization.timing_input,
        )
    )

    assert (
        result.orchestration_status is CacheAwareTokenOptimizationOrchestrationStatus.DEFERRED
    )
    assert result.timing_decision is not None
    assert result.timing_decision.decision is CacheAwareCompactionDecision.DEFER
    assert (
        result.timing_decision.reason
        is CacheAwareCompactionReason.CACHE_INVALIDATION_COST_TOO_HIGH
    )
    assert result.executed is False
    assert result.pipeline_result is None
    assert adapter.pipeline_execution_count == 1


def test_explicit_cache_miss_runs_cold_history_once() -> None:
    adapter = FakeNativeToolsAdapter()
    router = TokenOptimizationLLMRouter(adapter=adapter)
    orchestrator = CacheAwareTokenOptimizationOrchestrator(router=router)

    llm_response = _vllm_adapter_response(cached_input_tokens=0, details_reported=True)
    normalization = normalize_cache_aware_compaction_signals(
        CacheAwareCompactionSignalNormalizationRequest(
            target=CacheAwareCompactionTarget.COLD_HISTORY,
            cache_attribution=_attribution_from_response(llm_response),
            estimated_content_reduction_chars=200,
        )
    )
    assert normalization.timing_input is not None
    assert normalization.timing_input.cache_hot is False

    result = orchestrator.orchestrate(
        CacheAwareTokenOptimizationOrchestrationRequest(
            router_request=_router_request(),
            timing_input=normalization.timing_input,
        )
    )

    assert (
        result.orchestration_status is CacheAwareTokenOptimizationOrchestrationStatus.EXECUTED
    )
    assert result.timing_decision is not None
    assert result.timing_decision.decision is CacheAwareCompactionDecision.RUN
    assert result.executed is True
    assert result.pipeline_result is not None
    assert adapter.pipeline_execution_count == 1


def test_missing_cache_details_require_review_without_pipeline() -> None:
    adapter = FakeNativeToolsAdapter()
    router = TokenOptimizationLLMRouter(adapter=adapter)
    orchestrator = CacheAwareTokenOptimizationOrchestrator(router=router)

    llm_response = _vllm_adapter_response(cached_input_tokens=0, details_reported=False)
    normalization = normalize_cache_aware_compaction_signals(
        CacheAwareCompactionSignalNormalizationRequest(
            target=CacheAwareCompactionTarget.STABLE_PREFIX,
            cache_attribution=_attribution_from_response(llm_response),
            estimated_content_reduction_chars=100,
        )
    )
    assert normalization.timing_input is not None
    assert normalization.timing_input.cache_hot is None

    result = orchestrator.orchestrate(
        CacheAwareTokenOptimizationOrchestrationRequest(
            router_request=_router_request(),
            timing_input=normalization.timing_input,
        )
    )

    assert (
        result.orchestration_status
        is CacheAwareTokenOptimizationOrchestrationStatus.REVIEW_REQUIRED
    )
    assert result.timing_decision is not None
    assert (
        result.timing_decision.decision
        is CacheAwareCompactionDecision.REQUIRE_MANUAL_REVIEW
    )
    assert result.executed is False
    assert result.pipeline_result is None
    assert adapter.pipeline_execution_count == 1
