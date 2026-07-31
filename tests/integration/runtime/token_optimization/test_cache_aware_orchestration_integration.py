# © Artur Czarnecki. All rights reserved.

"""Integration tests for cache-aware orchestration gate (TOKEN-10D-1)."""

from __future__ import annotations

from typing import Any, Sequence

import pytest

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters._shared.adapter_response_builders import build_adapter_response
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.contracts.tool_call import LLMToolCall
from intergrax.runtime.token_optimization.cache_aware_orchestration import (
    CacheAwareTokenOptimizationOrchestrator,
)
from intergrax.runtime.token_optimization.cache_aware_orchestration_contracts import (
    CacheAwareTokenOptimizationOrchestrationRequest,
    CacheAwareTokenOptimizationOrchestrationStatus,
)
from intergrax.runtime.token_optimization.contracts import (
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
from tests.fixtures.token_optimization.cache_aware_compaction_corpus import (
    CACHE_AWARE_COMPACTION_CORPUS,
)
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


def _request_from_corpus(case_id: str) -> TokenOptimizationLLMRouterRequest:
    case = next(item for item in LLM_ROUTER_CORPUS if item.case_id == case_id)
    return TokenOptimizationLLMRouterRequest(
        request=TokenOptimizationRequest(
            content=case.content,
            source_type=case.source_type,
            policy=case.policy,
            protected_regions=case.protected_regions,
            metadata=dict(case.metadata),
        ),
        policy=TokenOptimizationLLMRouterPolicy(),
        request_id=f"integration-orchestration-{case.case_id}",
    )


def _timing_input_for_case(case_id: str):
    case = next(item for item in CACHE_AWARE_COMPACTION_CORPUS if item.case_id == case_id)
    return case.timing_input


def test_orchestration_run_executes_real_pipeline() -> None:
    adapter = FakeNativeToolsAdapter(
        decision=_decision(TokenOptimizationRouterConfigurationId.EXACT_ONLY)
    )
    router = TokenOptimizationLLMRouter(adapter=adapter)
    orchestrator = CacheAwareTokenOptimizationOrchestrator(router=router)
    router_request = _request_from_corpus("router.rag_exact_duplicates")
    timing_input = _timing_input_for_case("cache_aware_compaction.dynamic_tail_safe_to_reduce")

    result = orchestrator.orchestrate(
        CacheAwareTokenOptimizationOrchestrationRequest(
            router_request=router_request,
            timing_input=timing_input,
        )
    )

    assert (
        result.orchestration_status is CacheAwareTokenOptimizationOrchestrationStatus.EXECUTED
    )
    assert result.executed is True
    assert result.pipeline_result is not None
    assert "builtin.exact_deduplication" in result.pipeline_result.applied_layer_ids
    assert result.pipeline_result.original_content != result.pipeline_result.final_content


def test_orchestration_defer_skips_real_pipeline() -> None:
    adapter = FakeNativeToolsAdapter(
        decision=_decision(TokenOptimizationRouterConfigurationId.EXACT_ONLY)
    )
    router = TokenOptimizationLLMRouter(adapter=adapter)
    orchestrator = CacheAwareTokenOptimizationOrchestrator(router=router)
    router_request = _request_from_corpus("router.rag_exact_duplicates")
    timing_input = _timing_input_for_case("cache_aware_compaction.hot_stable_prefix_deferred")

    result = orchestrator.orchestrate(
        CacheAwareTokenOptimizationOrchestrationRequest(
            router_request=router_request,
            timing_input=timing_input,
        )
    )

    assert result.orchestration_status is CacheAwareTokenOptimizationOrchestrationStatus.DEFERRED
    assert result.executed is False
    assert result.pipeline_result is None
    assert result.router_result.pipeline_result is None
    assert result.router_result.executed is False
