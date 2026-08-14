# © Artur Czarnecki. All rights reserved.

"""TOKEN-10D-CLOSEOUT-1: public cache-aware runtime contract consumer tests."""

from __future__ import annotations

import json
from typing import Any, Sequence
from unittest.mock import MagicMock

import pytest

import intergrax.runtime.token_optimization as token_optimization
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

pytestmark = pytest.mark.unit

_TOKEN_10D_1_SYMBOLS = (
    "CacheAwareTokenOptimizationOrchestrator",
    "CacheAwareTokenOptimizationOrchestrationRequest",
    "CacheAwareTokenOptimizationOrchestrationResult",
    "CacheAwareTokenOptimizationOrchestrationStatus",
    "cache_aware_orchestration_result_to_safe_dict",
)

_TOKEN_10D_2_SYMBOLS = (
    "CacheAwareCompactionSignalNormalizationRequest",
    "CacheAwareCompactionSignalNormalizationResult",
    "CacheAwareCompactionSignalNormalizationStatus",
    "CacheAwareCompactionSignalNormalizationReason",
    "CacheSignalValueSource",
    "normalize_cache_aware_compaction_signals",
    "prompt_cache_usage_snapshot_from_adapter_response",
    "cache_signal_normalization_result_to_safe_dict",
)

_TOKEN_10D_3_SYMBOLS = (
    "CacheAwareTokenOptimizationRuntime",
    "CacheAwareTokenOptimizationRuntimeRequest",
    "CacheAwareTokenOptimizationRuntimeResult",
    "CacheAwareTokenOptimizationRuntimeStatus",
    "CacheAwareTokenOptimizationEvidenceReconciliationReason",
    "cache_aware_runtime_result_to_safe_dict",
)

_SHARED_CONSUMER_SYMBOLS = (
    "CacheAwareCompactionDecision",
    "CacheAwareCompactionReason",
    "CacheAwareCompactionTarget",
    "CacheAwareCompactionTimingDecision",
    "CacheAwareCompactionTimingInput",
    "PromptCacheAttribution",
    "PromptCacheInvalidationReason",
    "PromptCachePolicy",
    "PromptCacheProviderCapabilities",
    "PromptCacheUsageSnapshot",
    "TokenOptimizationLLMRouterRequest",
    "TokenOptimizationLLMRouterResult",
    "TokenOptimizationRouterStatus",
    "decide_cache_aware_compaction_timing",
)

_FROZEN_TOKEN_10D_SYMBOLS = (
    _TOKEN_10D_1_SYMBOLS + _TOKEN_10D_2_SYMBOLS + _TOKEN_10D_3_SYMBOLS + _SHARED_CONSUMER_SYMBOLS
)

_CANONICAL_MODULES: dict[str, str] = {
    "CacheAwareTokenOptimizationOrchestrator": (
        "intergrax.runtime.token_optimization.cache_aware_orchestration"
    ),
    "CacheAwareTokenOptimizationOrchestrationRequest": (
        "intergrax.runtime.token_optimization.cache_aware_orchestration_contracts"
    ),
    "CacheAwareTokenOptimizationOrchestrationResult": (
        "intergrax.runtime.token_optimization.cache_aware_orchestration_contracts"
    ),
    "CacheAwareTokenOptimizationOrchestrationStatus": (
        "intergrax.runtime.token_optimization.cache_aware_orchestration_contracts"
    ),
    "cache_aware_orchestration_result_to_safe_dict": (
        "intergrax.runtime.token_optimization.cache_aware_orchestration"
    ),
    "CacheAwareCompactionSignalNormalizationRequest": (
        "intergrax.runtime.token_optimization.cache_signal_normalization_contracts"
    ),
    "CacheAwareCompactionSignalNormalizationResult": (
        "intergrax.runtime.token_optimization.cache_signal_normalization_contracts"
    ),
    "CacheAwareCompactionSignalNormalizationStatus": (
        "intergrax.runtime.token_optimization.cache_signal_normalization_contracts"
    ),
    "CacheAwareCompactionSignalNormalizationReason": (
        "intergrax.runtime.token_optimization.cache_signal_normalization_contracts"
    ),
    "CacheSignalValueSource": (
        "intergrax.runtime.token_optimization.cache_signal_normalization_contracts"
    ),
    "normalize_cache_aware_compaction_signals": (
        "intergrax.runtime.token_optimization.cache_signal_normalization"
    ),
    "prompt_cache_usage_snapshot_from_adapter_response": (
        "intergrax.runtime.token_optimization.cache_signal_normalization"
    ),
    "cache_signal_normalization_result_to_safe_dict": (
        "intergrax.runtime.token_optimization.cache_signal_normalization"
    ),
    "CacheAwareTokenOptimizationRuntime": (
        "intergrax.runtime.token_optimization.cache_aware_runtime"
    ),
    "CacheAwareTokenOptimizationRuntimeRequest": (
        "intergrax.runtime.token_optimization.cache_aware_runtime_contracts"
    ),
    "CacheAwareTokenOptimizationRuntimeResult": (
        "intergrax.runtime.token_optimization.cache_aware_runtime_contracts"
    ),
    "CacheAwareTokenOptimizationRuntimeStatus": (
        "intergrax.runtime.token_optimization.cache_aware_runtime_contracts"
    ),
    "CacheAwareTokenOptimizationEvidenceReconciliationReason": (
        "intergrax.runtime.token_optimization.cache_aware_runtime_contracts"
    ),
    "cache_aware_runtime_result_to_safe_dict": (
        "intergrax.runtime.token_optimization.cache_aware_runtime"
    ),
    "CacheAwareCompactionDecision": "intergrax.runtime.token_optimization.contracts",
    "CacheAwareCompactionReason": "intergrax.runtime.token_optimization.contracts",
    "CacheAwareCompactionTarget": "intergrax.runtime.token_optimization.contracts",
    "CacheAwareCompactionTimingDecision": "intergrax.runtime.token_optimization.contracts",
    "CacheAwareCompactionTimingInput": "intergrax.runtime.token_optimization.contracts",
    "PromptCacheAttribution": "intergrax.runtime.token_optimization.contracts",
    "PromptCacheInvalidationReason": "intergrax.runtime.token_optimization.contracts",
    "PromptCachePolicy": "intergrax.runtime.token_optimization.contracts",
    "PromptCacheProviderCapabilities": "intergrax.runtime.token_optimization.contracts",
    "PromptCacheUsageSnapshot": "intergrax.runtime.token_optimization.contracts",
    "TokenOptimizationLLMRouterRequest": (
        "intergrax.runtime.token_optimization.llm_router_contracts"
    ),
    "TokenOptimizationLLMRouterResult": (
        "intergrax.runtime.token_optimization.llm_router_contracts"
    ),
    "TokenOptimizationRouterStatus": (
        "intergrax.runtime.token_optimization.llm_router_contracts"
    ),
    "decide_cache_aware_compaction_timing": "intergrax.runtime.token_optimization.prompt_cache",
}


def _vllm_capabilities() -> token_optimization.PromptCacheProviderCapabilities:
    return token_optimization.PromptCacheProviderCapabilities(
        provider="vllm",
        supports_prompt_caching=True,
        supports_cache_usage_tokens=True,
    )


def _usage(
    *,
    cached_input_tokens: int | None = None,
    uncached_input_tokens: int | None = None,
    cache_hit_ratio: float | None = None,
) -> token_optimization.PromptCacheUsageSnapshot:
    return token_optimization.PromptCacheUsageSnapshot(
        provider="vllm",
        model="vllm-test",
        cached_input_tokens=cached_input_tokens,
        uncached_input_tokens=uncached_input_tokens,
        cache_hit_ratio=cache_hit_ratio,
    )


def _attribution(
    usage: token_optimization.PromptCacheUsageSnapshot | None,
) -> token_optimization.PromptCacheAttribution:
    return token_optimization.PromptCacheAttribution(
        policy=token_optimization.PromptCachePolicy(
            enabled=True,
            mode=token_optimization.PromptCacheMode.PROVIDER_DEFAULT,
        ),
        provider_capabilities=_vllm_capabilities(),
        usage=usage,
        prefix_stability_status=token_optimization.PREFIX_STABILITY_STABLE,
        invalidation_reason=token_optimization.PromptCacheInvalidationReason.NONE,
    )


def _vllm_response(*, cached_input_tokens: int) -> LLMAdapterResponse:
    return build_adapter_response(
        content="assistant-response",
        provider="vllm",
        model="vllm-test",
        usage=LLMTokenUsage.from_counts(
            input_tokens=1000,
            output_tokens=10,
            cached_input_tokens=cached_input_tokens,
        ),
        provider_extensions=LLMProviderExtensions(
            usage_source="sdk",
            vllm=VllmProviderExtensions(prompt_tokens_details_reported=True),
        ),
    )


def _router_request() -> token_optimization.TokenOptimizationLLMRouterRequest:
    return token_optimization.TokenOptimizationLLMRouterRequest(
        request=token_optimization.TokenOptimizationRequest(
            content="SYNTH-ALPHA\nSYNTH-ALPHA\n",
            source_type=token_optimization.TokenOptimizationSourceType.RAG_CONTEXT_PACK,
            policy=token_optimization.TokenOptimizationPolicy(
                enabled=True,
                profile=token_optimization.TokenOptimizationProfile.CONSERVATIVE,
                allow_lossy=True,
            ),
        ),
        policy=token_optimization.TokenOptimizationLLMRouterPolicy(),
        request_id="public-contract-test",
    )


class _CountingNativeToolsAdapter(LLMAdapter):
    provider = "fake-native"
    model = "fake-native"

    def __init__(self, *, decision: object | None = None) -> None:
        super().__init__()
        self._decision = decision
        self.generate_with_tools_calls = 0

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
        self.generate_with_tools_calls += 1
        if self._decision is None:
            return build_adapter_response(content="")
        return build_adapter_response(
            content="",
            tool_calls=[
                LLMToolCall(
                    id="call-1",
                    name=token_optimization.ROUTER_TOOL_ID,
                    arguments_json=self._decision.model_dump_json(),
                )
            ],
        )


def _router_decision() -> token_optimization.TokenOptimizationRouterToolInput:
    return token_optimization.TokenOptimizationRouterToolInput(
        configuration_id=token_optimization.TokenOptimizationRouterConfigurationId.EXACT_ONLY,
        reason_code=token_optimization.TokenOptimizationRouterReasonCode.EXACT_DUPLICATES,
        risk=token_optimization.TokenOptimizationRouterRisk.LOW,
        review_required=False,
        confidence=0.9,
    )


class _CountingRouter(token_optimization.TokenOptimizationLLMRouter):
    def __init__(self, adapter: LLMAdapter) -> None:
        super().__init__(adapter=adapter)
        self.route_calls = 0
        self.execute_routed_calls = 0

    def route(
        self,
        router_request: token_optimization.TokenOptimizationLLMRouterRequest,
    ) -> token_optimization.TokenOptimizationLLMRouterResult:
        self.route_calls += 1
        return super().route(router_request)

    def execute_routed(
        self,
        router_request: token_optimization.TokenOptimizationLLMRouterRequest,
        routed_result: token_optimization.TokenOptimizationLLMRouterResult,
    ) -> token_optimization.TokenOptimizationLLMRouterResult:
        self.execute_routed_calls += 1
        return super().execute_routed(router_request, routed_result)


def _runtime_with_counting_instrumentation() -> tuple[
    token_optimization.CacheAwareTokenOptimizationRuntime,
    _CountingRouter,
    _CountingNativeToolsAdapter,
]:
    adapter = _CountingNativeToolsAdapter(decision=_router_decision())
    router = _CountingRouter(adapter=adapter)
    orchestrator = token_optimization.CacheAwareTokenOptimizationOrchestrator(router=router)
    runtime = token_optimization.CacheAwareTokenOptimizationRuntime(orchestrator=orchestrator)
    return runtime, router, adapter


@pytest.mark.parametrize("symbol_name", _FROZEN_TOKEN_10D_SYMBOLS)
def test_frozen_token_10d_symbol_available_through_package_root(symbol_name: str) -> None:
    assert hasattr(token_optimization, symbol_name)
    assert symbol_name in token_optimization.__all__


@pytest.mark.parametrize("symbol_name", _FROZEN_TOKEN_10D_SYMBOLS)
def test_frozen_token_10d_symbol_has_canonical_module_identity(symbol_name: str) -> None:
    exported = object.__getattribute__(token_optimization, symbol_name)
    assert exported.__module__ == _CANONICAL_MODULES[symbol_name]


def test_public_runtime_construction_through_package_root() -> None:
    orchestrator = MagicMock(spec=token_optimization.CacheAwareTokenOptimizationOrchestrator)
    runtime = token_optimization.CacheAwareTokenOptimizationRuntime(orchestrator=orchestrator)
    assert isinstance(runtime, token_optimization.CacheAwareTokenOptimizationRuntime)


def test_public_request_construction_through_package_root() -> None:
    attribution = _attribution(_usage(cached_input_tokens=800, uncached_input_tokens=200))
    request = token_optimization.CacheAwareTokenOptimizationRuntimeRequest(
        router_request=_router_request(),
        cache_attribution=attribution,
        target=token_optimization.CacheAwareCompactionTarget.STABLE_PREFIX,
        estimated_content_reduction_chars=50,
    )
    assert isinstance(request, token_optimization.CacheAwareTokenOptimizationRuntimeRequest)
    assert isinstance(attribution, token_optimization.PromptCacheAttribution)
    assert request.target is token_optimization.CacheAwareCompactionTarget.STABLE_PREFIX


def test_conflict_rejection_fail_closed_without_router_or_pipeline() -> None:
    usage = _usage(cached_input_tokens=800)
    attribution = _attribution(usage)
    response = _vllm_response(cached_input_tokens=500)
    runtime, router, adapter = _runtime_with_counting_instrumentation()

    result = runtime.run(
        token_optimization.CacheAwareTokenOptimizationRuntimeRequest(
            router_request=_router_request(),
            cache_attribution=attribution,
            adapter_response=response,
            target=token_optimization.CacheAwareCompactionTarget.COLD_HISTORY,
        )
    )

    assert (
        result.status
        is token_optimization.CacheAwareTokenOptimizationRuntimeStatus.SIGNALS_REJECTED
    )
    assert result.executed is False
    assert result.orchestration_result is None
    assert router.route_calls == 0
    assert router.execute_routed_calls == 0
    assert adapter.generate_with_tools_calls == 0


def test_non_rejected_public_flow_deferred_without_execution() -> None:
    attribution = _attribution(
        _usage(cached_input_tokens=800, uncached_input_tokens=200, cache_hit_ratio=0.8)
    )
    runtime, router, adapter = _runtime_with_counting_instrumentation()

    result = runtime.run(
        token_optimization.CacheAwareTokenOptimizationRuntimeRequest(
            router_request=_router_request(),
            cache_attribution=attribution,
            target=token_optimization.CacheAwareCompactionTarget.STABLE_PREFIX,
            estimated_content_reduction_chars=50,
        )
    )

    assert result.status in {
        token_optimization.CacheAwareTokenOptimizationRuntimeStatus.DEFERRED,
        token_optimization.CacheAwareTokenOptimizationRuntimeStatus.BYPASSED,
        token_optimization.CacheAwareTokenOptimizationRuntimeStatus.ROUTER_TERMINAL,
    }
    assert result.normalization_result is not None
    assert result.orchestration_result is not None
    assert result.executed is False
    assert router.route_calls == 1
    assert router.execute_routed_calls == 0
    assert adapter.generate_with_tools_calls == 1


def test_safe_serializer_exposes_no_raw_content() -> None:
    attribution = _attribution(
        _usage(cached_input_tokens=0, uncached_input_tokens=1000, cache_hit_ratio=0.0)
    )
    runtime, _router, _adapter = _runtime_with_counting_instrumentation()
    result = runtime.run(
        token_optimization.CacheAwareTokenOptimizationRuntimeRequest(
            router_request=_router_request(),
            cache_attribution=attribution,
            target=token_optimization.CacheAwareCompactionTarget.COLD_HISTORY,
        )
    )

    safe = token_optimization.cache_aware_runtime_result_to_safe_dict(result)
    dumped = json.dumps(safe)

    assert safe["raw_content_included"] is False
    assert "SYNTH-ALPHA" not in dumped
    assert "assistant-response" not in dumped


def test_lower_level_public_apis_remain_available() -> None:
    assert callable(token_optimization.normalize_cache_aware_compaction_signals)
    assert callable(token_optimization.decide_cache_aware_compaction_timing)
    assert token_optimization.CacheAwareTokenOptimizationOrchestrator is not None


def test_package_imports_without_live_infrastructure() -> None:
    assert token_optimization.CacheAwareTokenOptimizationRuntime is not None


def test_public_runtime_has_no_application_dependencies() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[4] / "intergrax" / "runtime" / "token_optimization"
    targets = (
        root / "__init__.py",
        root / "cache_aware_runtime.py",
        root / "cache_aware_runtime_contracts.py",
    )
    forbidden = (
        "applications.local_workspace_application",
        "applications/local_workspace_application",
        "vendor_knowledge",
    )
    for path in targets:
        content = path.read_text(encoding="utf-8")
        for marker in forbidden:
            assert marker not in content
