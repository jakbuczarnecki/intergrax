# © Artur Czarnecki. All rights reserved.

"""TOKEN-10D-3: cache-aware runtime composition unit tests."""

from __future__ import annotations

import inspect
import json
from typing import Any, Sequence
from unittest.mock import MagicMock, patch

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
    CacheAwareTokenOptimizationOrchestrationResult,
    CacheAwareTokenOptimizationOrchestrationStatus,
)
from intergrax.runtime.token_optimization.cache_aware_runtime import (
    CacheAwareTokenOptimizationRuntime,
    cache_aware_runtime_result_to_safe_dict,
)
from intergrax.runtime.token_optimization.cache_aware_runtime_contracts import (
    CacheAwareTokenOptimizationEvidenceReconciliationReason,
    CacheAwareTokenOptimizationRuntimeRequest,
    CacheAwareTokenOptimizationRuntimeResult,
    CacheAwareTokenOptimizationRuntimeStatus,
)
from intergrax.runtime.token_optimization.cache_signal_normalization_contracts import (
    CacheAwareCompactionSignalNormalizationStatus,
)
from intergrax.runtime.token_optimization.contracts import (
    CacheAwareCompactionReason,
    CacheAwareCompactionTarget,
    PromptCacheAttribution,
    PromptCacheInvalidationReason,
    PromptCacheMode,
    PromptCachePolicy,
    PromptCacheProviderCapabilities,
    PromptCacheUsageSnapshot,
    TokenOptimizationAttribution,
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
    TokenOptimizationRouterConfigurationId,
    TokenOptimizationRouterReasonCode,
    TokenOptimizationRouterRisk,
    TokenOptimizationRouterStatus,
)
from intergrax.runtime.token_optimization.pipeline import TokenOptimizationPipelineRunner
from intergrax.runtime.token_optimization.prompt_cache import PREFIX_STABILITY_STABLE

pytestmark = pytest.mark.unit


def _vllm_capabilities() -> PromptCacheProviderCapabilities:
    return PromptCacheProviderCapabilities(
        provider="vllm",
        supports_prompt_caching=True,
        supports_cache_usage_tokens=True,
    )


def _usage(
    *,
    cached_input_tokens: int | None = None,
    uncached_input_tokens: int | None = None,
    cache_hit_ratio: float | None = None,
    cache_read_tokens: int | None = None,
    cache_creation_tokens: int | None = None,
    provider: str = "vllm",
    model: str | None = "vllm-test",
) -> PromptCacheUsageSnapshot:
    return PromptCacheUsageSnapshot(
        provider=provider,
        model=model,
        cached_input_tokens=cached_input_tokens,
        uncached_input_tokens=uncached_input_tokens,
        cache_hit_ratio=cache_hit_ratio,
        cache_read_tokens=cache_read_tokens,
        cache_creation_tokens=cache_creation_tokens,
    )


def _attribution(
    usage: PromptCacheUsageSnapshot | None,
    *,
    capabilities: PromptCacheProviderCapabilities | None = None,
) -> PromptCacheAttribution:
    return PromptCacheAttribution(
        policy=PromptCachePolicy(enabled=True, mode=PromptCacheMode.PROVIDER_DEFAULT),
        provider_capabilities=capabilities or _vllm_capabilities(),
        usage=usage,
        prefix_stability_status=PREFIX_STABILITY_STABLE,
        invalidation_reason=PromptCacheInvalidationReason.NONE,
    )


def _vllm_response(
    *,
    cached_input_tokens: int = 0,
    details_reported: bool = True,
    provider: str = "vllm",
    model: str = "vllm-test",
) -> LLMAdapterResponse:
    return build_adapter_response(
        content="assistant-response",
        provider=provider,
        model=model,
        usage=LLMTokenUsage.from_counts(
            input_tokens=1000,
            output_tokens=10,
            cached_input_tokens=cached_input_tokens,
        ),
        provider_extensions=LLMProviderExtensions(
            usage_source="sdk",
            vllm=VllmProviderExtensions(
                prompt_tokens_details_reported=details_reported,
            ),
        ),
    )


def _router_request(
    *,
    attribution: TokenOptimizationAttribution | None = None,
) -> TokenOptimizationLLMRouterRequest:
    return TokenOptimizationLLMRouterRequest(
        request=TokenOptimizationRequest(
            content="SYNTH-ALPHA\nSYNTH-ALPHA\n",
            source_type=TokenOptimizationSourceType.RAG_CONTEXT_PACK,
            policy=TokenOptimizationPolicy(
                enabled=True,
                profile=TokenOptimizationProfile.CONSERVATIVE,
                allow_lossy=True,
            ),
            attribution=attribution,
        ),
        policy=TokenOptimizationLLMRouterPolicy(),
        request_id="runtime-test-1",
    )


class _NativeToolsAdapter(LLMAdapter):
    provider = "fake-native"
    model = "fake-native"

    def __init__(
        self,
        *,
        decision: object | None = None,
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


def _decision(
    configuration_id: TokenOptimizationRouterConfigurationId,
    *,
    review_required: bool = False,
    confidence: float = 0.9,
    reason_code: TokenOptimizationRouterReasonCode = TokenOptimizationRouterReasonCode.EXACT_DUPLICATES,
) -> TokenOptimizationRouterToolInput:
    from intergrax.runtime.token_optimization.llm_router_contracts import (
        TokenOptimizationRouterToolInput,
    )

    return TokenOptimizationRouterToolInput(
        configuration_id=configuration_id,
        reason_code=reason_code,
        risk=TokenOptimizationRouterRisk.LOW,
        review_required=review_required,
        confidence=confidence,
    )


def _runtime_request(
    *,
    attribution: PromptCacheAttribution,
    target: CacheAwareCompactionTarget = CacheAwareCompactionTarget.COLD_HISTORY,
    adapter_response: LLMAdapterResponse | None = None,
    router_attribution: TokenOptimizationAttribution | None = None,
    estimated_content_reduction_chars: int | None = 200,
) -> CacheAwareTokenOptimizationRuntimeRequest:
    return CacheAwareTokenOptimizationRuntimeRequest(
        router_request=_router_request(attribution=router_attribution),
        cache_attribution=attribution,
        adapter_response=adapter_response,
        target=target,
        estimated_content_reduction_chars=estimated_content_reduction_chars,
    )


def _runtime_with_real_orchestrator(
    *,
    decision: object,
) -> CacheAwareTokenOptimizationRuntime:
    adapter = _NativeToolsAdapter(decision=decision)
    router = TokenOptimizationLLMRouter(adapter=adapter)
    orchestrator = CacheAwareTokenOptimizationOrchestrator(router=router)
    return CacheAwareTokenOptimizationRuntime(orchestrator=orchestrator)


def test_attribution_only_path_calls_normalizer_and_orchestrator_once() -> None:
    attribution = _attribution(_usage(cached_input_tokens=0, uncached_input_tokens=1000, cache_hit_ratio=0.0))
    runtime = _runtime_with_real_orchestrator(
        decision=_decision(TokenOptimizationRouterConfigurationId.EXACT_ONLY)
    )

    with patch(
        "intergrax.runtime.token_optimization.cache_aware_runtime.normalize_cache_aware_compaction_signals",
        wraps=__import__(
            "intergrax.runtime.token_optimization.cache_signal_normalization",
            fromlist=["normalize_cache_aware_compaction_signals"],
        ).normalize_cache_aware_compaction_signals,
    ) as normalize_mock:
        with patch.object(runtime._orchestrator, "orchestrate", wraps=runtime._orchestrator.orchestrate) as orchestrate_mock:
            result = runtime.run(
                _runtime_request(
                    attribution=attribution,
                    target=CacheAwareCompactionTarget.COLD_HISTORY,
                )
            )

    normalize_mock.assert_called_once()
    orchestrate_mock.assert_called_once()
    assert result.status is CacheAwareTokenOptimizationRuntimeStatus.EXECUTED


def test_adapter_fills_missing_usage_immutably() -> None:
    response = _vllm_response(cached_input_tokens=800)
    attribution = _attribution(None)
    runtime = _runtime_with_real_orchestrator(
        decision=_decision(TokenOptimizationRouterConfigurationId.EXACT_ONLY)
    )

    result = runtime.run(
        _runtime_request(
            attribution=attribution,
            adapter_response=response,
            target=CacheAwareCompactionTarget.STABLE_PREFIX,
            estimated_content_reduction_chars=50,
        )
    )

    assert attribution.usage is None
    assert result.reconciled_cache_attribution.usage is not None
    assert result.reconciled_cache_attribution.usage.cached_input_tokens == 800
    assert (
        result.evidence_reconciliation_reason
        is CacheAwareTokenOptimizationEvidenceReconciliationReason.ADAPTER_FILLED_MISSING_USAGE
    )


def test_missing_adapter_cache_details_remain_unknown() -> None:
    response = _vllm_response(cached_input_tokens=0, details_reported=False)
    attribution = _attribution(None)
    runtime = _runtime_with_real_orchestrator(
        decision=_decision(TokenOptimizationRouterConfigurationId.EXACT_ONLY)
    )

    result = runtime.run(
        _runtime_request(
            attribution=attribution,
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


def test_existing_attribution_usage_retained_when_adapter_has_no_details() -> None:
    usage = _usage(cached_input_tokens=800, uncached_input_tokens=200, cache_hit_ratio=0.8)
    attribution = _attribution(usage)
    response = _vllm_response(cached_input_tokens=0, details_reported=False)
    runtime = _runtime_with_real_orchestrator(
        decision=_decision(TokenOptimizationRouterConfigurationId.EXACT_ONLY)
    )

    result = runtime.run(
        _runtime_request(
            attribution=attribution,
            adapter_response=response,
            target=CacheAwareCompactionTarget.STABLE_PREFIX,
            estimated_content_reduction_chars=50,
        )
    )

    assert result.reconciled_cache_attribution.usage == usage
    assert result.normalization_result is not None
    assert result.normalization_result.timing_input is not None
    assert result.normalization_result.timing_input.cache_hot is True


def test_identical_evidence_accepted() -> None:
    usage = _usage(cached_input_tokens=800, uncached_input_tokens=200, cache_hit_ratio=0.8)
    attribution = _attribution(usage)
    response = _vllm_response(cached_input_tokens=800)
    runtime = _runtime_with_real_orchestrator(
        decision=_decision(TokenOptimizationRouterConfigurationId.EXACT_ONLY)
    )

    result = runtime.run(
        _runtime_request(
            attribution=attribution,
            adapter_response=response,
            target=CacheAwareCompactionTarget.STABLE_PREFIX,
            estimated_content_reduction_chars=50,
        )
    )

    assert result.status is not CacheAwareTokenOptimizationRuntimeStatus.SIGNALS_REJECTED
    assert (
        result.evidence_reconciliation_reason
        is CacheAwareTokenOptimizationEvidenceReconciliationReason.IDENTICAL_EVIDENCE
    )


def test_complementary_evidence_merged_without_conflict() -> None:
    usage = _usage(
        cached_input_tokens=800,
        uncached_input_tokens=None,
        cache_hit_ratio=None,
    )
    attribution = _attribution(usage)
    response = _vllm_response(cached_input_tokens=800)
    runtime = _runtime_with_real_orchestrator(
        decision=_decision(TokenOptimizationRouterConfigurationId.EXACT_ONLY)
    )

    result = runtime.run(
        _runtime_request(
            attribution=attribution,
            adapter_response=response,
            target=CacheAwareCompactionTarget.STABLE_PREFIX,
            estimated_content_reduction_chars=50,
        )
    )

    assert result.status is not CacheAwareTokenOptimizationRuntimeStatus.SIGNALS_REJECTED
    assert (
        result.evidence_reconciliation_reason
        is CacheAwareTokenOptimizationEvidenceReconciliationReason.COMPLEMENTARY_MERGE
    )
    assert result.reconciled_cache_attribution.usage is not None
    assert result.reconciled_cache_attribution.usage.uncached_input_tokens == 200


def test_conflicting_cached_tokens_rejected_without_downstream_calls() -> None:
    usage = _usage(cached_input_tokens=800)
    attribution = _attribution(usage)
    response = _vllm_response(cached_input_tokens=500)
    orchestrator = MagicMock(spec=CacheAwareTokenOptimizationOrchestrator)
    runtime = CacheAwareTokenOptimizationRuntime(orchestrator=orchestrator)

    with patch(
        "intergrax.runtime.token_optimization.cache_aware_runtime.normalize_cache_aware_compaction_signals"
    ) as normalize_mock:
        result = runtime.run(
            _runtime_request(attribution=attribution, adapter_response=response)
        )

    assert result.status is CacheAwareTokenOptimizationRuntimeStatus.SIGNALS_REJECTED
    assert (
        result.evidence_reconciliation_reason
        is CacheAwareTokenOptimizationEvidenceReconciliationReason.CONFLICTING_CACHE_EVIDENCE
    )
    normalize_mock.assert_not_called()
    orchestrator.orchestrate.assert_not_called()


def test_provider_mismatch_rejected() -> None:
    usage = _usage(cached_input_tokens=800, provider="vllm")
    attribution = _attribution(usage)
    response = _vllm_response(cached_input_tokens=800, provider="openai")
    orchestrator = MagicMock(spec=CacheAwareTokenOptimizationOrchestrator)
    runtime = CacheAwareTokenOptimizationRuntime(orchestrator=orchestrator)

    result = runtime.run(
        _runtime_request(attribution=attribution, adapter_response=response)
    )

    assert result.status is CacheAwareTokenOptimizationRuntimeStatus.SIGNALS_REJECTED
    assert (
        result.evidence_reconciliation_reason
        is CacheAwareTokenOptimizationEvidenceReconciliationReason.PROVIDER_MISMATCH
    )
    orchestrator.orchestrate.assert_not_called()


def test_model_mismatch_rejected() -> None:
    usage = _usage(cached_input_tokens=800, model="model-a")
    attribution = _attribution(usage)
    response = _vllm_response(cached_input_tokens=800, model="model-b")
    orchestrator = MagicMock(spec=CacheAwareTokenOptimizationOrchestrator)
    runtime = CacheAwareTokenOptimizationRuntime(orchestrator=orchestrator)

    result = runtime.run(
        _runtime_request(attribution=attribution, adapter_response=response)
    )

    assert result.status is CacheAwareTokenOptimizationRuntimeStatus.SIGNALS_REJECTED
    assert (
        result.evidence_reconciliation_reason
        is CacheAwareTokenOptimizationEvidenceReconciliationReason.MODEL_MISMATCH
    )


def test_request_attribution_mismatch_rejected() -> None:
    usage = _usage(cached_input_tokens=800)
    attribution = _attribution(usage)
    runtime = CacheAwareTokenOptimizationRuntime(
        orchestrator=MagicMock(spec=CacheAwareTokenOptimizationOrchestrator)
    )

    result = runtime.run(
        _runtime_request(
            attribution=attribution,
            router_attribution=TokenOptimizationAttribution(provider="anthropic"),
        )
    )

    assert result.status is CacheAwareTokenOptimizationRuntimeStatus.SIGNALS_REJECTED


def test_missing_optional_identity_not_rejected() -> None:
    usage = _usage(cached_input_tokens=800, model=None)
    attribution = _attribution(usage)
    runtime = _runtime_with_real_orchestrator(
        decision=_decision(TokenOptimizationRouterConfigurationId.EXACT_ONLY)
    )

    result = runtime.run(
        _runtime_request(
            attribution=attribution,
            target=CacheAwareCompactionTarget.STABLE_PREFIX,
            estimated_content_reduction_chars=50,
        )
    )

    assert result.status is not CacheAwareTokenOptimizationRuntimeStatus.SIGNALS_REJECTED


def test_extraction_error_rejected() -> None:
    response = build_adapter_response(
        content="bad",
        provider="vllm",
        model="vllm-test",
        usage=LLMTokenUsage.from_counts(
            input_tokens=100,
            output_tokens=1,
            cached_input_tokens=200,
        ),
        provider_extensions=LLMProviderExtensions(usage_source="sdk"),
    )
    attribution = _attribution(None)
    orchestrator = MagicMock(spec=CacheAwareTokenOptimizationOrchestrator)
    runtime = CacheAwareTokenOptimizationRuntime(orchestrator=orchestrator)

    result = runtime.run(
        _runtime_request(attribution=attribution, adapter_response=response)
    )

    assert result.status is CacheAwareTokenOptimizationRuntimeStatus.SIGNALS_REJECTED
    assert (
        result.evidence_reconciliation_reason
        is CacheAwareTokenOptimizationEvidenceReconciliationReason.EXTRACTION_ERROR
    )
    orchestrator.orchestrate.assert_not_called()


def test_normalization_rejected_skips_orchestrator() -> None:
    usage = _usage(cached_input_tokens=800, cache_hit_ratio=0.0)
    attribution = PromptCacheAttribution(
        policy=PromptCachePolicy(enabled=True, mode=PromptCacheMode.PROVIDER_DEFAULT),
        provider_capabilities=PromptCacheProviderCapabilities(
            provider="vllm",
            supports_prompt_caching=False,
        ),
        usage=usage,
        prefix_stability_status=PREFIX_STABILITY_STABLE,
    )
    orchestrator = MagicMock(spec=CacheAwareTokenOptimizationOrchestrator)
    runtime = CacheAwareTokenOptimizationRuntime(orchestrator=orchestrator)

    result = runtime.run(
        _runtime_request(
            attribution=attribution,
            target=CacheAwareCompactionTarget.STABLE_PREFIX,
            estimated_content_reduction_chars=50,
        )
    )

    assert result.status is CacheAwareTokenOptimizationRuntimeStatus.SIGNALS_REJECTED
    assert result.normalization_result is not None
    assert (
        result.normalization_result.status
        is CacheAwareCompactionSignalNormalizationStatus.REJECTED
    )
    orchestrator.orchestrate.assert_not_called()


def test_normalization_partial_passes_timing_input_unchanged() -> None:
    attribution = _attribution(None)
    runtime = _runtime_with_real_orchestrator(
        decision=_decision(TokenOptimizationRouterConfigurationId.EXACT_ONLY)
    )

    with patch.object(runtime._orchestrator, "orchestrate", wraps=runtime._orchestrator.orchestrate) as orchestrate_mock:
        result = runtime.run(
            _runtime_request(
                attribution=attribution,
                adapter_response=_vllm_response(cached_input_tokens=0, details_reported=False),
                target=CacheAwareCompactionTarget.STABLE_PREFIX,
                estimated_content_reduction_chars=100,
            )
        )

    orchestrate_mock.assert_called_once()
    assert result.normalization_result is not None
    assert (
        result.normalization_result.status
        is CacheAwareCompactionSignalNormalizationStatus.PARTIAL
    )
    orchestration_request = orchestrate_mock.call_args.args[0]
    assert isinstance(orchestration_request, CacheAwareTokenOptimizationOrchestrationRequest)
    assert orchestration_request.timing_input is result.normalization_result.timing_input


def test_runtime_executed_has_pipeline_result() -> None:
    attribution = _attribution(
        _usage(cached_input_tokens=0, uncached_input_tokens=1000, cache_hit_ratio=0.0)
    )
    runtime = _runtime_with_real_orchestrator(
        decision=_decision(TokenOptimizationRouterConfigurationId.EXACT_ONLY)
    )

    result = runtime.run(
        _runtime_request(
            attribution=attribution,
            target=CacheAwareCompactionTarget.COLD_HISTORY,
        )
    )

    assert result.status is CacheAwareTokenOptimizationRuntimeStatus.EXECUTED
    assert result.executed is True
    assert result.orchestration_result is not None
    assert result.orchestration_result.pipeline_result is not None


def test_runtime_deferred_preserves_timing_reason() -> None:
    attribution = _attribution(
        _usage(cached_input_tokens=800, uncached_input_tokens=200, cache_hit_ratio=0.8)
    )
    runtime = _runtime_with_real_orchestrator(
        decision=_decision(TokenOptimizationRouterConfigurationId.EXACT_ONLY)
    )

    result = runtime.run(
        _runtime_request(
            attribution=attribution,
            target=CacheAwareCompactionTarget.STABLE_PREFIX,
            estimated_content_reduction_chars=50,
        )
    )

    assert result.status is CacheAwareTokenOptimizationRuntimeStatus.DEFERRED
    assert result.executed is False
    assert result.orchestration_result is not None
    assert result.orchestration_result.timing_decision is not None
    assert (
        result.orchestration_result.timing_decision.reason
        is CacheAwareCompactionReason.CACHE_INVALIDATION_COST_TOO_HIGH
    )


def test_runtime_bypassed_does_not_execute_pipeline() -> None:
    attribution = _attribution(
        _usage(cached_input_tokens=0, uncached_input_tokens=1000, cache_hit_ratio=0.0)
    )
    runtime = _runtime_with_real_orchestrator(
        decision=_decision(TokenOptimizationRouterConfigurationId.EXACT_ONLY)
    )

    with patch.object(TokenOptimizationPipelineRunner, "run") as pipeline_run:
        result = runtime.run(
            CacheAwareTokenOptimizationRuntimeRequest(
                router_request=_router_request(),
                cache_attribution=attribution,
                target=CacheAwareCompactionTarget.DYNAMIC_TAIL,
                estimated_content_reduction_chars=0,
                dynamic_tail_reduction_available=False,
            )
        )

    pipeline_run.assert_not_called()
    assert result.status is CacheAwareTokenOptimizationRuntimeStatus.BYPASSED
    assert result.executed is False


def test_runtime_review_required_flag() -> None:
    attribution = _attribution(None)
    runtime = _runtime_with_real_orchestrator(
        decision=_decision(TokenOptimizationRouterConfigurationId.EXACT_ONLY)
    )

    result = runtime.run(
        _runtime_request(
            attribution=attribution,
            adapter_response=_vllm_response(cached_input_tokens=0, details_reported=False),
            target=CacheAwareCompactionTarget.STABLE_PREFIX,
            estimated_content_reduction_chars=100,
        )
    )

    assert result.status is CacheAwareTokenOptimizationRuntimeStatus.REVIEW_REQUIRED
    assert result.review_required is True
    assert result.executed is False


def test_runtime_router_terminal_preserves_router_reason() -> None:
    attribution = _attribution(
        _usage(cached_input_tokens=0, uncached_input_tokens=1000, cache_hit_ratio=0.0)
    )
    runtime = _runtime_with_real_orchestrator(
        decision=_decision(
            TokenOptimizationRouterConfigurationId.NO_OPTIMIZATION,
            reason_code=TokenOptimizationRouterReasonCode.CLEAN_NO_OP,
        )
    )

    with patch.object(TokenOptimizationPipelineRunner, "run") as pipeline_run:
        result = runtime.run(
            _runtime_request(
                attribution=attribution,
                target=CacheAwareCompactionTarget.COLD_HISTORY,
            )
        )

    pipeline_run.assert_not_called()
    assert result.status is CacheAwareTokenOptimizationRuntimeStatus.ROUTER_TERMINAL
    assert result.orchestration_result is not None
    assert (
        result.orchestration_result.router_result.status
        is TokenOptimizationRouterStatus.NO_OPTIMIZATION
    )
    assert result.executed is False


def test_exactly_once_stage_calls() -> None:
    attribution = _attribution(
        _usage(cached_input_tokens=0, uncached_input_tokens=1000, cache_hit_ratio=0.0)
    )
    response = _vllm_response(cached_input_tokens=0)
    adapter = _NativeToolsAdapter(
        decision=_decision(TokenOptimizationRouterConfigurationId.EXACT_ONLY)
    )
    router = TokenOptimizationLLMRouter(adapter=adapter)
    orchestrator = CacheAwareTokenOptimizationOrchestrator(router=router)
    runtime = CacheAwareTokenOptimizationRuntime(orchestrator=orchestrator)
    pipeline_calls = 0
    original_run = TokenOptimizationPipelineRunner.run

    def tracked_run(self, **kwargs: object) -> object:
        nonlocal pipeline_calls
        pipeline_calls += 1
        return original_run(self, **kwargs)

    with patch(
        "intergrax.runtime.token_optimization.cache_aware_runtime.prompt_cache_usage_snapshot_from_adapter_response"
    ) as extract_mock:
        extract_mock.return_value = _usage(
            cached_input_tokens=0,
            uncached_input_tokens=1000,
            cache_hit_ratio=0.0,
        )
        with patch(
            "intergrax.runtime.token_optimization.cache_aware_runtime.normalize_cache_aware_compaction_signals",
            wraps=__import__(
                "intergrax.runtime.token_optimization.cache_signal_normalization",
                fromlist=["normalize_cache_aware_compaction_signals"],
            ).normalize_cache_aware_compaction_signals,
        ) as normalize_mock:
            with patch.object(
                runtime._orchestrator,
                "orchestrate",
                wraps=runtime._orchestrator.orchestrate,
            ) as orchestrate_mock:
                with patch.object(TokenOptimizationPipelineRunner, "run", tracked_run):
                    runtime.run(
                        _runtime_request(
                            attribution=attribution,
                            adapter_response=response,
                            target=CacheAwareCompactionTarget.COLD_HISTORY,
                        )
                    )

    extract_mock.assert_called_once()
    normalize_mock.assert_called_once()
    orchestrate_mock.assert_called_once()
    assert adapter.route_calls == 1
    assert pipeline_calls == 1


def test_no_fallback_to_route_and_execute() -> None:
    source = inspect.getsource(CacheAwareTokenOptimizationRuntime.run)
    assert "route_and_execute" not in source


def test_result_invariants_rejected() -> None:
    with pytest.raises(ValueError, match="SIGNALS_REJECTED requires orchestration_result=None"):
        CacheAwareTokenOptimizationRuntimeResult(
            status=CacheAwareTokenOptimizationRuntimeStatus.SIGNALS_REJECTED,
            normalization_result=None,
            orchestration_result=MagicMock(),
            reconciled_cache_attribution=_attribution(None),
            adapter_cache_evidence_reported=False,
            evidence_reconciliation_reason=None,
            executed=False,
            review_required=False,
        )


def test_safe_report_allowlist_and_no_raw_content() -> None:
    attribution = _attribution(
        _usage(cached_input_tokens=0, uncached_input_tokens=1000, cache_hit_ratio=0.0)
    )
    runtime = _runtime_with_real_orchestrator(
        decision=_decision(TokenOptimizationRouterConfigurationId.EXACT_ONLY)
    )
    result = runtime.run(
        _runtime_request(
            attribution=attribution,
            adapter_response=_vllm_response(cached_input_tokens=0),
            target=CacheAwareCompactionTarget.COLD_HISTORY,
        )
    )

    safe = cache_aware_runtime_result_to_safe_dict(result)
    dumped = json.dumps(safe)

    assert safe["raw_content_included"] is False
    assert "SYNTH-ALPHA" not in dumped
    assert "assistant-response" not in dumped
    assert set(safe.keys()) <= {
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
