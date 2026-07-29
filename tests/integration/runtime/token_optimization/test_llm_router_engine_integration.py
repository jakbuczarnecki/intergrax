# © Artur Czarnecki. All rights reserved.

"""Integration tests for Token Optimization LLM router engine execution (TOKEN-9)."""

from __future__ import annotations

import json
from typing import Any, Sequence

import pytest

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters._shared.adapter_response_builders import build_adapter_response
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.contracts.structured_result import LLMStructuredResult
from intergrax.llm_adapters.contracts.tool_call import LLMToolCall
from intergrax.llm_adapters.providers.ollama_capabilities import (
    OllamaCapabilityResolutionSource,
    OllamaModelCapabilities,
)
from intergrax.runtime.token_optimization.contracts import (
    ProtectedRegion,
    ProtectedRegionKind,
    TokenOptimizationPolicy,
    TokenOptimizationProfile,
    TokenOptimizationRequest,
    TokenOptimizationSourceType,
)
from intergrax.runtime.token_optimization.llm_router import (
    ROUTER_TOOL_ID,
    TokenOptimizationLLMRouter,
    token_optimization_router_result_to_safe_dict,
)
from intergrax.runtime.token_optimization.llm_router_contracts import (
    TokenOptimizationLLMRouterPolicy,
    TokenOptimizationLLMRouterRequest,
    TokenOptimizationRouterConfigurationId,
    TokenOptimizationRouterReason,
    TokenOptimizationRouterReasonCode,
    TokenOptimizationRouterRisk,
    TokenOptimizationRouterStatus,
    TokenOptimizationRouterToolInput,
    TokenOptimizationRouterTransport,
)
from tests.fixtures.token_optimization.llm_router_corpus import LLM_ROUTER_CORPUS

pytestmark = pytest.mark.integration


def _decision(
    configuration_id: TokenOptimizationRouterConfigurationId,
    *,
    review_required: bool = False,
    confidence: float = 0.95,
    reason_code: TokenOptimizationRouterReasonCode = TokenOptimizationRouterReasonCode.EXACT_DUPLICATES,
) -> TokenOptimizationRouterToolInput:
    return TokenOptimizationRouterToolInput(
        configuration_id=configuration_id,
        reason_code=reason_code,
        risk=TokenOptimizationRouterRisk.LOW,
        review_required=review_required,
        confidence=confidence,
    )


class FakeNativeToolsAdapter(LLMAdapter):
    provider = "fake-native"
    model = "fake-native"

    def __init__(
        self,
        *,
        decision: TokenOptimizationRouterToolInput | None = None,
        tool_calls: tuple[LLMToolCall, ...] | None = None,
        fail_native: bool = False,
    ) -> None:
        super().__init__()
        self._decision = decision
        self._tool_calls = tool_calls
        self._fail_native = fail_native
        self.structured_calls = 0
        self.generate_with_tools_calls = 0
        self.generate_structured_calls = 0

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
        if self._fail_native:
            raise RuntimeError("native failure")
        if self._tool_calls is not None:
            return build_adapter_response(content="", tool_calls=self._tool_calls)
        assert self._decision is not None
        return build_adapter_response(
            content="",
            tool_calls=(
                LLMToolCall(
                    id="integration-call-1",
                    name=ROUTER_TOOL_ID,
                    arguments_json=self._decision.model_dump_json(),
                ),
            ),
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
        self.structured_calls += 1
        self.generate_structured_calls += 1
        raise AssertionError("structured fallback must not run after native failure")


class FakeStructuredOutputAdapter(LLMAdapter):
    provider = "fake-structured"
    model = "fake-structured"

    def __init__(self, *, decision: TokenOptimizationRouterToolInput) -> None:
        super().__init__()
        self._decision = decision

    @property
    def context_window_tokens(self) -> int:
        return 8192

    def supports_tools(self) -> bool:
        return False

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

    def generate_structured(
        self,
        messages: Sequence[ChatMessage],
        output_model: type,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        run_id: str | None = None,
    ) -> LLMStructuredResult[Any]:
        return LLMStructuredResult(
            parsed=self._decision,
            response=build_adapter_response(content=self._decision.model_dump_json()),
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
        request_id=f"integration-{case.case_id}",
    )


def test_native_exact_only_executes_real_pipeline() -> None:
    adapter = FakeNativeToolsAdapter(
        decision=_decision(TokenOptimizationRouterConfigurationId.EXACT_ONLY)
    )
    router = TokenOptimizationLLMRouter(adapter=adapter)
    result = router.route_and_execute(_request_from_corpus("router.rag_exact_duplicates"))
    assert result.transport is TokenOptimizationRouterTransport.NATIVE_TOOLS
    assert result.executed is True
    assert result.pipeline_result is not None
    assert "builtin.exact_deduplication" in result.pipeline_result.applied_layer_ids
    assert result.pipeline_result.original_content != result.pipeline_result.final_content


def test_lossy_selection_blocked_by_policy() -> None:
    case = next(item for item in LLM_ROUTER_CORPUS if item.case_id == "router.tool_noisy_output")
    adapter = FakeNativeToolsAdapter(
        decision=_decision(
            TokenOptimizationRouterConfigurationId.EXTRACTIVE_ONLY,
            reason_code=TokenOptimizationRouterReasonCode.NOISY_TOOL_OUTPUT,
        )
    )
    router = TokenOptimizationLLMRouter(adapter=adapter)
    request = TokenOptimizationLLMRouterRequest(
        request=TokenOptimizationRequest(
            content=case.content,
            source_type=case.source_type,
            policy=TokenOptimizationPolicy(
                enabled=True,
                profile=TokenOptimizationProfile.CONSERVATIVE,
                allow_lossy=False,
            ),
        ),
        policy=TokenOptimizationLLMRouterPolicy(),
        request_id="integration-lossy-blocked",
    )
    result = router.route_and_execute(request)
    assert result.executed is False
    assert result.reason is TokenOptimizationRouterReason.LOSSY_NOT_ALLOWED


def test_noisy_tool_output_executes_extractive_filtering() -> None:
    adapter = FakeNativeToolsAdapter(
        decision=_decision(
            TokenOptimizationRouterConfigurationId.EXTRACTIVE_ONLY,
            reason_code=TokenOptimizationRouterReasonCode.NOISY_TOOL_OUTPUT,
        )
    )
    result = TokenOptimizationLLMRouter(adapter=adapter).route_and_execute(
        _request_from_corpus("router.tool_noisy_output")
    )
    assert result.executed is True
    assert "builtin.extractive_filtering" in result.pipeline_result.applied_layer_ids


def test_packing_executes_with_typed_input() -> None:
    adapter = FakeNativeToolsAdapter(
        decision=_decision(
            TokenOptimizationRouterConfigurationId.PACKING_ONLY,
            reason_code=TokenOptimizationRouterReasonCode.PRIORITY_PACKING,
        )
    )
    result = TokenOptimizationLLMRouter(adapter=adapter).route_and_execute(
        _request_from_corpus("router.rag_priority_packing")
    )
    assert result.executed is True
    assert "builtin.budget_aware_context_packing" in result.pipeline_result.applied_layer_ids


def test_packing_blocked_without_input() -> None:
    adapter = FakeNativeToolsAdapter(
        decision=_decision(
            TokenOptimizationRouterConfigurationId.PACKING_ONLY,
            reason_code=TokenOptimizationRouterReasonCode.PRIORITY_PACKING,
        )
    )
    request = TokenOptimizationLLMRouterRequest(
        request=TokenOptimizationRequest(
            content="fragment-a\nfragment-b",
            source_type=TokenOptimizationSourceType.RAG_CONTEXT_PACK,
            policy=TokenOptimizationPolicy(
                enabled=True,
                profile=TokenOptimizationProfile.CONSERVATIVE,
                allow_lossy=False,
            ),
        ),
        policy=TokenOptimizationLLMRouterPolicy(),
        request_id="integration-packing-missing",
    )
    result = TokenOptimizationLLMRouter(adapter=adapter).route(request)
    assert result.reason is TokenOptimizationRouterReason.PACKING_INPUT_REQUIRED


def test_protected_lossy_does_not_execute() -> None:
    adapter = FakeNativeToolsAdapter(
        decision=_decision(
            TokenOptimizationRouterConfigurationId.EXTRACTIVE_ONLY,
            reason_code=TokenOptimizationRouterReasonCode.NOISY_TOOL_OUTPUT,
        )
    )
    result = TokenOptimizationLLMRouter(adapter=adapter).route_and_execute(
        _request_from_corpus("router.protected_noisy_output")
    )
    assert result.executed is False
    assert result.status is TokenOptimizationRouterStatus.REVIEW_REQUIRED


def test_protected_lossless_exact_still_available() -> None:
    adapter = FakeNativeToolsAdapter(
        decision=_decision(TokenOptimizationRouterConfigurationId.EXACT_ONLY)
    )
    duplicate_case = next(
        item for item in LLM_ROUTER_CORPUS if item.case_id == "router.rag_exact_duplicates"
    )
    protected_case = next(
        item for item in LLM_ROUTER_CORPUS if item.case_id == "router.protected_noisy_output"
    )
    request = TokenOptimizationLLMRouterRequest(
        request=TokenOptimizationRequest(
            content=duplicate_case.content,
            source_type=duplicate_case.source_type,
            policy=duplicate_case.policy,
            protected_regions=protected_case.protected_regions,
        ),
        policy=TokenOptimizationLLMRouterPolicy(),
        request_id="integration-protected-lossless",
    )
    result = TokenOptimizationLLMRouter(adapter=adapter).route_and_execute(request)
    assert result.executed is True
    assert result.pipeline_result is not None
    assert (
        "builtin.exact_deduplication" in result.pipeline_result.applied_layer_ids
        or "builtin.exact_deduplication" in result.pipeline_result.bypassed_layer_ids
    )


def test_no_optimization_skips_engine() -> None:
    adapter = FakeNativeToolsAdapter(
        decision=_decision(
            TokenOptimizationRouterConfigurationId.NO_OPTIMIZATION,
            reason_code=TokenOptimizationRouterReasonCode.CLEAN_NO_OP,
        )
    )
    result = TokenOptimizationLLMRouter(adapter=adapter).route_and_execute(
        _request_from_corpus("router.clean_short_output")
    )
    assert result.status is TokenOptimizationRouterStatus.NO_OPTIMIZATION
    assert result.pipeline_result is None


def test_structured_output_fallback_reaches_same_compiler() -> None:
    adapter = FakeStructuredOutputAdapter(
        decision=_decision(TokenOptimizationRouterConfigurationId.EXACT_ONLY)
    )
    result = TokenOptimizationLLMRouter(adapter=adapter).route_and_execute(
        _request_from_corpus("router.rag_exact_duplicates")
    )
    assert result.transport is TokenOptimizationRouterTransport.STRUCTURED_OUTPUT
    assert result.executed is True


def test_native_failure_does_not_switch_transport() -> None:
    adapter = FakeNativeToolsAdapter(fail_native=True)
    result = TokenOptimizationLLMRouter(adapter=adapter).route(
        _request_from_corpus("router.rag_exact_duplicates")
    )
    assert result.transport is TokenOptimizationRouterTransport.NATIVE_TOOLS
    assert result.status is TokenOptimizationRouterStatus.LLM_ERROR
    assert adapter.structured_calls == 0


def test_safe_reporting_remains_content_safe() -> None:
    adapter = FakeNativeToolsAdapter(
        decision=_decision(TokenOptimizationRouterConfigurationId.EXACT_ONLY)
    )
    result = TokenOptimizationLLMRouter(adapter=adapter).route_and_execute(
        _request_from_corpus("router.rag_exact_duplicates")
    )
    safe = token_optimization_router_result_to_safe_dict(result)
    dumped = json.dumps(safe)
    assert "PROTECTED-SYNTH" not in dumped
    assert "SYNTH-EVIDENCE" not in dumped


class _OllamaLikeAdapter(LLMAdapter):
    provider = "ollama"
    model = "legacy:7b"

    def __init__(self, *, capabilities: OllamaModelCapabilities) -> None:
        super().__init__()
        self._model_capabilities = capabilities
        self.structured_calls = 0

    @property
    def context_window_tokens(self) -> int:
        return 8192

    @property
    def model_capabilities(self) -> OllamaModelCapabilities:
        return self._model_capabilities

    def supports_tools(self) -> bool:
        return self._model_capabilities.supports_tools

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

    def generate_structured(
        self,
        messages: Sequence[ChatMessage],
        output_model: type,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        run_id: str | None = None,
    ) -> LLMStructuredResult[Any]:
        self.structured_calls += 1
        return LLMStructuredResult(
            parsed=_decision(TokenOptimizationRouterConfigurationId.EXACT_ONLY),
            response=build_adapter_response(content="{}"),
        )

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
            tool_calls=(
                LLMToolCall(
                    id="integration-call-1",
                    name=ROUTER_TOOL_ID,
                    arguments_json=_decision(
                        TokenOptimizationRouterConfigurationId.EXACT_ONLY
                    ).model_dump_json(),
                ),
            ),
        )


def test_policy_disabled_route_and_execute_does_not_call_adapter() -> None:
    adapter = FakeNativeToolsAdapter(
        decision=_decision(TokenOptimizationRouterConfigurationId.EXACT_ONLY)
    )
    request = TokenOptimizationLLMRouterRequest(
        request=TokenOptimizationRequest(
            content="x",
            source_type=TokenOptimizationSourceType.TOOL_OUTPUT,
            policy=TokenOptimizationPolicy(enabled=False, profile=TokenOptimizationProfile.OFF),
        ),
        policy=TokenOptimizationLLMRouterPolicy(),
        request_id="integration-policy-disabled",
    )
    result = TokenOptimizationLLMRouter(adapter=adapter).route_and_execute(request)
    assert result.reason is TokenOptimizationRouterReason.POLICY_DISABLED
    assert adapter.generate_with_tools_calls == 0
    assert adapter.structured_calls == 0


def test_profile_off_route_and_execute_does_not_call_adapter() -> None:
    adapter = FakeNativeToolsAdapter(
        decision=_decision(TokenOptimizationRouterConfigurationId.EXACT_ONLY)
    )
    request = TokenOptimizationLLMRouterRequest(
        request=TokenOptimizationRequest(
            content="x",
            source_type=TokenOptimizationSourceType.TOOL_OUTPUT,
            policy=TokenOptimizationPolicy(enabled=True, profile=TokenOptimizationProfile.OFF),
        ),
        policy=TokenOptimizationLLMRouterPolicy(),
        request_id="integration-profile-off",
    )
    result = TokenOptimizationLLMRouter(adapter=adapter).route_and_execute(request)
    assert result.reason is TokenOptimizationRouterReason.PROFILE_OFF
    assert adapter.generate_with_tools_calls == 0
    assert adapter.structured_calls == 0


def test_unresolved_capability_does_not_call_structured_output() -> None:
    adapter = _OllamaLikeAdapter(
        capabilities=OllamaModelCapabilities(
            model="legacy:7b",
            capabilities=frozenset(),
            resolved=False,
            source=OllamaCapabilityResolutionSource.UNAVAILABLE,
            error_type="ConnectionError",
        )
    )
    result = TokenOptimizationLLMRouter(adapter=adapter).route_and_execute(
        _request_from_corpus("router.rag_exact_duplicates")
    )
    assert result.reason is TokenOptimizationRouterReason.CAPABILITY_RESOLUTION_FAILED
    assert adapter.structured_calls == 0


def test_resolved_no_tools_capability_uses_structured_output() -> None:
    adapter = _OllamaLikeAdapter(
        capabilities=OllamaModelCapabilities(
            model="legacy:7b",
            capabilities=frozenset({"completion"}),
            resolved=True,
            source=OllamaCapabilityResolutionSource.EXPLICIT_TEST_OVERRIDE,
        )
    )
    result = TokenOptimizationLLMRouter(adapter=adapter).route_and_execute(
        _request_from_corpus("router.rag_exact_duplicates")
    )
    assert result.transport is TokenOptimizationRouterTransport.STRUCTURED_OUTPUT
    assert adapter.structured_calls == 1
    assert result.executed is True


def test_lossy_disallowed_request_never_executes_lossy_configuration() -> None:
    adapter = FakeNativeToolsAdapter(
        decision=_decision(
            TokenOptimizationRouterConfigurationId.EXTRACTIVE_ONLY,
            reason_code=TokenOptimizationRouterReasonCode.NOISY_TOOL_OUTPUT,
        )
    )
    case = next(item for item in LLM_ROUTER_CORPUS if item.case_id == "router.lossy_disallowed")
    request = TokenOptimizationLLMRouterRequest(
        request=TokenOptimizationRequest(
            content=case.content,
            source_type=case.source_type,
            policy=case.policy,
        ),
        policy=TokenOptimizationLLMRouterPolicy(),
        request_id="integration-lossy-disallowed",
    )
    result = TokenOptimizationLLMRouter(adapter=adapter).route_and_execute(request)
    assert result.executed is False
    assert result.reason is TokenOptimizationRouterReason.LOSSY_NOT_ALLOWED


def test_mixed_bypass_apply_execution_order_reported_canonically() -> None:
    adapter = FakeNativeToolsAdapter(
        decision=_decision(
            TokenOptimizationRouterConfigurationId.EXACT_THEN_PACKING,
            reason_code=TokenOptimizationRouterReasonCode.MIXED_DEDUPLICATION_PACKING,
        )
    )
    case = next(item for item in LLM_ROUTER_CORPUS if item.case_id == "router.rag_mixed_dedupe_packing")
    request = TokenOptimizationLLMRouterRequest(
        request=TokenOptimizationRequest(
            content="unique-line-one\nunique-line-two\n",
            source_type=case.source_type,
            policy=case.policy,
            metadata=dict(case.metadata),
        ),
        policy=TokenOptimizationLLMRouterPolicy(),
        request_id="integration-mixed-order",
    )
    result = TokenOptimizationLLMRouter(adapter=adapter).route_and_execute(request)
    assert result.executed is True
    assert result.pipeline_result is not None
    canonical = list(result.pipeline_result.receipt_metadata["executed_layer_ids"])
    grouped = list(result.pipeline_result.applied_layer_ids) + list(
        result.pipeline_result.bypassed_layer_ids
    )
    assert canonical != grouped
    safe = token_optimization_router_result_to_safe_dict(result)
    assert safe["executed_layer_ids"] == canonical
