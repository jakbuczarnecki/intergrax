# © Artur Czarnecki. All rights reserved.

"""Unit tests for Token Optimization LLM router service (TOKEN-9)."""

from __future__ import annotations

import json
from typing import Any, Sequence

import pytest

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters._shared.adapter_response_builders import build_adapter_response
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.contracts.structured_result import LLMStructuredResult
from intergrax.llm_adapters.contracts.tool_call import LLMToolCall
from intergrax.llm_adapters.registry.catalog_capabilities import (
    CatalogCapabilityAdapter,
    enrich_adapter_with_catalog_capabilities,
)
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
from intergrax.runtime.token_optimization.layers import BudgetAwarePackingInput
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
    source_type: TokenOptimizationSourceType = TokenOptimizationSourceType.RAG_CONTEXT_PACK,
    policy: TokenOptimizationPolicy | None = None,
    metadata: dict[str, object] | None = None,
    protected_regions: tuple[ProtectedRegion, ...] = (),
) -> TokenOptimizationLLMRouterRequest:
    return TokenOptimizationLLMRouterRequest(
        request=TokenOptimizationRequest(
            content=content,
            source_type=source_type,
            policy=policy
            or TokenOptimizationPolicy(
                enabled=True,
                profile=TokenOptimizationProfile.CONSERVATIVE,
                allow_lossy=True,
            ),
            protected_regions=protected_regions,
            metadata=metadata or {},
        ),
        policy=TokenOptimizationLLMRouterPolicy(),
        request_id="router-test-1",
    )


class _NativeToolsAdapter(LLMAdapter):
    provider = "fake-native"
    model = "fake-native"

    def __init__(
        self,
        *,
        decision: TokenOptimizationRouterToolInput | None = None,
        tool_calls: tuple[LLMToolCall, ...] | None = None,
        raise_on_generate: bool = False,
        structured_calls: int = 0,
    ) -> None:
        super().__init__()
        self._decision = decision
        self._tool_calls = tool_calls
        self._raise_on_generate = raise_on_generate
        self.structured_calls = structured_calls

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
        if self._raise_on_generate:
            raise RuntimeError("native tool failure")
        if self._tool_calls is not None:
            return build_adapter_response(content="", tool_calls=self._tool_calls)
        assert self._decision is not None
        return build_adapter_response(
            content="",
            tool_calls=(
                LLMToolCall(
                    id="call-router-1",
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
        raise AssertionError("structured output must not be used when native tools are available")


class _StructuredOutputAdapter(LLMAdapter):
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


class _UnsupportedAdapter(LLMAdapter):
    provider = "fake-unsupported"
    model = "fake-unsupported"

    @property
    def context_window_tokens(self) -> int:
        return 8192

    def supports_tools(self) -> bool:
        return False

    def supports_structured_output(self) -> bool:
        return False

    def generate_messages(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        run_id: str | None = None,
    ) -> LLMAdapterResponse:
        return build_adapter_response(content="unused")


def test_transport_prefers_native_tools() -> None:
    adapter = _NativeToolsAdapter(decision=_decision(TokenOptimizationRouterConfigurationId.EXACT_ONLY))
    router = TokenOptimizationLLMRouter(adapter=adapter)
    result = router.route(_router_request())
    assert result.transport is TokenOptimizationRouterTransport.NATIVE_TOOLS


def test_structured_output_only_when_tools_false() -> None:
    adapter = _StructuredOutputAdapter(
        decision=_decision(TokenOptimizationRouterConfigurationId.NO_OPTIMIZATION)
    )
    router = TokenOptimizationLLMRouter(adapter=adapter)
    result = router.route(_router_request(content="ok"))
    assert result.transport is TokenOptimizationRouterTransport.STRUCTURED_OUTPUT


def test_unsupported_when_both_false() -> None:
    router = TokenOptimizationLLMRouter(adapter=_UnsupportedAdapter())
    result = router.route(_router_request())
    assert result.status is TokenOptimizationRouterStatus.UNSUPPORTED_ADAPTER


def test_fallback_disabled_blocks_structured_output() -> None:
    adapter = _StructuredOutputAdapter(
        decision=_decision(TokenOptimizationRouterConfigurationId.NO_OPTIMIZATION)
    )
    router = TokenOptimizationLLMRouter(adapter=adapter)
    request = TokenOptimizationLLMRouterRequest(
        request=_router_request().request,
        policy=TokenOptimizationLLMRouterPolicy(allow_structured_output_fallback=False),
        request_id="router-test-fallback-off",
    )
    result = router.route(request)
    assert result.status is TokenOptimizationRouterStatus.UNSUPPORTED_ADAPTER


def test_native_failure_does_not_invoke_structured_fallback() -> None:
    adapter = _NativeToolsAdapter(raise_on_generate=True)
    router = TokenOptimizationLLMRouter(adapter=adapter)
    result = router.route(_router_request())
    assert result.status is TokenOptimizationRouterStatus.LLM_ERROR
    assert adapter.structured_calls == 0


def test_exactly_one_expected_tool_succeeds() -> None:
    adapter = _NativeToolsAdapter(decision=_decision(TokenOptimizationRouterConfigurationId.EXACT_ONLY))
    result = TokenOptimizationLLMRouter(adapter=adapter).route(_router_request())
    assert result.configuration_id is TokenOptimizationRouterConfigurationId.EXACT_ONLY
    assert result.tool_call_id == "call-router-1"


def test_no_tool_call_fails_safely() -> None:
    adapter = _NativeToolsAdapter(tool_calls=())
    result = TokenOptimizationLLMRouter(adapter=adapter).route(_router_request())
    assert result.reason is TokenOptimizationRouterReason.NO_TOOL_CALL


def test_multiple_tool_calls_fail_safely() -> None:
    valid = _decision(TokenOptimizationRouterConfigurationId.EXACT_ONLY).model_dump_json()
    adapter = _NativeToolsAdapter(
        tool_calls=(
            LLMToolCall(id="1", name=ROUTER_TOOL_ID, arguments_json=valid),
            LLMToolCall(id="2", name=ROUTER_TOOL_ID, arguments_json=valid),
        )
    )
    result = TokenOptimizationLLMRouter(adapter=adapter).route(_router_request())
    assert result.reason is TokenOptimizationRouterReason.MULTIPLE_TOOL_CALLS


def test_unexpected_tool_fails_safely() -> None:
    adapter = _NativeToolsAdapter(
        tool_calls=(
            LLMToolCall(id="1", name="other.tool", arguments_json="{}"),
        )
    )
    result = TokenOptimizationLLMRouter(adapter=adapter).route(_router_request())
    assert result.reason is TokenOptimizationRouterReason.UNEXPECTED_TOOL


def test_malformed_json_fails_safely() -> None:
    adapter = _NativeToolsAdapter(
        tool_calls=(
            LLMToolCall(id="1", name=ROUTER_TOOL_ID, arguments_json="{bad"),
        )
    )
    result = TokenOptimizationLLMRouter(adapter=adapter).route(_router_request())
    assert result.reason is TokenOptimizationRouterReason.INVALID_TOOL_ARGUMENTS


def test_invalid_pydantic_arguments_fail_safely() -> None:
    payload = _decision(TokenOptimizationRouterConfigurationId.EXACT_ONLY).model_dump()
    payload["confidence"] = 2.0
    adapter = _NativeToolsAdapter(
        tool_calls=(
            LLMToolCall(
                id="1",
                name=ROUTER_TOOL_ID,
                arguments_json=json.dumps(payload),
            ),
        )
    )
    result = TokenOptimizationLLMRouter(adapter=adapter).route(_router_request())
    assert result.reason is TokenOptimizationRouterReason.INVALID_TOOL_ARGUMENTS


def test_disabled_policy_blocked() -> None:
    adapter = _NativeToolsAdapter(decision=_decision(TokenOptimizationRouterConfigurationId.EXACT_ONLY))
    request = _router_request(
        policy=TokenOptimizationPolicy(enabled=False, profile=TokenOptimizationProfile.OFF),
    )
    result = TokenOptimizationLLMRouter(adapter=adapter).route(request)
    assert result.status is TokenOptimizationRouterStatus.BLOCKED
    assert result.reason is TokenOptimizationRouterReason.POLICY_DISABLED
    assert result.transport is TokenOptimizationRouterTransport.UNSUPPORTED


def test_off_profile_blocked() -> None:
    adapter = _NativeToolsAdapter(decision=_decision(TokenOptimizationRouterConfigurationId.EXACT_ONLY))
    request = _router_request(
        policy=TokenOptimizationPolicy(enabled=True, profile=TokenOptimizationProfile.OFF),
    )
    result = TokenOptimizationLLMRouter(adapter=adapter).route(request)
    assert result.status is TokenOptimizationRouterStatus.BLOCKED
    assert result.reason is TokenOptimizationRouterReason.PROFILE_OFF
    assert result.transport is TokenOptimizationRouterTransport.UNSUPPORTED


def test_lossy_blocked_when_not_allowed() -> None:
    adapter = _NativeToolsAdapter(
        decision=_decision(TokenOptimizationRouterConfigurationId.EXTRACTIVE_ONLY)
    )
    request = _router_request(
        content="INFO\n" * 200,
        source_type=TokenOptimizationSourceType.TOOL_OUTPUT,
        policy=TokenOptimizationPolicy(
            enabled=True,
            profile=TokenOptimizationProfile.CONSERVATIVE,
            allow_lossy=False,
        ),
    )
    result = TokenOptimizationLLMRouter(adapter=adapter).route(request)
    assert result.reason is TokenOptimizationRouterReason.LOSSY_NOT_ALLOWED


def test_missing_packing_input_blocked() -> None:
    adapter = _NativeToolsAdapter(
        decision=_decision(
            TokenOptimizationRouterConfigurationId.PACKING_ONLY,
            reason_code=TokenOptimizationRouterReasonCode.PRIORITY_PACKING,
        )
    )
    result = TokenOptimizationLLMRouter(adapter=adapter).route(_router_request())
    assert result.reason is TokenOptimizationRouterReason.PACKING_INPUT_REQUIRED


def test_protected_lossy_requires_review() -> None:
    adapter = _NativeToolsAdapter(
        decision=_decision(TokenOptimizationRouterConfigurationId.EXTRACTIVE_ONLY)
    )
    protected = ProtectedRegion(kind=ProtectedRegionKind.IDENTIFIER, value="SECRET-SYNTH")
    request = _router_request(
        content="INFO\n" * 200,
        source_type=TokenOptimizationSourceType.TOOL_OUTPUT,
        protected_regions=(protected,),
    )
    result = TokenOptimizationLLMRouter(adapter=adapter).route(request)
    assert result.status is TokenOptimizationRouterStatus.REVIEW_REQUIRED
    assert result.reason is TokenOptimizationRouterReason.PROTECTED_REGIONS_REQUIRE_REVIEW


def test_model_requested_review_prevents_execution() -> None:
    adapter = _NativeToolsAdapter(
        decision=_decision(
            TokenOptimizationRouterConfigurationId.EXACT_ONLY,
            review_required=True,
        )
    )
    result = TokenOptimizationLLMRouter(adapter=adapter).route_and_execute(_router_request())
    assert result.executed is False
    assert result.reason is TokenOptimizationRouterReason.MODEL_REQUESTED_REVIEW


def test_low_confidence_prevents_execution() -> None:
    adapter = _NativeToolsAdapter(
        decision=_decision(
            TokenOptimizationRouterConfigurationId.EXACT_ONLY,
            confidence=0.1,
        )
    )
    result = TokenOptimizationLLMRouter(adapter=adapter).route(_router_request())
    assert result.reason is TokenOptimizationRouterReason.CONFIDENCE_BELOW_THRESHOLD


def test_no_optimization_safe_noop() -> None:
    adapter = _NativeToolsAdapter(
        decision=_decision(
            TokenOptimizationRouterConfigurationId.NO_OPTIMIZATION,
            reason_code=TokenOptimizationRouterReasonCode.CLEAN_NO_OP,
        )
    )
    result = TokenOptimizationLLMRouter(adapter=adapter).route_and_execute(
        _router_request(content="ok")
    )
    assert result.status is TokenOptimizationRouterStatus.NO_OPTIMIZATION
    assert result.executed is False
    assert result.pipeline_config is None


def test_valid_lossless_decision_compiles() -> None:
    adapter = _NativeToolsAdapter(decision=_decision(TokenOptimizationRouterConfigurationId.EXACT_ONLY))
    result = TokenOptimizationLLMRouter(adapter=adapter).route(_router_request())
    assert result.status is TokenOptimizationRouterStatus.ROUTED
    assert result.pipeline_config is not None
    assert result.pipeline_config.layers[0].layer_id == "builtin.exact_deduplication"


def test_route_and_execute_runs_pipeline() -> None:
    adapter = _NativeToolsAdapter(decision=_decision(TokenOptimizationRouterConfigurationId.EXACT_ONLY))
    result = TokenOptimizationLLMRouter(adapter=adapter).route_and_execute(_router_request())
    assert result.executed is True
    assert result.pipeline_result is not None
    assert "builtin.exact_deduplication" in result.pipeline_result.applied_layer_ids


def test_packing_route_and_execute_with_typed_input() -> None:
    packing_case = next(
        case for case in LLM_ROUTER_CORPUS if case.case_id == "router.rag_priority_packing"
    )
    packing_input = packing_case.metadata["packing_input"]
    assert isinstance(packing_input, BudgetAwarePackingInput)
    adapter = _NativeToolsAdapter(
        decision=_decision(
            TokenOptimizationRouterConfigurationId.PACKING_ONLY,
            reason_code=TokenOptimizationRouterReasonCode.PRIORITY_PACKING,
        )
    )
    request = _router_request(
        content=packing_case.content,
        metadata={"packing_input": packing_input},
    )
    result = TokenOptimizationLLMRouter(adapter=adapter).route_and_execute(request)
    assert result.executed is True
    assert "builtin.budget_aware_context_packing" in result.pipeline_result.applied_layer_ids


def test_safe_report_has_no_raw_content() -> None:
    adapter = _NativeToolsAdapter(decision=_decision(TokenOptimizationRouterConfigurationId.EXACT_ONLY))
    result = TokenOptimizationLLMRouter(adapter=adapter).route_and_execute(_router_request())
    safe = token_optimization_router_result_to_safe_dict(result)
    dumped = json.dumps(safe)
    assert "SYNTH-ALPHA" not in dumped
    assert "arguments_json" not in dumped
    assert "traceback" not in dumped


@pytest.mark.parametrize("case", LLM_ROUTER_CORPUS)
def test_corpus_cases_define_required_fields(case: object) -> None:
    from tests.fixtures.token_optimization.llm_router_corpus import LLMRouterCorpusCase

    assert isinstance(case, LLMRouterCorpusCase)
    assert case.case_id.startswith("router.")
    if case.evaluate_suitability:
        assert case.acceptable_configuration_ids
    assert case.synthetic_marker


class _CountingAdapter(LLMAdapter):
    provider = "fake-counting"
    model = "fake-counting"

    def __init__(self) -> None:
        super().__init__()
        self.supports_tools_calls = 0
        self.supports_structured_output_calls = 0
        self.generate_with_tools_calls = 0
        self.generate_structured_calls = 0
        self.generate_messages_calls = 0
        self.messages_built = False

    @property
    def context_window_tokens(self) -> int:
        return 8192

    def supports_tools(self) -> bool:
        self.supports_tools_calls += 1
        return True

    def supports_structured_output(self) -> bool:
        self.supports_structured_output_calls += 1
        return True

    def generate_messages(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        run_id: str | None = None,
    ) -> LLMAdapterResponse:
        self.generate_messages_calls += 1
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
        return build_adapter_response(content="")

    def generate_structured(
        self,
        messages: Sequence[ChatMessage],
        output_model: type,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        run_id: str | None = None,
    ) -> LLMStructuredResult[Any]:
        self.generate_structured_calls += 1
        raise AssertionError("structured output must not be called during preflight")


class _OllamaLikeAdapter(LLMAdapter):
    provider = LLMProvider.OLLAMA
    model = "qwen2.5:7b"

    def __init__(
        self,
        *,
        capabilities: OllamaModelCapabilities,
    ) -> None:
        super().__init__()
        self._model_capabilities = capabilities
        self.structured_calls = 0
        self.generate_with_tools_calls = 0

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
            parsed=_decision(TokenOptimizationRouterConfigurationId.NO_OPTIMIZATION),
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
        self.generate_with_tools_calls += 1
        return build_adapter_response(
            content="",
            tool_calls=(
                LLMToolCall(
                    id="ollama-call",
                    name=ROUTER_TOOL_ID,
                    arguments_json=_decision(
                        TokenOptimizationRouterConfigurationId.NO_OPTIMIZATION
                    ).model_dump_json(),
                ),
            ),
        )


def _wrap_ollama_like_adapter(inner: _OllamaLikeAdapter) -> CatalogCapabilityAdapter:
    wrapped = enrich_adapter_with_catalog_capabilities(
        inner,
        provider=LLMProvider.OLLAMA,
        model="qwen2.5:7b",
    )
    assert isinstance(wrapped, CatalogCapabilityAdapter)
    return wrapped


def _assert_zero_ollama_activity(inner: _OllamaLikeAdapter) -> None:
    assert inner.generate_with_tools_calls == 0
    assert inner.structured_calls == 0


def _assert_zero_adapter_activity(adapter: _CountingAdapter) -> None:
    assert adapter.supports_tools_calls == 0
    assert adapter.supports_structured_output_calls == 0
    assert adapter.generate_with_tools_calls == 0
    assert adapter.generate_structured_calls == 0
    assert adapter.generate_messages_calls == 0


@pytest.mark.parametrize(
    ("policy", "expected_reason"),
    [
        (
            TokenOptimizationPolicy(enabled=False, profile=TokenOptimizationProfile.OFF),
            TokenOptimizationRouterReason.POLICY_DISABLED,
        ),
        (
            TokenOptimizationPolicy(enabled=True, profile=TokenOptimizationProfile.OFF),
            TokenOptimizationRouterReason.PROFILE_OFF,
        ),
    ],
)
def test_policy_preflight_blocks_before_adapter_activity(
    policy: TokenOptimizationPolicy,
    expected_reason: TokenOptimizationRouterReason,
) -> None:
    adapter = _CountingAdapter()
    request = _router_request(policy=policy)
    result = TokenOptimizationLLMRouter(adapter=adapter).route(request)
    assert result.status is TokenOptimizationRouterStatus.BLOCKED
    assert result.reason is expected_reason
    assert result.transport is TokenOptimizationRouterTransport.UNSUPPORTED
    assert result.executed is False
    _assert_zero_adapter_activity(adapter)


def test_unresolved_ollama_capabilities_fail_closed() -> None:
    adapter = _OllamaLikeAdapter(
        capabilities=OllamaModelCapabilities(
            model="legacy:7b",
            capabilities=frozenset(),
            resolved=False,
            source=OllamaCapabilityResolutionSource.UNAVAILABLE,
            error_type="ConnectionError",
        )
    )
    result = TokenOptimizationLLMRouter(adapter=adapter).route(_router_request())
    assert result.status is TokenOptimizationRouterStatus.UNSUPPORTED_ADAPTER
    assert result.reason is TokenOptimizationRouterReason.CAPABILITY_RESOLUTION_FAILED
    assert adapter.structured_calls == 0


def test_unresolved_capabilities_do_not_use_structured_fallback() -> None:
    adapter = _OllamaLikeAdapter(
        capabilities=OllamaModelCapabilities(
            model="legacy:7b",
            capabilities=frozenset(),
            resolved=False,
            source=OllamaCapabilityResolutionSource.UNAVAILABLE,
            error_type="InvalidCapabilities",
        )
    )
    TokenOptimizationLLMRouter(adapter=adapter).route(_router_request())
    assert adapter.structured_calls == 0


def test_resolved_no_tools_model_uses_structured_fallback() -> None:
    adapter = _OllamaLikeAdapter(
        capabilities=OllamaModelCapabilities(
            model="legacy:7b",
            capabilities=frozenset({"completion"}),
            resolved=True,
            source=OllamaCapabilityResolutionSource.EXPLICIT_TEST_OVERRIDE,
        )
    )
    result = TokenOptimizationLLMRouter(adapter=adapter).route(_router_request(content="ok"))
    assert result.transport is TokenOptimizationRouterTransport.STRUCTURED_OUTPUT
    assert adapter.structured_calls == 1


def test_resolved_tools_model_uses_native_transport() -> None:
    adapter = _OllamaLikeAdapter(
        capabilities=OllamaModelCapabilities(
            model="qwen2.5:7b",
            capabilities=frozenset({"tools", "completion"}),
            resolved=True,
            source=OllamaCapabilityResolutionSource.EXPLICIT_TEST_OVERRIDE,
        )
    )
    result = TokenOptimizationLLMRouter(adapter=adapter).route(_router_request(content="ok"))
    assert result.transport is TokenOptimizationRouterTransport.NATIVE_TOOLS
    assert adapter.structured_calls == 0


def test_wrapped_unresolved_ollama_capabilities_fail_closed() -> None:
    inner = _OllamaLikeAdapter(
        capabilities=OllamaModelCapabilities(
            model="qwen2.5:7b",
            capabilities=frozenset(),
            resolved=False,
            source=OllamaCapabilityResolutionSource.UNAVAILABLE,
            error_type="ConnectionError",
        )
    )
    adapter = _wrap_ollama_like_adapter(inner)
    result = TokenOptimizationLLMRouter(adapter=adapter).route(_router_request())
    assert result.status is TokenOptimizationRouterStatus.UNSUPPORTED_ADAPTER
    assert result.reason is TokenOptimizationRouterReason.CAPABILITY_RESOLUTION_FAILED
    assert result.transport is TokenOptimizationRouterTransport.UNSUPPORTED
    assert result.executed is False
    assert result.pipeline_config is None
    assert result.pipeline_result is None
    _assert_zero_ollama_activity(inner)


def test_wrapped_unresolved_capabilities_do_not_use_structured_fallback() -> None:
    inner = _OllamaLikeAdapter(
        capabilities=OllamaModelCapabilities(
            model="qwen2.5:7b",
            capabilities=frozenset(),
            resolved=False,
            source=OllamaCapabilityResolutionSource.UNAVAILABLE,
            error_type="InvalidCapabilities",
        )
    )
    adapter = _wrap_ollama_like_adapter(inner)
    TokenOptimizationLLMRouter(adapter=adapter).route(_router_request())
    _assert_zero_ollama_activity(inner)


def test_wrapped_resolved_no_tools_model_uses_structured_fallback() -> None:
    inner = _OllamaLikeAdapter(
        capabilities=OllamaModelCapabilities(
            model="qwen2.5:7b",
            capabilities=frozenset({"completion"}),
            resolved=True,
            source=OllamaCapabilityResolutionSource.EXPLICIT_TEST_OVERRIDE,
        )
    )
    adapter = _wrap_ollama_like_adapter(inner)
    result = TokenOptimizationLLMRouter(adapter=adapter).route(_router_request(content="ok"))
    assert result.transport is TokenOptimizationRouterTransport.STRUCTURED_OUTPUT
    assert inner.structured_calls == 1
    assert inner.generate_with_tools_calls == 0


def test_wrapped_resolved_tools_model_uses_native_transport() -> None:
    inner = _OllamaLikeAdapter(
        capabilities=OllamaModelCapabilities(
            model="qwen2.5:7b",
            capabilities=frozenset({"tools", "completion"}),
            resolved=True,
            source=OllamaCapabilityResolutionSource.EXPLICIT_TEST_OVERRIDE,
        )
    )
    adapter = _wrap_ollama_like_adapter(inner)
    result = TokenOptimizationLLMRouter(adapter=adapter).route(_router_request(content="ok"))
    assert result.transport is TokenOptimizationRouterTransport.NATIVE_TOOLS
    assert inner.generate_with_tools_calls == 1
    assert inner.structured_calls == 0


@pytest.mark.parametrize(
    ("capabilities_resolved", "native_tools_supported", "structured_output_supported", "expected"),
    [
        (True, True, False, TokenOptimizationRouterTransport.NATIVE_TOOLS.value),
        (True, False, True, TokenOptimizationRouterTransport.STRUCTURED_OUTPUT.value),
        (False, False, True, TokenOptimizationRouterTransport.UNSUPPORTED.value),
        (True, False, False, TokenOptimizationRouterTransport.UNSUPPORTED.value),
    ],
)
def test_summary_transport_selection(
    capabilities_resolved: bool,
    native_tools_supported: bool,
    structured_output_supported: bool,
    expected: str,
) -> None:
    from tests.e2e.token_optimization.test_llm_router_ollama_live import _summary_transport

    assert (
        _summary_transport(
            capabilities_resolved=capabilities_resolved,
            native_tools_supported=native_tools_supported,
            structured_output_supported=structured_output_supported,
        )
        == expected
    )


def test_safe_report_preserves_canonical_executed_order() -> None:
    packing_case = next(
        case for case in LLM_ROUTER_CORPUS if case.case_id == "router.rag_mixed_dedupe_packing"
    )
    packing_input = packing_case.metadata["packing_input"]
    adapter = _NativeToolsAdapter(
        decision=_decision(
            TokenOptimizationRouterConfigurationId.EXACT_THEN_PACKING,
            reason_code=TokenOptimizationRouterReasonCode.MIXED_DEDUPLICATION_PACKING,
        )
    )
    request = _router_request(
        content="unique-line-one\nunique-line-two\n",
        metadata={"packing_input": packing_input},
    )
    result = TokenOptimizationLLMRouter(adapter=adapter).route_and_execute(request)
    assert result.pipeline_result is not None
    canonical_order = list(result.pipeline_result.receipt_metadata["executed_layer_ids"])
    grouped_order = list(result.pipeline_result.applied_layer_ids) + list(
        result.pipeline_result.bypassed_layer_ids
    )
    assert len(canonical_order) == 2
    assert canonical_order != grouped_order
    safe = token_optimization_router_result_to_safe_dict(result)
    assert safe["executed_layer_ids"] == canonical_order
    assert safe["completed"] is result.pipeline_result.receipt_metadata["completed"]


def test_safe_report_includes_required_failure_layer_id() -> None:
    from intergrax.runtime.token_optimization.contracts import TokenOptimizationPipelineResult
    from intergrax.runtime.token_optimization.llm_router_contracts import (
        TokenOptimizationLLMRouterResult,
    )

    pipeline_result = TokenOptimizationPipelineResult(
        pipeline_id="router.exact_only",
        original_content="a",
        final_content="a",
        receipt_metadata={
            "executed_layer_ids": ["builtin.exact_deduplication"],
            "completed": False,
            "required_failure_layer_id": "builtin.exact_deduplication",
            "secret_payload": "must-not-leak",
        },
    )
    router_result = TokenOptimizationLLMRouterResult(
        request_id="safe-report-test",
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
        pipeline_result=pipeline_result,
        executed=True,
    )
    safe = token_optimization_router_result_to_safe_dict(router_result)
    assert safe["required_failure_layer_id"] == "builtin.exact_deduplication"
    assert safe["completed"] is False
    dumped = json.dumps(safe)
    assert "secret_payload" not in dumped
    assert "must-not-leak" not in dumped
