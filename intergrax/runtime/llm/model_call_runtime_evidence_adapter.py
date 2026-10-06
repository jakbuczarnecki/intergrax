# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""LLMAdapter wrapper binding model-input hash before outbound provider calls (TRACE-X-P4)."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Optional

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.contracts.native_tool_choice import NativeToolChoice
from intergrax.llm_adapters.contracts.stream_event import LLMStreamEvent
from intergrax.llm_adapters.contracts.strict_tool_arguments import CanonicalFunctionToolDefinition
from intergrax.llm_adapters.contracts.structured_result import LLMStructuredResult, TStructured
from intergrax.runtime.llm.model_call_attribution import model_call_attribution_scope


class ModelCallRuntimeEvidenceAdapter:
    """Delegates to inner ``LLMAdapter`` while scoping canonical model-input fingerprints."""

    __slots__ = ("_inner",)

    def __init__(self, inner: LLMAdapter) -> None:
        self._inner = inner

    @property
    def provider(self):
        return self._inner.provider

    @property
    def model(self) -> str:
        return self._inner.model

    @property
    def context_window_tokens(self) -> int:
        return self._inner.context_window_tokens

    @property
    def inner_adapter(self) -> LLMAdapter:
        return self._inner

    def supports_streaming(self) -> bool:
        return self._inner.supports_streaming()

    def supports_tools(self) -> bool:
        return self._inner.supports_tools()

    def supports_strict_tool_argument_conformance(self) -> bool:
        return self._inner.supports_strict_tool_argument_conformance()

    def supports_structured_output(self) -> bool:
        return self._inner.supports_structured_output()

    def generate_messages(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        run_id: Optional[str] = None,
    ) -> LLMAdapterResponse:
        with model_call_attribution_scope(messages=messages):
            return self._inner.generate_messages(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                run_id=run_id,
            )

    def stream_messages(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        run_id: Optional[str] = None,
    ) -> Iterable[LLMStreamEvent]:
        with model_call_attribution_scope(messages=messages):
            yield from self._inner.stream_messages(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                run_id=run_id,
            )

    def generate_with_tools(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[CanonicalFunctionToolDefinition],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tool_choice: NativeToolChoice | None = None,
        run_id: Optional[str] = None,
    ) -> LLMAdapterResponse:
        with model_call_attribution_scope(messages=messages):
            return self._inner.generate_with_tools(
                messages,
                tools,
                temperature=temperature,
                max_tokens=max_tokens,
                tool_choice=tool_choice,
                run_id=run_id,
            )

    def stream_with_tools(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[CanonicalFunctionToolDefinition],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tool_choice: NativeToolChoice | None = None,
        run_id: Optional[str] = None,
    ) -> Iterable[LLMStreamEvent]:
        with model_call_attribution_scope(messages=messages):
            yield from self._inner.stream_with_tools(
                messages,
                tools,
                temperature=temperature,
                max_tokens=max_tokens,
                tool_choice=tool_choice,
                run_id=run_id,
            )

    def generate_structured(
        self,
        messages: Sequence[ChatMessage],
        output_model: type[TStructured],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        run_id: Optional[str] = None,
    ) -> LLMStructuredResult[TStructured]:
        with model_call_attribution_scope(messages=messages):
            return self._inner.generate_structured(
                messages,
                output_model,
                temperature=temperature,
                max_tokens=max_tokens,
                run_id=run_id,
            )


def wrap_model_call_runtime_evidence(adapter: LLMAdapter) -> LLMAdapter:
    if isinstance(adapter, ModelCallRuntimeEvidenceAdapter):
        return adapter
    return ModelCallRuntimeEvidenceAdapter(adapter)


__all__ = ["ModelCallRuntimeEvidenceAdapter", "wrap_model_call_runtime_evidence"]
