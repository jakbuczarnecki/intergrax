# © Artur Czarnecki. All rights reserved.

"""Shared structural LLMAdapter fake for EBH-2E replaceability architecture gates."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Optional

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters._shared.adapter_response_builders import build_adapter_response
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.contracts.native_tool_choice import NativeToolChoice
from intergrax.llm_adapters.contracts.stream_event import LLMStreamEvent
from intergrax.llm_adapters.contracts.strict_tool_arguments import CanonicalFunctionToolDefinition
from intergrax.llm_adapters.contracts.structured_result import LLMStructuredResult, TStructured


class ExternalStructuralAdapter:
    """Third-party structural LLMAdapter — no framework base class inheritance."""

    provider = "external-structural"
    model = "ext-model"

    @property
    def context_window_tokens(self) -> int:
        return 8192

    def generate_messages(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        run_id: Optional[str] = None,
    ) -> LLMAdapterResponse:
        _ = messages, temperature, max_tokens, run_id
        return build_adapter_response(content="external")

    def supports_streaming(self) -> bool:
        return False

    def stream_messages(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        run_id: Optional[str] = None,
    ) -> Iterable[LLMStreamEvent]:
        _ = messages, temperature, max_tokens, run_id
        raise NotImplementedError

    def supports_tools(self) -> bool:
        return False

    def supports_strict_tool_argument_conformance(self) -> bool:
        return False

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
        _ = messages, tools, temperature, max_tokens, tool_choice, run_id
        raise NotImplementedError

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
        _ = messages, tools, temperature, max_tokens, tool_choice, run_id
        raise NotImplementedError

    def supports_structured_output(self) -> bool:
        return False

    def generate_structured(
        self,
        messages: Sequence[ChatMessage],
        output_model: type[TStructured],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        run_id: Optional[str] = None,
    ) -> LLMStructuredResult[TStructured]:
        _ = messages, output_model, temperature, max_tokens, run_id
        raise NotImplementedError

    def supports_vision(self) -> bool:
        return False

    def supports_audio_input(self) -> bool:
        return False

    def supports_audio_output(self) -> bool:
        return False


def assert_external_structural_llm_adapter(adapter: ExternalStructuralAdapter) -> None:
    assert isinstance(adapter, LLMAdapter)


__all__ = ["ExternalStructuralAdapter", "assert_external_structural_llm_adapter"]
