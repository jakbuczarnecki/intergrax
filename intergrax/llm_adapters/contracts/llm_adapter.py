# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""Canonical LLM execution contract (EBH-2E-R1)."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Optional, Protocol, Union, runtime_checkable

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.contracts.stream_event import LLMStreamEvent
from intergrax.llm_adapters.contracts.structured_result import LLMStructuredResult
from intergrax.llm_adapters.contracts.strict_tool_arguments import CanonicalFunctionToolDefinition


@runtime_checkable
class LLMAdapter(Protocol):
    """
    Minimal cross-layer LLM execution port.

    Optional capabilities (streaming, tools, structured output, modalities) are
    expressed as methods with default-not-supported semantics on framework base
    implementations; consumers should probe capability flags before dispatch.
    """

    provider: LLMProvider | str
    model: str

    @property
    def context_window_tokens(self) -> int:
        ...

    def generate_messages(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        run_id: Optional[str] = None,
    ) -> LLMAdapterResponse:
        ...

    def supports_streaming(self) -> bool:
        ...

    def stream_messages(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        run_id: Optional[str] = None,
    ) -> Iterable[LLMStreamEvent]:
        ...

    def supports_tools(self) -> bool:
        ...

    def supports_strict_tool_argument_conformance(self) -> bool:
        ...

    def generate_with_tools(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[CanonicalFunctionToolDefinition | Mapping[str, Any]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tool_choice: Optional[Union[str, dict[str, Any]]] = None,
        run_id: Optional[str] = None,
    ) -> LLMAdapterResponse:
        ...

    def stream_with_tools(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[CanonicalFunctionToolDefinition | Mapping[str, Any]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tool_choice: Optional[Union[str, dict[str, Any]]] = None,
        run_id: Optional[str] = None,
    ) -> Iterable[LLMStreamEvent]:
        ...

    def supports_structured_output(self) -> bool:
        ...

    def generate_structured(
        self,
        messages: Sequence[ChatMessage],
        output_model: type,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        run_id: Optional[str] = None,
    ) -> LLMStructuredResult[Any]:
        ...

    def supports_vision(self) -> bool:
        ...

    def supports_audio_input(self) -> bool:
        ...

    def supports_audio_output(self) -> bool:
        ...
