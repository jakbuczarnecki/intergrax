# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

"""Catalog-driven LLM capability flags (M-LLM-X.1.7 · AUDIT-IDEAL-6.3)."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Optional

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.base.base_llm_adapter import BaseLLMAdapter
from intergrax.llm_adapters.contracts.strict_tool_arguments import CanonicalFunctionToolDefinition
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.contracts.structured_result import LLMStructuredResult, TStructured
from intergrax.llm_adapters.contracts.stream_event import LLMStreamEvent
from intergrax.llm_adapters.contracts.native_tool_choice import NativeToolChoice
from intergrax.llm_adapters.registry.model_catalog import ModelRecord, lookup_model_record


class CatalogCapabilityAdapter(BaseLLMAdapter):
    """Overlay ModelCatalog capability flags on a concrete adapter."""

    def __init__(self, inner: LLMAdapter, record: ModelRecord) -> None:
        super().__init__()
        self._inner = inner
        self._record = record
        self.provider = inner.provider
        self.model = inner.model
        self.model_name_for_token_estimation = inner.model or None

    @property
    def context_window_tokens(self) -> int:
        return self._inner.context_window_tokens

    def supports_vision(self) -> bool:
        return self._record.supports_vision or self._inner.supports_vision()

    def supports_tools(self) -> bool:
        return self._record.supports_tools and self._inner.supports_tools()

    def supports_strict_tool_argument_conformance(self) -> bool:
        return self._inner.supports_strict_tool_argument_conformance()

    def supports_structured_output(self) -> bool:
        return self._record.supports_structured_output or self._inner.supports_structured_output()

    def supports_streaming(self) -> bool:
        return self._inner.supports_streaming()

    def supports_audio_input(self) -> bool:
        return self._inner.supports_audio_input()

    def supports_audio_output(self) -> bool:
        return self._inner.supports_audio_output()

    def generate_messages(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        run_id: Optional[str] = None,
    ) -> LLMAdapterResponse:
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
        return self._inner.stream_messages(
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
        return self._inner.generate_with_tools(
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
        return self._inner.generate_structured(
            messages,
            output_model,
            temperature=temperature,
            max_tokens=max_tokens,
            run_id=run_id,
        )


def unwrap_catalog_capability_adapter(adapter: LLMAdapter) -> LLMAdapter:
    """Return the concrete adapter when catalog enrichment wrapped it."""
    if isinstance(adapter, CatalogCapabilityAdapter):
        return adapter._inner
    return adapter


def enrich_adapter_with_catalog_capabilities(
    adapter: LLMAdapter,
    *,
    provider: str | LLMProvider,
    model: str | None,
) -> LLMAdapter:
    """Return adapter wrapped with catalog capability flags when model is known."""
    model_id = (model or adapter.model or "").strip()
    if not model_id:
        return adapter
    record = lookup_model_record(model_id)
    if record is None:
        return adapter
    return CatalogCapabilityAdapter(adapter, record)
