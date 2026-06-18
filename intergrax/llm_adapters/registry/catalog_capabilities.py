# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Catalog-driven LLM capability flags (M-LLM-X.1.7 · AUDIT-IDEAL-6.3)."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, Dict, List, Optional, Union

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.contracts.structured_result import LLMStructuredResult
from intergrax.llm_adapters.contracts.stream_event import LLMStreamEvent
from intergrax.llm_adapters.registry.model_catalog import ModelRecord, lookup_model_record
from intergrax.utils import attribute_access


class CatalogCapabilityAdapter(LLMAdapter):
    """Overlay ModelCatalog capability flags on a concrete adapter."""

    def __init__(self, inner: LLMAdapter, record: ModelRecord) -> None:
        super().__init__()
        self._inner = inner
        self._record = record
        self.provider = inner.provider
        self.model = attribute_access.optional_str(inner, "model")
        self.model_name_for_token_estimation = attribute_access.optional(
            inner, "model_name_for_token_estimation", None
        )
        inner_call_config = attribute_access.optional(inner, "call_config", None)
        if inner_call_config is not None:
            self.call_config = inner_call_config
        inner_usage = attribute_access.optional(inner, "usage", None)
        if inner_usage is not None:
            self.usage = inner_usage
        inner_id = attribute_access.optional(inner, "id", None)
        if inner_id:
            self.id = inner_id

    @property
    def context_window_tokens(self) -> int:
        return self._inner.context_window_tokens

    def supports_vision(self) -> bool:
        return self._record.supports_vision or self._inner.supports_vision()

    def supports_tools(self) -> bool:
        return self._record.supports_tools and self._inner.supports_tools()

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
        tools_schema: List[Dict[str, Any]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        run_id: Optional[str] = None,
    ) -> LLMAdapterResponse:
        return self._inner.generate_with_tools(
            messages,
            tools_schema,
            temperature=temperature,
            max_tokens=max_tokens,
            tool_choice=tool_choice,
            run_id=run_id,
        )

    def generate_structured(
        self,
        messages: Sequence[ChatMessage],
        schema: Dict[str, Any],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        run_id: Optional[str] = None,
    ) -> LLMStructuredResult:
        return self._inner.generate_structured(
            messages,
            schema,
            temperature=temperature,
            max_tokens=max_tokens,
            run_id=run_id,
        )

    def count_messages_tokens(self, messages: Sequence[ChatMessage]) -> int:
        return self._inner.count_messages_tokens(messages)

    def validate(self) -> None:
        self._inner.validate()


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
    model_id = (model or attribute_access.optional(adapter, "model", None) or "").strip()
    if not model_id:
        return adapter
    record = lookup_model_record(model_id)
    if record is None:
        return adapter
    return CatalogCapabilityAdapter(adapter, record)
