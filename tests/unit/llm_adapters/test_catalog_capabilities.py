# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from typing import Optional, Sequence
from unittest.mock import MagicMock

import pytest

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.registry.catalog_capabilities import (
    CatalogCapabilityAdapter,
    enrich_adapter_with_catalog_capabilities,
)
from intergrax.llm_adapters.registry.model_catalog import ModelRecord

pytestmark = pytest.mark.unit


class _StubAdapter(LLMAdapter):
    provider = LLMProvider.OPENAI
    model = "gpt-4o-mini"

    def __init__(self) -> None:
        super().__init__()

    @property
    def context_window_tokens(self) -> int:
        return 8192

    def supports_vision(self) -> bool:
        return False

    def generate_messages(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        run_id: Optional[str] = None,
    ) -> LLMAdapterResponse:
        del temperature, max_tokens, run_id
        return LLMAdapterResponse(content="ok")


def test_catalog_capability_adapter_overlays_vision_flag() -> None:
    inner = _StubAdapter()
    record = ModelRecord(
        model_id="gpt-4o-mini",
        context_window_tokens=128_000,
        supports_vision=True,
    )
    wrapped = CatalogCapabilityAdapter(inner, record)
    assert wrapped.supports_vision() is True


def test_enrich_adapter_with_catalog_capabilities_for_known_model() -> None:
    inner = _StubAdapter()
    inner.model = "gemini-2.0-flash"
    enriched = enrich_adapter_with_catalog_capabilities(
        inner,
        provider=LLMProvider.GEMINI,
        model="gemini-2.0-flash",
    )
    assert isinstance(enriched, CatalogCapabilityAdapter)
    assert enriched.supports_vision() is True


def test_enrich_adapter_unknown_model_passthrough() -> None:
    inner = _StubAdapter()
    enriched = enrich_adapter_with_catalog_capabilities(
        inner,
        provider=LLMProvider.OPENAI,
        model="unknown-model-id-xyz",
    )
    assert enriched is inner
