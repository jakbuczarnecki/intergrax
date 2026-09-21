# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""
Unit tests for BaseLLMAdapter framework base and LLMAdapter execution contract.
"""

from __future__ import annotations

from typing import Optional, Sequence

import pytest

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters._shared.adapter_response_builders import build_adapter_response
from intergrax.llm_adapters.base.base_llm_adapter import BaseLLMAdapter
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter


pytestmark = pytest.mark.unit


def test_base_llm_adapter_is_abstract() -> None:
    with pytest.raises(TypeError):
        BaseLLMAdapter()  # type: ignore[abstract]


class _MinimalValidAdapter(BaseLLMAdapter):
    provider = "unit-test"
    model = "unit-test-model"

    @property
    def context_window_tokens(self) -> int:
        return 1000

    def generate_messages(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        run_id: Optional[str] = None,
    ) -> LLMAdapterResponse:
        return build_adapter_response(content="")


def test_framework_adapter_satisfies_execution_contract() -> None:
    adapter = _MinimalValidAdapter()
    assert isinstance(adapter, LLMAdapter)


def test_adapter_with_empty_provider_is_rejected() -> None:
    class EmptyProviderAdapter(_MinimalValidAdapter):
        provider = ""

    adapter = EmptyProviderAdapter()
    with pytest.raises(ValueError):
        adapter.validate()


def test_adapter_with_none_provider_is_rejected() -> None:
    class NoneProviderAdapter(_MinimalValidAdapter):
        provider = None  # type: ignore[assignment]

    adapter = NoneProviderAdapter()
    with pytest.raises(ValueError):
        adapter.validate()
