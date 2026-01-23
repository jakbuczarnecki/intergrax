# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""
Unit tests for LLMAdapter.

These tests define the minimal contract that every concrete adapter must satisfy:
- LLMAdapter is abstract and cannot be instantiated directly,
- provider must be a non-empty string.

Why this matters:
Violations of this contract cause late runtime failures that are hard to debug.
This enforces correctness at adapter definition time.
"""

from __future__ import annotations
from typing import Sequence

import pytest
from requests_cache import Optional

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.llm_adapter import LLMAdapter


pytestmark = pytest.mark.unit


def test_llm_adapter_is_abstract() -> None:
    """
    LLMAdapter must be abstract and not instantiable directly.
    """
    with pytest.raises(TypeError):
        LLMAdapter()  # type: ignore[abstract]


class _MinimalValidAdapter(LLMAdapter):
    """
    Minimal concrete adapter used for contract tests.
    """

    provider = "unit-test"

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
    ) -> str:
        return ""


def test_adapter_with_empty_provider_is_rejected() -> None:
    """
    Adapters with an empty provider must fail early.
    """

    class EmptyProviderAdapter(_MinimalValidAdapter):
        provider = ""

    adapter = EmptyProviderAdapter()
    with pytest.raises(ValueError):
        adapter.validate()


def test_adapter_with_none_provider_is_rejected() -> None:
    """
    Adapters with provider=None must fail early.
    """

    class NoneProviderAdapter(_MinimalValidAdapter):
        provider = None  # type: ignore[assignment]

    adapter = NoneProviderAdapter()

    with pytest.raises(ValueError):
        adapter.validate()
