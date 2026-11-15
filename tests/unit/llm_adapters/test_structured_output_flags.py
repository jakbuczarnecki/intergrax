# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

# The two adapters import their vendor SDKs at module level: skip, instead of a collection error,
# when the optional extras are not installed (``llm-anthropic`` / ``llm-openai``).
pytest.importorskip("anthropic", reason="requires the llm-anthropic extra")
pytest.importorskip("openai", reason="requires the llm-openai extra")

from intergrax.llm_adapters.providers.claude_adapter import ClaudeChatAdapter
from intergrax.llm_adapters.providers.openai_responses_adapter import OpenAIChatResponsesAdapter

pytestmark = pytest.mark.gate


def test_openai_declares_structured_output() -> None:
    adapter = OpenAIChatResponsesAdapter(client=MagicMock(), model="gpt-4o-mini")
    assert adapter.supports_structured_output() is True


def test_claude_declares_structured_output() -> None:
    adapter = ClaudeChatAdapter(client=MagicMock(), model="claude-3-5-sonnet-20241022")
    assert adapter.supports_structured_output() is True
