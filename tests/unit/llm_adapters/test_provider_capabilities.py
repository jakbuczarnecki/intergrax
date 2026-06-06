# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional, Sequence
from unittest.mock import MagicMock, patch

import pytest

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.llm_provider_registry import LLMAdapterRegistry
from intergrax.llm_adapters.providers.openai_responses_adapter import OpenAIChatResponsesAdapter

pytestmark = pytest.mark.unit


@pytest.fixture()
def _restore_registry_state():
    snapshot = dict(LLMAdapterRegistry._factories)
    try:
        yield snapshot
    finally:
        LLMAdapterRegistry._factories = snapshot


def test_openai_supports_tools_and_streaming() -> None:
    adapter = OpenAIChatResponsesAdapter(client=MagicMock(), model="gpt-4o-mini")
    assert adapter.supports_tools() is True
    assert adapter.supports_streaming() is True
    assert adapter.supports_structured_output() is True


def test_lazy_registry_loads_openai(_restore_registry_state: Dict[str, Any]) -> None:
    LLMAdapterRegistry._factories.clear()
    with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}, clear=False):
        adapter = LLMAdapterRegistry.create(LLMProvider.OPENAI, client=MagicMock(), model="gpt-4o-mini")
    assert isinstance(adapter, OpenAIChatResponsesAdapter)


def test_openai_generate_with_tools_mocked() -> None:
    client = MagicMock()
    usage = MagicMock(input_tokens=10, output_tokens=5)
    response = MagicMock()
    response.usage = usage
    response.status = "completed"
    response.output_text = "hi"
    response.output = []
    client.responses.create.return_value = response

    adapter = OpenAIChatResponsesAdapter(client=client, model="gpt-4o-mini")
    out = adapter.generate_with_tools(
        [ChatMessage(role="user", content="hello")],
        [{"type": "function", "function": {"name": "t", "parameters": {}}}],
        run_id="r1",
    )
    assert out.content == "hi"
    assert out.tool_calls == ()
