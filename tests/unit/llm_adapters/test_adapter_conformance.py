# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence
from unittest.mock import MagicMock, patch

import pytest

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters._shared.conformance import (
    assert_generate_messages_returns_text,
    assert_supports_streaming,
    assert_supports_tools_contract,
    assert_usage_tracking,
)
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.providers.claude_adapter import ClaudeChatAdapter
from intergrax.llm_adapters.providers.openai_responses_adapter import OpenAIChatResponsesAdapter

pytestmark = pytest.mark.unit


class _StubAdapter(LLMAdapter):
    provider = "stub"
    model = "stub"

    def __init__(self) -> None:
        super().__init__()

    @property
    def context_window_tokens(self) -> int:
        return 4096

    def generate_messages(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        run_id: Optional[str] = None,
    ) -> str:
        call = self.usage.begin_call(run_id=run_id)
        self.usage.end_call(call, input_tokens=1, output_tokens=1, success=True)
        return "ok"

    def stream_messages(self, messages, **kwargs):
        yield "a"
        yield "b"

    def supports_streaming(self) -> bool:
        return True

    def supports_tools(self) -> bool:
        return True

    def generate_with_tools(self, messages, tools_schema, **kwargs) -> Dict[str, Any]:
        return {"content": "x", "tool_calls": [], "finish_reason": "completed"}


def test_stub_adapter_conformance_suite() -> None:
    adapter = _StubAdapter()
    assert_generate_messages_returns_text(adapter)
    assert_supports_streaming(adapter)
    assert_supports_tools_contract(adapter)
    assert_usage_tracking(adapter)


def test_openai_conformance_with_mock() -> None:
    client = MagicMock()
    usage = MagicMock(input_tokens=3, output_tokens=2)
    response = MagicMock()
    response.usage = usage
    response.output_text = "hello"
    response.output = []
    client.responses.create.return_value = response

    adapter = OpenAIChatResponsesAdapter(client=client, model="gpt-4o-mini")
    assert_generate_messages_returns_text(adapter)
    assert_supports_tools_contract(adapter)


def test_claude_conformance_with_mock() -> None:
    client = MagicMock()
    block = MagicMock(type="text", text="hi")
    response = MagicMock(content=[block], usage=MagicMock(input_tokens=1, output_tokens=1))
    client.messages.create.return_value = response

    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}, clear=False):
        adapter = ClaudeChatAdapter(client=client, model="claude-3-5-sonnet-latest")
    assert_generate_messages_returns_text(adapter)
