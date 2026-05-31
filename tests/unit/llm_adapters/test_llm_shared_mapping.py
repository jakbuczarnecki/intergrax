# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

import pytest

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters._shared.messages import map_chat_completion_messages
from intergrax.llm_adapters._shared.responses_input import messages_to_responses_input
from intergrax.llm_adapters._shared.tool_schema import openai_tools_to_anthropic

pytestmark = pytest.mark.unit


def test_map_chat_completion_preserves_tool_round_trip() -> None:
    convo = [
        ChatMessage(role="assistant", content="", tool_calls=[{"id": "c1", "type": "function", "function": {"name": "search", "arguments": "{}"}}]),
        ChatMessage(role="tool", content="result", tool_call_id="c1", name="search"),
        ChatMessage(role="user", content="next"),
    ]
    mapped = map_chat_completion_messages(system_text="", convo=convo)
    assert mapped[0]["role"] == "assistant"
    assert mapped[0]["tool_calls"]
    assert mapped[1]["role"] == "tool"
    assert mapped[1]["tool_call_id"] == "c1"


def test_responses_input_maps_function_call_output() -> None:
    mapped = [
        {"role": "tool", "content": "ok", "tool_call_id": "call_1"},
    ]
    items = messages_to_responses_input(mapped)
    assert items[0]["type"] == "function_call_output"
    assert items[0]["call_id"] == "call_1"


def test_openai_tools_to_anthropic() -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search the web",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    out = openai_tools_to_anthropic(tools)
    assert out[0]["name"] == "search"
    assert out[0]["input_schema"]["type"] == "object"
