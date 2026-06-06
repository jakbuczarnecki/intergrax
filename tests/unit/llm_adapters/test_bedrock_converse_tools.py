# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters._shared.tool_schema import openai_tools_to_bedrock_converse
from intergrax.llm_adapters.providers.aws_bedrock_adapter import BedrockChatAdapter, BedrockModelFamily


def test_openai_tools_to_bedrock_converse_shape() -> None:
    tools = openai_tools_to_bedrock_converse(
        [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
    )
    assert tools[0]["toolSpec"]["name"] == "search"
    assert "inputSchema" in tools[0]["toolSpec"]


def test_bedrock_generate_with_tools_via_converse() -> None:
    client = MagicMock()
    client.converse.return_value = {
        "output": {
            "message": {
                "content": [
                    {"text": "done"},
                    {
                        "toolUse": {
                            "toolUseId": "t1",
                            "name": "search",
                            "input": {"q": "x"},
                        }
                    },
                ]
            }
        }
    }

    adapter = BedrockChatAdapter(
        client=client,
        model_id="anthropic.claude-3-5-sonnet-20241022-v2:0",
        region="us-east-1",
        family=BedrockModelFamily.ANTHROPIC,
        use_converse=True,
    )
    out = adapter.generate_with_tools(
        [ChatMessage(role="user", content="find")],
        [{"type": "function", "function": {"name": "search", "parameters": {}}}],
        run_id="b1",
    )
    assert out.content == "done"
    assert len(out.tool_calls) == 1
    assert out.tool_calls[0].name == "search"
    client.converse.assert_called_once()
