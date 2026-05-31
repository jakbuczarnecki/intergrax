# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from unittest.mock import MagicMock, patch

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.providers.aws_bedrock_adapter import BedrockChatAdapter, BedrockModelFamily


def test_bedrock_stream_with_tools_converse_stream() -> None:
    client = MagicMock()
    client.converse_stream.return_value = {
        "stream": [
            {"contentBlockDelta": {"delta": {"text": "hi"}}},
            {"messageStop": {}},
        ],
    }

    with patch.dict(
        "os.environ",
        {
            "INTERGRAX_DEFAULT_AWS_REGION": "us-east-1",
            "INTERGRAX_DEFAULT_BEDROCK_MODEL_ID": "anthropic.claude-3-haiku-20240307-v1:0",
        },
        clear=False,
    ):
        adapter = BedrockChatAdapter(
            client=client,
            family=BedrockModelFamily.ANTHROPIC,
            use_converse=True,
        )

    events = list(
        adapter.stream_with_tools(
            [ChatMessage(role="user", content="ping")],
            [{"type": "function", "function": {"name": "noop", "parameters": {}}}],
            run_id="bt1",
        )
    )
    assert len(events) >= 2
    assert events[-1].get("finish_reason") == "completed"
    client.converse_stream.assert_called_once()
