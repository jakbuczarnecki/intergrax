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
        },
        clear=False,
    ):
        adapter = BedrockChatAdapter(
            client=client,
            model_id="anthropic.claude-3-haiku-20240307-v1:0",
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
    assert events[-1].is_final
    assert events[-1].response is not None
    assert events[-1].response.finish_reason.value == "completed"
    client.converse_stream.assert_called_once()
