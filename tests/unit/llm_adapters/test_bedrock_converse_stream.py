# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters._shared.bedrock_converse import iter_converse_stream_text
from intergrax.llm_adapters.providers.aws_bedrock_adapter import BedrockChatAdapter, BedrockModelFamily


def test_iter_converse_stream_text_yields_deltas() -> None:
    events = [
        {"contentBlockDelta": {"delta": {"text": "Hel"}}},
        {"contentBlockDelta": {"delta": {"text": "lo"}}},
        {"messageStop": {}},
    ]
    assert list(iter_converse_stream_text(events)) == ["Hel", "lo"]


def test_bedrock_stream_messages_uses_converse_stream() -> None:
    client = MagicMock()
    client.converse_stream.return_value = {
        "stream": [
            {"contentBlockDelta": {"delta": {"text": "a"}}},
            {"contentBlockDelta": {"delta": {"text": "b"}}},
        ],
    }

    with patch.dict(
        "os.environ",
        {
            "INTERGRAX_DEFAULT_AWS_REGION": "us-east-1",
            "INTERGRAX_DEFAULT_BEDROCK_MODEL_ID": "anthropic.claude-3-haiku-20240307-v1:0",
            "INTERGRAX_BEDROCK_USE_CONVERSE": "true",
        },
        clear=False,
    ):
        adapter = BedrockChatAdapter(
            client=client,
            family=BedrockModelFamily.ANTHROPIC,
            use_converse=True,
        )

    chunks = list(
        adapter.stream_messages([ChatMessage(role="user", content="hi")], run_id="s1")
    )
    assert chunks == ["a", "b"]
    client.converse_stream.assert_called_once()
