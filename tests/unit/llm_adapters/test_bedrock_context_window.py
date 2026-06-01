# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from intergrax.llm_adapters.providers.aws_bedrock_adapter import BedrockChatAdapter

pytestmark = pytest.mark.gate


def test_bedrock_context_window_known_model() -> None:
    with patch.dict("os.environ", {"INTERGRAX_DEFAULT_AWS_REGION": "us-east-1"}):
        adapter = BedrockChatAdapter(
            client=MagicMock(),
            model_id="anthropic.claude-3-5-sonnet-20241022-v2:0",
        )
    assert adapter.context_window_tokens == 200_000
