# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from typing import Optional, Sequence
from unittest.mock import MagicMock

import pytest

from intergrax.llm.messages import ChatMessage
from intergrax.runtime.nexus.context.context_preflight import count_message_tokens

pytestmark = pytest.mark.unit


def test_count_message_tokens_delegates_to_adapter() -> None:
    adapter = MagicMock()
    adapter.count_messages_tokens.return_value = 42
    messages = [ChatMessage(role="user", content="hello")]
    assert count_message_tokens(messages, adapter=adapter) == 42
    adapter.count_messages_tokens.assert_called_once_with(messages)


def test_count_message_tokens_uses_custom_counter_when_provided() -> None:
    messages = [ChatMessage(role="user", content="abcd")]
    assert count_message_tokens(messages, count_tokens=lambda text: len(text)) == 4
