# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from intergrax.agents_packages.legal_agent.legal_memory_policy import (
    LegalMemoryPolicy,
    NO_PRIOR_TURNS_PLACEHOLDER,
    build_legal_conversation_snippet,
)
from intergrax.llm.messages import ChatMessage


def _state_with_messages(
    *,
    messages_for_llm: list[ChatMessage] | None = None,
    built_history_messages: list[ChatMessage] | None = None,
) -> MagicMock:
    s = MagicMock()
    s.messages_for_llm = messages_for_llm or []
    s.built_history_messages = built_history_messages or []
    return s


def test_build_snippet_empty_messages() -> None:
    policy = LegalMemoryPolicy()
    out = build_legal_conversation_snippet(_state_with_messages(), policy=policy)
    assert out == NO_PRIOR_TURNS_PLACEHOLDER


def test_build_snippet_prefers_messages_for_llm() -> None:
    policy = LegalMemoryPolicy()
    state = _state_with_messages(
        messages_for_llm=[ChatMessage(role="user", content="a")],
        built_history_messages=[ChatMessage(role="user", content="b")],
    )
    assert build_legal_conversation_snippet(state, policy=policy) == "user: a"


def test_build_snippet_falls_back_to_built_history() -> None:
    policy = LegalMemoryPolicy()
    state = _state_with_messages(
        messages_for_llm=[],
        built_history_messages=[ChatMessage(role="assistant", content="x")],
    )
    assert build_legal_conversation_snippet(state, policy=policy) == "assistant: x"


def test_build_snippet_tail_and_char_cap() -> None:
    long = "y" * 100
    state = _state_with_messages(
        messages_for_llm=[
            ChatMessage(role="user", content="drop"),
            ChatMessage(role="assistant", content="keep-a"),
            ChatMessage(role="user", content=long),
        ],
    )
    policy = LegalMemoryPolicy(
        conversation_tail_message_limit=2,
        conversation_snippet_max_chars_per_message=8,
    )
    out = build_legal_conversation_snippet(state, policy=policy)
    assert "drop" not in out
    assert out == "assistant: keep-a\nuser: " + "y" * 8


def test_legal_memory_policy_field_bounds() -> None:
    with pytest.raises(ValidationError):
        LegalMemoryPolicy(conversation_tail_message_limit=0)
    with pytest.raises(ValidationError):
        LegalMemoryPolicy(conversation_snippet_max_chars_per_message=0)
