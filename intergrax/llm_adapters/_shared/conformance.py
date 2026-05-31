# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

"""Conformance helpers for LLM adapter contract tests."""

from __future__ import annotations

from typing import Iterable, Sequence

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter


def assert_supports_streaming(adapter: LLMAdapter) -> None:
    assert adapter.supports_streaming() is True
    msgs = [ChatMessage(role="user", content="ping")]
    stream = adapter.stream_messages(msgs, max_tokens=16, run_id="conformance")
    assert isinstance(stream, Iterable)
    chunks = list(stream)
    assert isinstance(chunks, list)


def assert_supports_tools_contract(adapter: LLMAdapter) -> None:
    assert adapter.supports_tools() is True
    out = adapter.generate_with_tools(
        [ChatMessage(role="user", content="hi")],
        [{"type": "function", "function": {"name": "noop", "parameters": {"type": "object"}}}],
        run_id="conformance-tools",
    )
    assert "content" in out
    assert "tool_calls" in out
    assert "finish_reason" in out


def assert_generate_messages_returns_text(adapter: LLMAdapter, *, user_text: str = "Say OK") -> str:
    text = adapter.generate_messages(
        [ChatMessage(role="user", content=user_text)],
        max_tokens=32,
        run_id="conformance-chat",
    )
    assert isinstance(text, str)
    return text


def assert_usage_tracking(adapter: LLMAdapter, *, run_id: str = "conformance-usage") -> None:
    adapter.generate_messages(
        [ChatMessage(role="user", content="count")],
        max_tokens=8,
        run_id=run_id,
    )
    stats = adapter.usage.get_run_stats(run_id)
    assert stats.calls >= 1
