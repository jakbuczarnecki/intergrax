# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

"""Conformance helpers for LLM adapter contract tests."""

from __future__ import annotations

from collections.abc import Iterable

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.base.base_llm_adapter import BaseLLMAdapter
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.capabilities import LLMFeature, read_feature_flag
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.contracts.stream_event import LLMStreamEvent


def capability_flag_violations(adapter: object, *, delegating: bool = False) -> list[str]:
    """Return mismatches between declared ``supports_*`` flags and implemented methods.

    A *leaf* adapter (default) must declare streaming exactly when it overrides
    ``stream_messages`` and tool streaming exactly when it overrides ``stream_with_tools``.
    A *delegating* wrapper (catalog overlay, failover, routing) forwards both methods, so its
    flags follow the wrapped adapter(s); there a flag may be ``False`` while the method exists,
    but never ``True`` without the method. In both cases tool streaming implies tools and
    streaming, and a declared tool-capable adapter must implement ``generate_with_tools``.
    """
    violations: list[str] = []
    name = type(adapter).__name__

    def overrides(method: str) -> bool:
        implemented = getattr(type(adapter), method, None)
        return callable(implemented) and implemented is not getattr(BaseLLMAdapter, method)

    streaming = read_feature_flag(adapter, LLMFeature.STREAMING)
    tool_streaming = read_feature_flag(adapter, LLMFeature.TOOL_STREAMING)
    tools = read_feature_flag(adapter, LLMFeature.TOOLS)
    impl_streaming = overrides("stream_messages")
    impl_tool_streaming = overrides("stream_with_tools")

    if streaming and not impl_streaming:
        violations.append(f"{name}: supports_streaming() is True but stream_messages is not implemented")
    if tool_streaming and not impl_tool_streaming:
        violations.append(f"{name}: supports_tool_streaming() is True but stream_with_tools is not implemented")
    if tools and not overrides("generate_with_tools"):
        violations.append(f"{name}: supports_tools() is True but generate_with_tools is not implemented")
    if tool_streaming and not (tools and streaming):
        violations.append(f"{name}: supports_tool_streaming() requires supports_tools() and supports_streaming()")
    if not delegating:
        if impl_streaming and not streaming:
            violations.append(f"{name}: stream_messages is implemented but supports_streaming() is False")
        if impl_tool_streaming and not tool_streaming:
            violations.append(f"{name}: stream_with_tools is implemented but supports_tool_streaming() is False")
    return violations


def assert_capability_flags_match_methods(adapter: object, *, delegating: bool = False) -> None:
    violations = capability_flag_violations(adapter, delegating=delegating)
    assert not violations, "; ".join(violations)


def assert_supports_streaming(adapter: LLMAdapter) -> None:
    assert adapter.supports_streaming() is True
    msgs = [ChatMessage(role="user", content="ping")]
    stream = adapter.stream_messages(msgs, max_tokens=16, run_id="conformance")
    assert isinstance(stream, Iterable)
    chunks = list(stream)
    assert isinstance(chunks, list)
    assert all(isinstance(event, LLMStreamEvent) for event in chunks)


def assert_supports_tools_contract(adapter: LLMAdapter) -> None:
    assert adapter.supports_tools() is True
    out = adapter.generate_with_tools(
        [ChatMessage(role="user", content="hi")],
        [{"type": "function", "function": {"name": "noop", "parameters": {"type": "object"}}}],
        run_id="conformance-tools",
    )
    assert isinstance(out, LLMAdapterResponse)
    assert isinstance(out.content, str)
    assert isinstance(out.tool_calls, tuple)
    assert out.finish_reason is not None


def assert_stream_with_tools_contract(adapter: LLMAdapter) -> None:
    """Native tools streaming must yield at least one partial or final chunk."""
    if not adapter.supports_tools():
        return
    stream = adapter.stream_with_tools(
        [ChatMessage(role="user", content="hi")],
        [{"type": "function", "function": {"name": "noop", "parameters": {"type": "object"}}}],
        max_tokens=32,
        run_id="conformance-tools-stream",
    )
    chunks = list(stream)
    assert chunks, "stream_with_tools produced no events"
    assert all(isinstance(event, LLMStreamEvent) for event in chunks)
    assert any(event.is_final for event in chunks)


def assert_generate_messages_returns_response(adapter: LLMAdapter, *, user_text: str = "Say OK") -> LLMAdapterResponse:
    result = adapter.generate_messages(
        [ChatMessage(role="user", content=user_text)],
        max_tokens=32,
        run_id="conformance-chat",
    )
    assert isinstance(result, LLMAdapterResponse)
    assert isinstance(result.content, str)
    return result


def assert_generate_messages_returns_text(adapter: LLMAdapter, *, user_text: str = "Say OK") -> str:
    """Deprecated alias — prefer ``assert_generate_messages_returns_response``."""
    return assert_generate_messages_returns_response(adapter, user_text=user_text).content


def assert_usage_tracking(adapter: LLMAdapter, *, run_id: str = "conformance-usage") -> None:
    adapter.generate_messages(
        [ChatMessage(role="user", content="count")],
        max_tokens=8,
        run_id=run_id,
    )
    stats = adapter.usage.get_run_stats(run_id)
    assert stats.calls >= 1


def run_adapter_conformance(
    adapter: LLMAdapter,
    *,
    check_tools_stream: bool = False,
) -> None:
    """
    Full mocked/live-safe conformance suite for a single adapter instance.

    Call from unit tests after constructing an adapter with a mocked SDK client.
    """
    assert_generate_messages_returns_response(adapter)
    if adapter.supports_streaming():
        assert_supports_streaming(adapter)
    if adapter.supports_tools():
        assert_supports_tools_contract(adapter)
        if check_tools_stream:
            assert_stream_with_tools_contract(adapter)
    assert_usage_tracking(adapter)
