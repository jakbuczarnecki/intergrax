# © Artur Czarnecki. All rights reserved.

"""CE-02-R1G: mandatory preservation during compiler hard-budget enforcement."""

from __future__ import annotations

import pytest

from intergrax.context.budget.contracts import ContextBudgetUnsatisfiableError
from intergrax.context.budget.mandatory_base_messages import mandatory_base_message_indices
from intergrax.llm.messages import ChatMessage
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.context.context_compiler import ContextCompiler
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.base.base_llm_adapter import BaseLLMAdapter

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _count_chars(text: str) -> int:
    return len(text)


def _mandatory_snapshot(messages: list[ChatMessage]) -> list[ChatMessage]:
    indices = mandatory_base_message_indices(messages)
    return [messages[index] for index in sorted(indices)]


def _assert_mandatory_byte_for_byte(
    before: list[ChatMessage],
    after: list[ChatMessage],
) -> None:
    before_mandatory = _mandatory_snapshot(before)
    after_mandatory = _mandatory_snapshot(after)
    assert len(before_mandatory) == len(after_mandatory)
    for original, preserved in zip(before_mandatory, after_mandatory, strict=True):
        assert preserved.role == original.role
        assert preserved.content == original.content
        assert preserved.entry_id == original.entry_id
        assert preserved.tool_calls == original.tool_calls
        assert preserved.tool_call_id == original.tool_call_id


class _Adapter(BaseLLMAdapter):
    provider = "fake"
    model = "fake-r1g"

    def __init__(self, window: int = 4096) -> None:
        super().__init__()
        self._window = window

    @property
    def context_window_tokens(self) -> int:
        return self._window

    def generate_messages(self, messages, **kwargs) -> LLMAdapterResponse:
        _ = messages, kwargs
        return LLMAdapterResponse(content="ok")


def test_hard_budget_preserves_multiple_systems_and_drops_optional_history() -> None:
    sys1 = "a" * 20
    sys2 = "b" * 20
    optional_history = "c" * 20
    final_user = "d" * 10
    messages = [
        ChatMessage(role="system", content=sys1, entry_id="sys-1"),
        ChatMessage(role="system", content=sys2, entry_id="sys-2"),
        ChatMessage(role="assistant", content=optional_history, entry_id="hist"),
        ChatMessage(role="user", content=final_user, entry_id="user-final"),
    ]
    compiler = ContextCompiler(count_tokens=_count_chars)
    result = compiler._enforce_hard_budget(messages, budget_tokens=55)
    _assert_mandatory_byte_for_byte(messages, result)
    assert sum(_count_chars(m.content or "") for m in result) <= 55
    assert optional_history not in {m.content for m in result}


def test_hard_budget_fail_closed_when_mandatory_exceeds_budget() -> None:
    messages = [
        ChatMessage(role="system", content="a" * 20),
        ChatMessage(role="system", content="b" * 20),
        ChatMessage(role="assistant", content="x" * 5),
        ChatMessage(role="user", content="c" * 10),
    ]
    compiler = ContextCompiler(count_tokens=_count_chars)
    with pytest.raises(ContextBudgetUnsatisfiableError, match="mandatory"):
        compiler._enforce_hard_budget(messages, budget_tokens=49)


def test_hard_budget_preserves_incomplete_tool_group() -> None:
    messages = [
        ChatMessage(role="system", content="s" * 5, entry_id="sys"),
        ChatMessage(
            role="assistant",
            content="invoke",
            entry_id="asst-incomplete",
            tool_calls=[{"id": "call-1", "type": "function", "function": {"name": "lookup"}}],
        ),
        ChatMessage(role="assistant", content="o" * 40, entry_id="optional-hist"),
        ChatMessage(role="user", content="u" * 5, entry_id="final-user"),
    ]
    compiler = ContextCompiler(count_tokens=_count_chars)
    result = compiler._enforce_hard_budget(messages, budget_tokens=30)
    _assert_mandatory_byte_for_byte(messages, result)
    assert sum(_count_chars(m.content or "") for m in result) <= 30
    assert "o" * 40 not in {m.content for m in result}


def test_hard_budget_preserves_standalone_tool_message() -> None:
    messages = [
        ChatMessage(role="system", content="s" * 5, entry_id="sys"),
        ChatMessage(
            role="tool",
            content="tool-body" * 3,
            entry_id="tool-1",
            tool_call_id="call-orphan",
        ),
        ChatMessage(role="assistant", content="h" * 30, entry_id="hist"),
        ChatMessage(role="user", content="q" * 5, entry_id="final"),
    ]
    compiler = ContextCompiler(count_tokens=_count_chars)
    result = compiler._enforce_hard_budget(messages, budget_tokens=40)
    _assert_mandatory_byte_for_byte(messages, result)
    assert any(m.role == "tool" and m.tool_call_id == "call-orphan" for m in result)


def test_compile_hard_budget_path_preserves_dual_system_instructions() -> None:
    adapter = _Adapter(window=4096)
    config = RuntimeConfig(llm_adapter=adapter, production_mode=False)
    messages = [
        ChatMessage(role="system", content="primary" * 4, entry_id="sys-a"),
        ChatMessage(role="system", content="secondary" * 4, entry_id="sys-b"),
        ChatMessage(role="assistant", content="history" * 200, entry_id="hist"),
        ChatMessage(role="user", content="task", entry_id="user-final"),
    ]
    compiler = ContextCompiler(count_tokens=_count_chars)
    result = compiler.compile(messages, config, max_output_tokens=64, input_budget_tokens=80)
    _assert_mandatory_byte_for_byte(messages, result.messages)
    assert result.total_tokens <= 80


def test_canonical_engine_compile_does_not_trim_mandatory_system_pair() -> None:
    from intergrax.runtime.nexus.context.context_engine import DefaultNexusContextEngine

    adapter = _Adapter(window=4096)
    config = RuntimeConfig(llm_adapter=adapter, production_mode=False)
    engine = DefaultNexusContextEngine()
    messages = [
        ChatMessage(role="system", content="alpha" * 5, entry_id="sys-1"),
        ChatMessage(role="system", content="beta" * 5, entry_id="sys-2"),
        ChatMessage(role="assistant", content="noise" * 50, entry_id="hist"),
        ChatMessage(role="user", content="ask", entry_id="user-final"),
    ]
    compiled = engine._compiler.compile(
        list(messages),
        config,
        max_output_tokens=64,
        input_budget_tokens=70,
    )
    _assert_mandatory_byte_for_byte(messages, compiled.messages)
