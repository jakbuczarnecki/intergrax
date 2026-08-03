# © Artur Czarnecki. All rights reserved.

"""CTX-UCL-6A: legacy HistoryLayer fail-closed gate and OFF raw-history load."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from intergrax.llm.messages import ChatMessage
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.context.engine_history_layer import (
    LEGACY_HISTORY_COMPRESSION_DISABLED_REASON,
    HistoryLayer,
    LegacyHistoryCompressionDisabledError,
)
from intergrax.runtime.nexus.responses.response_schema import (
    HistoryCompressionStrategy,
    RuntimeRequest,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_HISTORY_LAYER_SOURCE = (
    Path(__file__).resolve().parents[5]
    / "intergrax"
    / "runtime"
    / "nexus"
    / "context"
    / "engine_history_layer.py"
)


@dataclass
class _StubSession:
    tenant_id: str
    id: str


@dataclass
class _StubRuntimeState:
    session: _StubSession
    request: RuntimeRequest
    run_id: str
    base_history: list[ChatMessage] | None = None
    history_token_count: int | None = None
    trace_calls: int = 0

    def trace_event(self, **_kwargs) -> None:
        self.trace_calls += 1


@dataclass
class _RecordingSessionManager:
    history: list[ChatMessage]
    calls: list[tuple[str, str]] = field(default_factory=list)

    async def get_history(
        self,
        *,
        tenant_id: str,
        session_id: str,
    ) -> list[ChatMessage]:
        self.calls.append((tenant_id, session_id))
        return list(self.history)


class _FailOnGenerateAdapter:
    generate_calls: int = 0

    def count_messages_tokens(self, messages: list[ChatMessage]) -> int:
        return sum(len(msg.content or "") for msg in messages)

    def generate_messages(self, *_args, **_kwargs):
        type(self).generate_calls += 1
        raise AssertionError("generate_messages must not be called")


@dataclass
class _RecordingPromptBuilder:
    calls: int = 0

    def build_history_summary_prompt(self, **_kwargs):
        self.calls += 1
        raise AssertionError("history summary prompt builder must not be called")

    def build_history_summary_user_prompt(self, **_kwargs):
        self.calls += 1
        raise AssertionError("history summary prompt builder must not be called")


def _raw_history() -> list[ChatMessage]:
    return [
        ChatMessage(
            role="system",
            content="system instructions",
            entry_id="entry-system",
            metadata={"kind": "system"},
        ),
        ChatMessage(
            role="user",
            content="user question",
            entry_id="entry-user",
        ),
        ChatMessage(
            role="assistant",
            content="calling tool",
            entry_id="entry-assistant",
            tool_calls=[
                {
                    "id": "tool-call-1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{}"},
                }
            ],
        ),
        ChatMessage(
            role="tool",
            content="tool output",
            entry_id="entry-tool",
            tool_call_id="tool-call-1",
        ),
        ChatMessage(
            role="user",
            content="follow-up",
            entry_id="entry-user-2",
            metadata={"source": "test"},
        ),
    ]


def _history_layer(
    *,
    history: list[ChatMessage],
    strategy: HistoryCompressionStrategy = HistoryCompressionStrategy.OFF,
) -> tuple[HistoryLayer, _RecordingSessionManager, _RecordingPromptBuilder, _StubRuntimeState]:
    session_manager = _RecordingSessionManager(history=history)
    prompt_builder = _RecordingPromptBuilder()
    adapter = _FailOnGenerateAdapter()
    config = RuntimeConfig(llm_adapter=adapter, production_mode=False)
    layer = HistoryLayer(
        config=config,
        session_manager=session_manager,
        history_prompt_builder=prompt_builder,
    )
    state = _StubRuntimeState(
        session=_StubSession(tenant_id="tenant-1", id="session-1"),
        request=RuntimeRequest(
            tenant_id="tenant-1",
            agent_id="agent-1",
            user_id="user-1",
            session_id="session-1",
            message="probe",
            history_compression_strategy=strategy,
        ),
        run_id="run-1",
        base_history=None,
        history_token_count=None,
    )
    return layer, session_manager, prompt_builder, state


@pytest.mark.asyncio
async def test_off_loads_full_history_without_transformation() -> None:
    raw_history = _raw_history()
    layer, session_manager, prompt_builder, state = _history_layer(history=raw_history)

    await layer.build_base_history(state)

    assert len(session_manager.calls) == 1
    assert session_manager.calls[0] == ("tenant-1", "session-1")
    assert _FailOnGenerateAdapter.generate_calls == 0
    assert prompt_builder.calls == 0
    assert state.trace_calls == 1

    assert state.base_history is not None
    assert len(state.base_history) == len(raw_history)
    for index, message in enumerate(raw_history):
        assert state.base_history[index] is message

    entry_ids = [msg.entry_id for msg in state.base_history]
    assert entry_ids == [msg.entry_id for msg in raw_history]

    assistant = state.base_history[2]
    assert assistant.tool_calls == raw_history[2].tool_calls

    tool_message = state.base_history[3]
    assert tool_message.tool_call_id == raw_history[3].tool_call_id

    assert state.base_history[0].metadata == raw_history[0].metadata
    assert state.base_history[4].metadata == raw_history[4].metadata
    assert state.history_token_count is not None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "strategy",
    [
        HistoryCompressionStrategy.TRUNCATE_OLDEST,
        HistoryCompressionStrategy.SUMMARIZE_OLDEST,
        HistoryCompressionStrategy.HYBRID,
    ],
)
async def test_legacy_reduction_strategies_fail_before_side_effects(
    strategy: HistoryCompressionStrategy,
) -> None:
    raw_history = _raw_history()
    layer, session_manager, prompt_builder, state = _history_layer(
        history=raw_history,
        strategy=strategy,
    )
    initial_base_history = state.base_history
    initial_token_count = state.history_token_count

    with pytest.raises(LegacyHistoryCompressionDisabledError) as exc_info:
        await layer.build_base_history(state)

    assert str(exc_info.value) == LEGACY_HISTORY_COMPRESSION_DISABLED_REASON
    assert exc_info.value.reason == LEGACY_HISTORY_COMPRESSION_DISABLED_REASON
    assert session_manager.calls == []
    assert _FailOnGenerateAdapter.generate_calls == 0
    assert prompt_builder.calls == 0
    assert state.trace_calls == 0
    assert state.base_history is initial_base_history
    assert state.history_token_count is initial_token_count


@pytest.mark.asyncio
async def test_history_strategy_requires_exact_enum() -> None:
    raw_history = _raw_history()
    layer, session_manager, prompt_builder, state = _history_layer(history=raw_history)
    state.request.history_compression_strategy = "off"  # type: ignore[assignment]

    with pytest.raises(TypeError, match="history_compression_strategy must be HistoryCompressionStrategy"):
        await layer.build_base_history(state)

    assert session_manager.calls == []
    assert _FailOnGenerateAdapter.generate_calls == 0
    assert prompt_builder.calls == 0
    assert state.trace_calls == 0


def test_history_layer_has_no_independent_optimization_path() -> None:
    source = _HISTORY_LAYER_SOURCE.read_text(encoding="utf-8")
    banned_fragments = [
        ".generate_messages(",
        "_summarize_history_chunk",
        "_compress_history",
        "_truncate_history_by_tokens",
        "context_window_tokens",
        "history_budget_tokens",
        "reserved_for_output",
        "reserved_for_meta",
    ]
    for fragment in banned_fragments:
        assert fragment not in source

    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr != "generate_messages"


def test_runtime_context_constructor_contract_remains_compatible() -> None:
    prompt_builder = _RecordingPromptBuilder()
    adapter = _FailOnGenerateAdapter()
    config = RuntimeConfig(llm_adapter=adapter, production_mode=False)
    session_manager = _RecordingSessionManager(history=[])

    layer = HistoryLayer(
        config=config,
        session_manager=session_manager,
        history_prompt_builder=prompt_builder,
    )

    assert layer is not None
    assert prompt_builder.calls == 0
    assert _FailOnGenerateAdapter.generate_calls == 0
