# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

import pytest

from intergrax.llm.messages import ChatMessage
from intergrax.runtime.nexus.runtime_steps.rag_step import RagStep
from testing_support.builder import build_runtime_state_for_tests

pytestmark = pytest.mark.unit


@dataclass
class _BuiltContext:
    rag_used: bool
    rag_reason: Optional[str]
    retrieved_chunks: Optional[List[Any]]


@dataclass
class _RagPromptBundle:
    context_messages: List[ChatMessage]


class _FakeContextBuilder:
    def __init__(self, result: _BuiltContext) -> None:
        self._result = result
        self.called = False
        self.last_args: Optional[dict] = None

    async def build_context(self, *, session: Any, request: Any, base_history: Any) -> _BuiltContext:
        self.called = True
        self.last_args = {"session": session, "request": request, "base_history": base_history}
        return self._result


class _FakeRagPromptBuilder:
    def __init__(self, context_messages: List[ChatMessage]) -> None:
        self._context_messages = context_messages
        self.called = False
        self.last_built: Optional[_BuiltContext] = None

    def build_rag_prompt(self, built: _BuiltContext) -> _RagPromptBundle:
        self.called = True
        self.last_built = built
        return _RagPromptBundle(context_messages=self._context_messages)


@pytest.mark.asyncio
async def test_rag_step_noop_when_disabled() -> None:
    state = build_runtime_state_for_tests(run_id="run-1")

    state.messages_for_llm = [
        ChatMessage(role="system", content="sys"),
        ChatMessage(role="user", content="hello"),
    ]
    state.tools_context_parts = ["PREV"]
    state.used_rag = True

    state.context.config.enable_rag = False
    state.context.context_builder = None
    state.context.rag_prompt_builder = None

    before_msgs = list(state.messages_for_llm)
    before_tools_ctx = list(state.tools_context_parts)

    await RagStep().run(state)

    assert state.used_rag is False
    assert state.messages_for_llm == before_msgs
    assert state.tools_context_parts == before_tools_ctx


@pytest.mark.asyncio
async def test_rag_step_raises_when_enabled_but_no_context_builder() -> None:
    state = build_runtime_state_for_tests(run_id="run-1")

    state.context.config.enable_rag = True
    state.context.context_builder = None

    with pytest.raises(RuntimeError, match=r"RAG enabled but ContextBuilder is not configured\."):
        await RagStep().run(state)


@pytest.mark.asyncio
async def test_rag_step_fallback_requires_session() -> None:
    state = build_runtime_state_for_tests(run_id="run-1")

    state.context.config.enable_rag = True
    state.context_builder_result = None

    state.context.context_builder = _FakeContextBuilder(
        _BuiltContext(rag_used=False, rag_reason="no_hits", retrieved_chunks=[])
    )
    state.context.rag_prompt_builder = _FakeRagPromptBuilder(context_messages=[])

    # session is None by default in test builder
    assert state.session is None

    with pytest.raises(AssertionError, match=r"Session must be set before RAG step\."):
        await RagStep().run(state)


@pytest.mark.asyncio
async def test_rag_step_fallback_requires_session() -> None:
    state = build_runtime_state_for_tests(run_id="run-1")

    state.context.config.enable_rag = True
    state.context_builder_result = None

    state.context.context_builder = _FakeContextBuilder(
        _BuiltContext(rag_used=False, rag_reason="no_hits", retrieved_chunks=[])
    )
    state.context.rag_prompt_builder = _FakeRagPromptBuilder(context_messages=[])

    # session is None by default in test builder
    assert state.session is None

    with pytest.raises(AssertionError, match=r"Session must be set before RAG step\."):
        await RagStep().run(state)



@pytest.mark.asyncio
async def test_rag_step_enabled_but_not_used_when_no_retrieved_context() -> None:
    state = build_runtime_state_for_tests(run_id="run-1")

    state.messages_for_llm = [
        ChatMessage(role="system", content="sys"),
        ChatMessage(role="user", content="hello"),
    ]
    state.tools_context_parts = []

    state.context.config.enable_rag = True

    built = _BuiltContext(
        rag_used=False,
        rag_reason="no_hits",
        retrieved_chunks=[],
    )

    # IMPORTANT: step must NOT call rag_prompt_builder when rag_used is False
    state.context.context_builder = _FakeContextBuilder(built)
    fake_prompt_builder = _FakeRagPromptBuilder(
        context_messages=[ChatMessage(role="system", content="RAG SHOULD NOT APPEAR")]
    )
    state.context.rag_prompt_builder = fake_prompt_builder

    state.context_builder_result = built

    before_msgs = list(state.messages_for_llm)

    await RagStep().run(state)

    assert state.used_rag is False
    assert state.messages_for_llm == before_msgs
    assert state.tools_context_parts == []
    assert fake_prompt_builder.called is False


@pytest.mark.asyncio
async def test_rag_step_raises_when_used_but_no_rag_prompt_builder() -> None:
    state = build_runtime_state_for_tests(run_id="run-1")

    state.messages_for_llm = [
        ChatMessage(role="system", content="sys"),
        ChatMessage(role="user", content="hello"),
    ]
    state.tools_context_parts = []

    state.context.config.enable_rag = True

    built = _BuiltContext(
        rag_used=True,
        rag_reason=None,
        retrieved_chunks=[{"text": "x", "metadata": {"source": "doc"}}],
    )

    state.context_builder_result = built
    state.context.context_builder = _FakeContextBuilder(built)
    state.context.rag_prompt_builder = None

    with pytest.raises(RuntimeError, match=r"RAG enabled but rag_prompt_builder is not configured\."):
        await RagStep().run(state)


@pytest.mark.asyncio
async def test_rag_step_injects_context_before_last_user_and_appends_tools_context() -> None:
    state = build_runtime_state_for_tests(run_id="run-1")

    state.messages_for_llm = [
        ChatMessage(role="system", content="sys"),
        ChatMessage(role="assistant", content="a1"),
        ChatMessage(role="user", content="u1"),
    ]
    state.tools_context_parts = []

    state.context.config.enable_rag = True

    chunks = [
        {"text": "chunk one", "metadata": {"source": "docA", "page": 1}},
        {"text": "chunk two", "metadata": {"source": "docB", "page": 2}},
    ]

    built = _BuiltContext(
        rag_used=True,
        rag_reason=None,
        retrieved_chunks=chunks,
    )
    state.context_builder_result = built

    rag_ctx_messages = [
        ChatMessage(role="system", content="RAG_CTX_1"),
        ChatMessage(role="system", content="RAG_CTX_2"),
    ]

    state.context.context_builder = _FakeContextBuilder(built)
    state.context.rag_prompt_builder = _FakeRagPromptBuilder(context_messages=rag_ctx_messages)

    await RagStep().run(state)

    assert state.used_rag is True

    # Injected before last user
    assert [m.content for m in state.messages_for_llm] == [
        "sys",
        "a1",
        "RAG_CTX_1",
        "RAG_CTX_2",
        "u1",
    ]

    # Tools context: prefix + formatted chunks (via format_rag_context)
    assert len(state.tools_context_parts) == 1
    assert state.tools_context_parts[0].startswith("RAG CONTEXT:\n")
    assert "chunk one" in state.tools_context_parts[0]
    assert "docA" in state.tools_context_parts[0]


@pytest.mark.asyncio
async def test_rag_step_inserts_context_at_end_when_no_user_message() -> None:
    state = build_runtime_state_for_tests(run_id="run-1")

    state.messages_for_llm = [
        ChatMessage(role="system", content="sys"),
        ChatMessage(role="assistant", content="a1"),
    ]
    state.tools_context_parts = []

    state.context.config.enable_rag = True

    built = _BuiltContext(
        rag_used=True,
        rag_reason=None,
        retrieved_chunks=[{"text": "chunk", "metadata": {"source": "doc"}}],
    )
    state.context_builder_result = built

    rag_ctx_messages = [
        ChatMessage(role="system", content="RAG_CTX"),
    ]

    state.context.context_builder = _FakeContextBuilder(built)
    state.context.rag_prompt_builder = _FakeRagPromptBuilder(context_messages=rag_ctx_messages)

    await RagStep().run(state)

    assert [m.content for m in state.messages_for_llm] == [
        "sys",
        "a1",
        "RAG_CTX",
    ]
