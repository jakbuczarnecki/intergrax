# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import pytest
from intergrax.llm.messages import ChatMessage
from intergrax.runtime.nexus.runtime_steps.persist_and_build_answer_step import PersistAndBuildAnswerStep
from intergrax.runtime.nexus.session.session_storage import SessionStorage
from testing_support.builder import build_runtime_state_for_tests

pytestmark = pytest.mark.unit


class _FakeAppendResult:
    consolidation_diag = None


class _FakeSessionManager(SessionStorage):
    def __init__(self):
        self.append_called = False
        self.last_message = None

    async def append_message(
        self,
        *,
        tenant_id: str,
        session_id: str,
        message: ChatMessage,
    ):
        self.append_called = True
        self.last_message = message
        return _FakeAppendResult()


@pytest.mark.asyncio
async def test_persist_step_requires_session():
    state = build_runtime_state_for_tests(run_id="run-1")
    state.session = None

    with pytest.raises(AssertionError, match=r"Session must be set before persistence\."):
        await PersistAndBuildAnswerStep().run(state)


@pytest.mark.asyncio
async def test_persist_step_uses_raw_answer():
    state = build_runtime_state_for_tests(run_id="run-1")
    state.session = type("S", (), {"id": "s1", "tenant_id": "t1"})()
    state.context.session_manager = _FakeSessionManager()

    state.raw_answer = "hello"
    await PersistAndBuildAnswerStep().run(state)

    assert state.context.session_manager.append_called is True
    assert state.runtime_answer.answer == "hello"


@pytest.mark.asyncio
async def test_persist_step_fallback_to_tools_agent_answer():
    state = build_runtime_state_for_tests(run_id="run-1")
    state.session = type("S", (), {"id": "s1", "tenant_id": "t1"})()
    state.context.session_manager = _FakeSessionManager()

    state.raw_answer = ""
    state.tools_agent_answer = "tools result"

    await PersistAndBuildAnswerStep().run(state)

    assert state.runtime_answer.answer == "tools result"


@pytest.mark.asyncio
async def test_persist_step_strategy_selection():
    state = build_runtime_state_for_tests(run_id="run-1")
    state.session = type("S", (), {"id": "s1", "tenant_id": "t1"})()
    state.context.session_manager = _FakeSessionManager()

    state.raw_answer = "ok"
    state.used_rag = True
    state.used_websearch = True
    state.used_tools = True
    state.context.config.enable_rag = True
    state.context.config.enable_websearch = True
    state.context.config.tools_mode = "auto"

    await PersistAndBuildAnswerStep().run(state)

    assert state.runtime_answer.route.strategy == "llm_with_rag_websearch_and_tools"


@pytest.mark.asyncio
async def test_persist_step_tool_calls_mapping():
    state = build_runtime_state_for_tests(run_id="run-1")
    state.session = type("S", (), {"id": "s1", "tenant_id": "t1"})()
    state.context.session_manager = _FakeSessionManager()

    state.raw_answer = "ok"

    class _T:
        tool_name = "t"
        arguments = {"a": 1}
        output_preview = "res"
        success = True
        error_message = None
        raw_trace = {"x": 1}

    state.tool_traces = [_T()]

    await PersistAndBuildAnswerStep().run(state)

    tc = state.runtime_answer.tool_calls[0]
    assert tc.tool_name == "t"
    assert tc.success is True


@pytest.mark.asyncio
async def test_persist_step_used_attachments_in_route():
    state = build_runtime_state_for_tests(run_id="run-1")
    state.session = type("S", (), {"id": "s1", "tenant_id": "test-tenant"})()
    state.context.session_manager = _FakeSessionManager()

    state.raw_answer = "ok"
    state.used_attachments_context = True
    state.attachments_chunks_count = 3

    await PersistAndBuildAnswerStep().run(state)

    extra = state.runtime_answer.route.extra
    assert extra["used_attachments_context"] is True
    assert extra["attachments_chunks"] == 3
