# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import pytest
from intergrax.llm.messages import ChatMessage
from intergrax.runtime.nexus.runtime_steps.core_llm_step import CoreLLMStep
from testing_support.builder import build_runtime_state_for_tests

pytestmark = pytest.mark.unit


class _FakeAdapter:
    def __init__(self, result="OK"):
        self.result = result
        self.called = False
        self.kwargs = None

    def generate_messages(self, messages, run_id=None, **kwargs):
        self.called = True
        self.kwargs = kwargs
        return self.result


class _FailingAdapter:
    def generate_messages(self, *args, **kwargs):
        raise RuntimeError("boom")


@pytest.mark.asyncio
async def test_core_llm_step_uses_tools_answer_when_present():
    state = build_runtime_state_for_tests(run_id="run-1")
    state.used_tools = True
    state.tools_agent_answer = "TOOLS"

    await CoreLLMStep().run(state)

    assert state.raw_answer == "TOOLS"


@pytest.mark.asyncio
async def test_core_llm_step_invalid_last_message_produces_error_answer():
    state = build_runtime_state_for_tests(run_id="run-1")

    state.used_tools = False
    state.tools_agent_answer = None

    state.messages_for_llm = [ChatMessage(role="assistant", content="a1")]
    state.context.config.llm_adapter = _FakeAdapter()

    await CoreLLMStep().run(state)

    assert state.raw_answer.startswith("[ERROR] LLM adapter failed:")
    assert "messages_for_llm must end with a 'user' message" in state.raw_answer




@pytest.mark.asyncio
async def test_core_llm_step_calls_adapter_and_sets_answer():
    state = build_runtime_state_for_tests(run_id="run-1")
    state.messages_for_llm = [ChatMessage(role="user", content="u1")]

    adapter = _FakeAdapter(result="LLM_OK")
    state.context.config.llm_adapter = adapter

    await CoreLLMStep().run(state)

    assert adapter.called is True
    assert state.raw_answer == "LLM_OK"


@pytest.mark.asyncio
async def test_core_llm_step_passes_max_tokens():
    state = build_runtime_state_for_tests(run_id="run-1")
    state.messages_for_llm = [ChatMessage(role="user", content="u1")]
    state.request.max_output_tokens = 123

    adapter = _FakeAdapter(result="OK")
    state.context.config.llm_adapter = adapter

    await CoreLLMStep().run(state)

    assert adapter.kwargs["max_tokens"] == 123


@pytest.mark.asyncio
async def test_core_llm_step_fallback_on_error_with_tools_answer():
    state = build_runtime_state_for_tests(run_id="run-1")
    state.messages_for_llm = [ChatMessage(role="user", content="u1")]
    state.context.config.llm_adapter = _FailingAdapter()

    state.tools_agent_answer = "TOOLS"

    await CoreLLMStep().run(state)

    assert "TOOLS" in state.raw_answer
    assert "[ERROR]" in state.raw_answer


@pytest.mark.asyncio
async def test_core_llm_step_error_without_tools_answer():
    state = build_runtime_state_for_tests(run_id="run-1")
    state.messages_for_llm = [ChatMessage(role="user", content="u1")]
    state.context.config.llm_adapter = _FailingAdapter()

    await CoreLLMStep().run(state)

    assert state.raw_answer.startswith("[ERROR]")
