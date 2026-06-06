# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

import pytest
from typing import Optional, Sequence

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.runtime.nexus.runtime_steps.core_llm_step import CoreLLMStep
from intergrax.runtime.nexus.tracing.adapters.core_llm_call_recorded import CoreLLMCallRecordedDiagV1
from testing_support.builder import FakeLLMAdapter, build_runtime_state_for_tests

pytestmark = pytest.mark.unit


class _FailingLLMAdapter(LLMAdapter):
    provider = LLMProvider.OPENAI
    model = "failing-stub"

    @property
    def context_window_tokens(self) -> int:
        return 128_000

    def generate_messages(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        run_id: Optional[str] = None,
    ) -> LLMAdapterResponse:
        raise RuntimeError("boom")


class _TrackingFakeLLMAdapter(FakeLLMAdapter):
    def __init__(self, *, fixed_text: str = "OK") -> None:
        super().__init__(fixed_text=fixed_text)
        self.called = False
        self.kwargs: dict[str, object] | None = None

    def generate_messages(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        run_id: Optional[str] = None,
        **kwargs: object,
    ) -> LLMAdapterResponse:
        self.called = True
        self.kwargs = {"max_tokens": max_tokens, **kwargs}
        return super().generate_messages(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            run_id=run_id,
        )


@pytest.mark.asyncio
async def test_core_llm_step_uses_tools_answer_when_present():
    state = build_runtime_state_for_tests(run_id="run-1")
    state.used_tools = True
    state.tool_planner_answer = "TOOLS"
    await CoreLLMStep().run(state)
    assert state.raw_answer == "TOOLS"


@pytest.mark.asyncio
async def test_core_llm_step_invalid_last_message_produces_error_answer():
    state = build_runtime_state_for_tests(run_id="run-1")
    state.used_tools = False
    state.tool_planner_answer = None
    state.messages_for_llm = [ChatMessage(role="assistant", content="a1")]
    state.context.config.llm_adapter = FakeLLMAdapter()
    await CoreLLMStep().run(state)
    assert state.raw_answer.startswith("[ERROR] LLM adapter failed:")
    assert "messages_for_llm must end with a 'user' message" in state.raw_answer


@pytest.mark.asyncio
async def test_core_llm_step_calls_adapter_and_sets_answer():
    state = build_runtime_state_for_tests(run_id="run-1")
    state.messages_for_llm = [ChatMessage(role="user", content="u1")]
    adapter = _TrackingFakeLLMAdapter(fixed_text="LLM_OK")
    state.context.config.llm_adapter = adapter
    await CoreLLMStep().run(state)
    assert adapter.called is True
    assert state.raw_answer == "LLM_OK"
    assert isinstance(state.last_llm_adapter_response, LLMAdapterResponse)


@pytest.mark.asyncio
async def test_core_llm_step_emits_llm_call_recorded_diag():
    state = build_runtime_state_for_tests(run_id="run-1")
    state.messages_for_llm = [ChatMessage(role="user", content="u1")]
    state.context.config.llm_adapter = FakeLLMAdapter(fixed_text="OK")
    await CoreLLMStep().run(state)
    recorded = [
        event
        for event in state.trace_events
        if isinstance(event.payload, CoreLLMCallRecordedDiagV1)
    ]
    assert len(recorded) == 1
    payload = recorded[0].payload
    assert payload.finish_reason
    assert payload.prompt_tokens >= 0


@pytest.mark.asyncio
async def test_core_llm_step_passes_max_tokens():
    state = build_runtime_state_for_tests(run_id="run-1")
    state.messages_for_llm = [ChatMessage(role="user", content="u1")]
    state.request.max_output_tokens = 123
    adapter = _TrackingFakeLLMAdapter(fixed_text="OK")
    state.context.config.llm_adapter = adapter
    await CoreLLMStep().run(state)
    assert adapter.kwargs is not None
    assert adapter.kwargs["max_tokens"] == 123


@pytest.mark.asyncio
async def test_core_llm_step_fallback_on_error_with_tools_answer():
    state = build_runtime_state_for_tests(run_id="run-1")
    state.messages_for_llm = [ChatMessage(role="user", content="u1")]
    state.context.config.llm_adapter = _FailingLLMAdapter()
    state.tool_planner_answer = "TOOLS"
    await CoreLLMStep().run(state)
    assert "TOOLS" in state.raw_answer
    assert "[ERROR]" in state.raw_answer


@pytest.mark.asyncio
async def test_core_llm_step_error_without_tools_answer():
    state = build_runtime_state_for_tests(run_id="run-1")
    state.messages_for_llm = [ChatMessage(role="user", content="u1")]
    state.context.config.llm_adapter = _FailingLLMAdapter()
    await CoreLLMStep().run(state)
    assert state.raw_answer.startswith("[ERROR]")
