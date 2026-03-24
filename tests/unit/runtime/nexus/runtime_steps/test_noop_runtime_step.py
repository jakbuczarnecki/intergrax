# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import pytest
from intergrax.runtime.nexus.runtime_steps.noop_runtime_step import NoOpRuntimeStep
from testing_support.builder import build_runtime_state_for_tests

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_noop_runtime_step_does_not_mutate_runtime_state():
    state = build_runtime_state_for_tests(run_id="run-1")

    before_messages = list(state.messages_for_llm)
    before_tools_ctx = list(state.tools_context_parts)
    before_raw_answer = state.raw_answer
    before_runtime_answer = state.runtime_answer
    before_session = state.session
    before_tool_traces = list(state.tool_traces)

    step = NoOpRuntimeStep()
    assert step.execution_kind() is None

    await step.run(state)

    assert state.messages_for_llm == before_messages
    assert state.tools_context_parts == before_tools_ctx
    assert state.raw_answer == before_raw_answer
    assert state.runtime_answer == before_runtime_answer
    assert state.session == before_session
    assert state.tool_traces == before_tool_traces
