# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import pytest
from intergrax.runtime.nexus.runtime_steps.build_base_history_step import BuildBaseHistoryStep
from tests._support.builder import build_runtime_state_for_tests

pytestmark = pytest.mark.unit


class _FakeHistoryLayer:
    def __init__(self):
        self.called = False
        self.last_state = None

    async def build_base_history(self, state):
        self.called = True
        self.last_state = state


@pytest.mark.asyncio
async def test_build_base_history_step_calls_history_layer():
    state = build_runtime_state_for_tests(run_id="run-1")
    fake_layer = _FakeHistoryLayer()
    state.context.history_layer = fake_layer

    step = BuildBaseHistoryStep()
    assert step.execution_kind() is None

    before_messages = list(state.messages_for_llm)
    before_tools_ctx = list(state.tools_context_parts)

    await step.run(state)

    assert fake_layer.called is True
    assert fake_layer.last_state is state
    assert state.messages_for_llm == before_messages
    assert state.tools_context_parts == before_tools_ctx
