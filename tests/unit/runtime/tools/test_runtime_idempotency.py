# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

import pytest
from pydantic import BaseModel

from intergrax.runtime.nexus.runtime_steps.tools_step import ToolsStep
from intergrax.runtime.tools.in_memory_idempotency_store import InMemoryIdempotencyStore
from intergrax.tools.core.contracts import ToolContract
from intergrax.tools.core.provider import ToolProvider
from tests._support.builder import build_runtime_state_for_tests

pytestmark = pytest.mark.unit


class DummyInput(BaseModel):
    value: int


class DummyOutput(BaseModel):
    result: int


class CountingHandler:
    def __init__(self):
        self.calls = 0

    def execute(self, request):
        self.calls += 1
        return DummyOutput(result=request.input.value * 2)


class DummyProvider(ToolProvider):
    def __init__(self, handler):
        self._handler = handler

    def register_tools(self, registry):
        contract = ToolContract(
            tool_id="double",
            name="double",
            description="double",
            input_schema=DummyInput,
            output_schema=DummyOutput,
            error_mapping={},
            side_effects=True,
        )
        registry.register(contract, self._handler)


class FakeToolsAgent:
    def plan_tools(self, input_data, context, run_id):
        class Call:
            step_id = "step1"
            tool_id = "double"
            input = DummyInput(value=5)

        class Plan:
            calls = [Call()]

        class Decision:
            tool_plan = Plan()

        return Decision()


async def test_tools_step_idempotent_retry():
    store = InMemoryIdempotencyStore()
    handler = CountingHandler()

    # Build minimal state with proper RuntimeContext
    state = build_runtime_state_for_tests(run_id="run-1")

    # Configure runtime properly BEFORE rebuilding context
    state.context.config.tools_mode = "auto"
    state.context.config.tools_agent = FakeToolsAgent()
    state.context.config.idempotency_store = store
    state.context.config.tool_providers = [DummyProvider(handler)]

    # Rebuild context to activate:
    # - ToolRegistry creation
    # - Provider registration
    # - IdempotentToolInvoker injection
    from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext

    state.context = RuntimeContext.build(
        config=state.context.config,
        session_manager=state.context.session_manager,
    )

    state.request.message = "use tool"
    state.tool_traces = []
    state.used_tools = False
    state.messages_for_llm = []

    step = ToolsStep()

    await step.run(state)
    await step.run(state)

    assert handler.calls == 1
