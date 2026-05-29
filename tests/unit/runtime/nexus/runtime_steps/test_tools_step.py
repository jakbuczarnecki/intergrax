import pytest

from intergrax.runtime.nexus.runtime_steps.tools_step import ToolsStep
from intergrax.tools.execution_models import ToolExecutionResult
from intergrax.tools.execution_models import ToolExecutionRequest
from intergrax.tools.registry import ToolRegistry
from intergrax.tools.tool_executor import ToolHandler

from pydantic import BaseModel

from intergrax.tools.tools_agent import ToolPlanDecision
from testing_support.builder import FakeLLMAdapter, build_runtime_state_for_tests, tools_agent_make_contract

pytestmark = pytest.mark.unit


class DummyInput(BaseModel):
    x: int

class DummyOutput(BaseModel):
    y: int

class DummyHandler(ToolHandler[DummyInput, DummyOutput]):
    def execute(self, request: ToolExecutionRequest[DummyInput]) -> ToolExecutionResult[DummyOutput]:
        return ToolExecutionResult.success(DummyOutput(y=request.input.x + 1))



class FakeToolsAgent:
    def plan_tools(self, input_data, context=None, run_id=None, **kwargs):
        from intergrax.tools.core.tool_plan import ToolCallPlan, PlannedToolCall

        return ToolPlanDecision(
            final_answer=None,
            tool_plan=ToolCallPlan(
                calls=[
                    PlannedToolCall(
                        step_id="tool",
                        tool_id="dummy_tool",
                        input=DummyInput(x=1),
                    )
                ]
            ),
            messages=[],
        )


class FakeInvoker:
    def invoke(self, state, request, agent_id):
        return ToolExecutionResult.ok(DummyOutput(y=42))


@pytest.mark.asyncio
async def test_tools_step_executes_tool_and_updates_state():
    registry = ToolRegistry()

    contract = tools_agent_make_contract("dummy_tool", DummyInput, DummyOutput)

    registry.register(contract, DummyHandler())

    state = build_runtime_state_for_tests(run_id="run-1",)    

    state.cap_tools_available = True    
    state.context.config.tools_agent = FakeToolsAgent()
    state.context.config.tool_invoker = FakeInvoker()
    state.context.config.tools_mode = "auto"
    state.context.config.llm_adapter = FakeLLMAdapter()

    state.request.message = "use tool"

    step = ToolsStep()

    await step.run(state)

    assert state.used_tools is True
    assert len(state.tool_traces) == 1
    trace = state.tool_traces[0]
    assert trace.tool_name == "dummy_tool"
    assert trace.success is True


