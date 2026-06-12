# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from pydantic import BaseModel
import pytest

from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.tools.core.contracts import ToolContract
from intergrax.tools.execution_models import ToolExecutionRequest, ToolExecutionResult
from testing_support.builder import build_runtime_state_for_tests

pytestmark = pytest.mark.unit

class InputModel(BaseModel):
    value: int


class OutputModel(BaseModel):
    result: int


class FakeExecutor:
    def execute(self, request: ToolExecutionRequest[BaseModel]) -> BaseModel:
        if not isinstance(request.input, InputModel):
            raise TypeError("Expected InputModel as tool input.")

        return OutputModel(result=request.input.value + 1)


from tests.unit.runtime.nexus.tools.conftest import FakeRegistry


class AllowAllScopePolicy:
    def is_allowed(self, *args, **kwargs) -> bool:
        return True


def test_runtime_tool_invoker_emits_duration_ms():
    contract = ToolContract(
        tool_id="test_tool",
        name="test_tool",
        description="test_tool",
        input_schema=InputModel,
        output_schema=OutputModel,
        side_effects=False,
        error_mapping={},
    )

    registry = FakeRegistry(contract)
    executor = FakeExecutor()
    scope_policy = AllowAllScopePolicy()

    invoker = RuntimeToolInvoker(
        registry=registry,
        executor=executor,
        scope_policy=scope_policy,
    )

    state = build_runtime_state_for_tests(run_id="test_run")

    request = ToolExecutionRequest(
        run_id="test_run",
        tool_id="test_tool",
        step_id="1",
        input=InputModel(value=10),
    )

    result = invoker.invoke(state=state, agent_id="agent_test", request=request)

    assert isinstance(result, ToolExecutionResult)
    assert result.success is True

    end_events = [
        e for e in state.trace_events
        if e.step == "tool_invocation_end"
    ]

    assert len(end_events) == 1

    payload = end_events[0].payload

    assert hasattr(payload, "duration_ms")
    assert isinstance(payload.duration_ms, int)
    assert payload.duration_ms >= 0