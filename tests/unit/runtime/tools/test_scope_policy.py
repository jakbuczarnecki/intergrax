# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

import pytest

from intergrax.runtime.tools.scope_policy import StaticToolScopePolicy
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.tools.core.contracts import ToolContract
from intergrax.tools.registry import ToolRegistry
from intergrax.tools.tool_executor import ToolExecutor, ToolHandler
from intergrax.tools.execution_models import ToolExecutionRequest
from intergrax.tools.execution_models import ToolExecutionResult
from pydantic import BaseModel


class DummyInput(BaseModel):
    value: int


class DummyOutput(BaseModel):
    result: int



def test_scope_policy_denies_tool_execution():
    registry = ToolRegistry()

    class DummyExecutor(ToolExecutor):
        def execute(self, request: ToolExecutionRequest[BaseModel]) -> BaseModel:
            return ToolExecutionResult.ok(DummyOutput(result=42))

    executor = DummyExecutor()

    policy = StaticToolScopePolicy(allowed_tools=set())

    invoker = RuntimeToolInvoker(
        registry=registry,
        executor=executor,
        scope_policy=policy,
    )

    request = ToolExecutionRequest(
        run_id="run1",
        step_id="step1",
        tool_id="forbidden_tool",
        input=DummyInput(value=1),
        idempotency_key=None,
    )

    class DummyTrace:
        def trace_event(self, **kwargs):
            pass

    result = invoker.invoke(
        state=DummyTrace(),
        agent_id="agentA",
        request=request,
    )

    assert result.success is False


def test_scope_policy_allows_tool_execution():
    registry = ToolRegistry()

    class DummyHandler(ToolHandler[DummyInput, DummyOutput]):
        def __init__(self):
            self.calls = 0

        def execute(self, request: ToolExecutionRequest[DummyInput]) -> DummyOutput:
            self.calls += 1
            return DummyOutput(result=123)

    handler = DummyHandler()

    contract = ToolContract(
        tool_id="allowed_tool",
        name="allowed_tool",
        description="Test tool",
        input_schema=DummyInput,
        output_schema=DummyOutput,
        error_mapping={},
        side_effects=False,
    )

    registry.register(contract, handler)

    policy = StaticToolScopePolicy(allowed_tools={"allowed_tool"})

    class DummyExecutor(ToolExecutor):
        def execute(self, request: ToolExecutionRequest[BaseModel]) -> BaseModel:
            reg = registry.get(request.tool_id)
            return reg.handler.execute(request)


    invoker = RuntimeToolInvoker(
        registry=registry,
        executor=DummyExecutor(),
        scope_policy=policy,
    )

    request = ToolExecutionRequest(
        run_id="run1",
        step_id="step1",
        tool_id="allowed_tool",
        input=DummyInput(value=1),
        idempotency_key=None,
    )

    class DummyTrace:
        def trace_event(self, **kwargs):
            pass

    result = invoker.invoke(
        state=DummyTrace(),
        agent_id="agentA",
        request=request,
    )

    assert result.success is True
    assert handler.calls == 1
