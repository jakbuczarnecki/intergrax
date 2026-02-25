# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

import pytest
from pydantic import BaseModel

from intergrax.runtime.nexus.errors.tool_scope_violation_error import ToolScopeViolationError
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.runtime.tools.scope_policy import StaticToolScopePolicy
from intergrax.tools.core.contracts import ToolContract
from intergrax.tools.execution_models import ToolExecutionRequest
from intergrax.tools.registry import ToolRegistry
from intergrax.tools.tool_executor import ToolExecutor, ToolHandler

pytestmark = pytest.mark.unit


class DummyInput(BaseModel):
    value: int


class DummyOutput(BaseModel):
    result: int


class DummyState:
    run_id = "run_test"

    def trace_event(
        self,
        *,
        component,
        step,
        message,
        level,
        payload=None,
        artifact_refs=None,
    ) -> None:
        pass


def test_scope_policy_denies_tool_execution():
    registry = ToolRegistry()

    class DummyExecutor(ToolExecutor):
        def execute(self, request: ToolExecutionRequest[BaseModel]) -> BaseModel:
            # Must never be reached when denied.
            return DummyOutput(result=42)

    policy = StaticToolScopePolicy(allowed_tools=set())

    invoker = RuntimeToolInvoker(
        registry=registry,
        executor=DummyExecutor(),
        scope_policy=policy,
    )

    request = ToolExecutionRequest(
        run_id="run1",
        step_id="step1",
        tool_id="forbidden_tool",
        input=DummyInput(value=1),
        idempotency_key=None,
    )

    with pytest.raises(ToolScopeViolationError):
        invoker.invoke(
            state=DummyState(),
            agent_id="agentA",
            request=request,
        )


def test_scope_policy_allows_tool_execution():
    registry = ToolRegistry()

    class DummyHandler(ToolHandler[DummyInput, DummyOutput]):
        def __init__(self) -> None:
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

    result = invoker.invoke(
        state=DummyState(),
        agent_id="agentA",
        request=request,
    )

    assert result.success is True
    assert handler.calls == 1
