import pytest
from pydantic import BaseModel
from intergrax.tools.execution_models import (
    ToolExecutionRequest,
)
from intergrax.tools.core.contracts import ToolContract
from intergrax.tools.registry import ToolRegistry
from intergrax.runtime.tools.in_memory_idempotency_store import (
    InMemoryIdempotencyStore,
)
from intergrax.runtime.tools.idempotency_pre_effect_coordinator import (
    IdempotencyPreEffectCoordinator,
)
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker

pytestmark = [pytest.mark.unit, pytest.mark.gate]


class DummyInput(BaseModel):
    value: int


class DummyOutput(BaseModel):
    result: int


class CountingExecutor:
    def __init__(self):
        self.calls = 0

    def execute(self, request):
        self.calls += 1
        return DummyOutput(result=request.input.value * 2)


class DummyState:
    def __init__(self) -> None:
        self._tenant_id = "tenant_test"

    @property
    def tenant_id(self) -> str:
        return self._tenant_id

    @property
    def context(self):
        return type("Ctx", (), {"config": type("Cfg", (), {"policy_bundle": None})()})()

    def trace_event(self, *args, **kwargs):
        pass

class DummyHandler:
    def execute(self, request):
        return DummyOutput(...)



def test_side_effect_tool_is_idempotent():
    registry = ToolRegistry()

    contract = ToolContract(
        tool_id="double",
        name="double",
        description="double value",
        input_schema=DummyInput,
        output_schema=DummyOutput,
        error_mapping={},
        side_effects=True,
    )

    registry.register(
        contract=contract,
        handler=DummyHandler(),
    )

    executor = CountingExecutor()
    store = InMemoryIdempotencyStore()
    coordinator = IdempotencyPreEffectCoordinator(idempotency_store=store)
    invoker = RuntimeToolInvoker(
        registry=registry,
        executor=executor,
        pre_effect_coordinator=coordinator,
    )

    state = DummyState()

    request = ToolExecutionRequest(
        run_id="run1",
        step_id="step1",
        tool_id="double",
        input=DummyInput(value=5),
        idempotency_key="key-123",
    )

    agent_id:str = "agent-test"

    r1 = invoker.invoke(state=state, agent_id=agent_id, request=request)
    r2 = invoker.invoke(state=state, agent_id=agent_id, request=request)

    assert r1.success
    assert r2.success
    assert r1.output == r2.output
    assert executor.calls == 1
