from pydantic import BaseModel
from intergrax.tools.execution_models import (
    ToolExecutionRequest,
)
from intergrax.tools.core.contracts import ToolContract
from intergrax.tools.registry import ToolRegistry
from intergrax.runtime.tools.in_memory_idempotency_store import (
    InMemoryIdempotencyStore,
)
from intergrax.runtime.tools.idempotent_invoker import (
    IdempotentToolInvoker,
)
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker


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
    def trace_event(self, **kwargs):
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
    base_invoker = RuntimeToolInvoker(
        registry=registry,
        executor=executor,
    )

    store = InMemoryIdempotencyStore()
    invoker = IdempotentToolInvoker(
        base_invoker=base_invoker,
        idempotency_store=store,
    )

    state = DummyState()

    request = ToolExecutionRequest(
        run_id="run1",
        step_id="step1",
        tool_id="double",
        input=DummyInput(value=5),
        idempotency_key="key-123",
    )

    r1 = invoker.invoke(state=state, request=request)
    r2 = invoker.invoke(state=state, request=request)

    assert r1.success
    assert r2.success
    assert r1.output == r2.output
    assert executor.calls == 1
