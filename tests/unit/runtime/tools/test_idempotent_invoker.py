import pytest
from pydantic import BaseModel
from intergrax.applications._shared.policy_wiring import wire_policy_bundle
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    PolicyRulesProfile,
)
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
from testing_support.builder import (
    build_runtime_state_for_tests,
    canonical_execution_identity_scope,
    canonical_run_id_for_tests,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_RUN_SEED = "run1"
_RUN_ID = canonical_run_id_for_tests(_RUN_SEED)


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


def _enforce_allow_bundle() -> object:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="tools.se.allow")
    env.policy_rules = PolicyRulesProfile(
        inline_rules=[],
        policy_enforcement_mode="enforce",
    )
    return wire_policy_bundle(env)


def _state_with_enforce_allow():
    state = build_runtime_state_for_tests(run_id=_RUN_SEED)
    state.context.config.policy_bundle = _enforce_allow_bundle()
    return state


class DummyHandler:
    def execute(self, request):
        return DummyOutput(result=request.input.value * 2)


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

    state = _state_with_enforce_allow()

    request = ToolExecutionRequest(
        run_id=_RUN_ID,
        step_id="step1",
        tool_id="double",
        input=DummyInput(value=5),
        idempotency_key="key-123",
    )

    agent_id: str = "agent-test"

    with canonical_execution_identity_scope(_RUN_SEED):
        r1 = invoker.invoke(state=state, agent_id=agent_id, request=request)
        r2 = invoker.invoke(state=state, agent_id=agent_id, request=request)

    assert r1.success
    assert r2.success
    assert r1.output == r2.output
    assert executor.calls == 1
