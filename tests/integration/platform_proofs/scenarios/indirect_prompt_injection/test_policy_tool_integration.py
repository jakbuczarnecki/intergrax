from __future__ import annotations

import pytest
from pydantic import BaseModel

from intergrax.applications._shared.policy_wiring import wire_policy_bundle
from intergrax.runtime.nexus.errors.declarative_policy_violation_error import (
    DeclarativePolicyViolationError,
)
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.tools.execution_models import ToolExecutionRequest
from intergrax.tools.registry import ToolRegistry
from intergrax.tools.tool_executor import ToolExecutor
from intergrax.runtime.nexus.tools import RegistryToolExecutor
from intergrax.contracts.execution_identity import (
    bind_active_execution_identity,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    reset_active_execution_identity,
)
from testing_support.builder import build_runtime_state_for_tests

from platform_proofs.scenarios.indirect_prompt_injection.application.tools import (
    TOOL_ORDER_UPDATE_SHIPPING_ADDRESS,
    UpdateShippingAddressInput,
    UpdateShippingAddressOutput,
    register_scenario_tools,
)
from platform_proofs.scenarios.indirect_prompt_injection.application.workflows import (
    WorkflowKind,
    build_scenario_environment_profile,
)
from tests.unit.platform_proofs.scenarios.indirect_prompt_injection.in_process_order_provider import (
    InProcessOrderProviderClient,
)

pytestmark = [pytest.mark.integration, pytest.mark.unit]


class _CountingExecutor(ToolExecutor):
    def __init__(self) -> None:
        self.calls = 0

    def execute(self, request: ToolExecutionRequest[BaseModel]) -> BaseModel:
        self.calls += 1
        return UpdateShippingAddressOutput(
            order_id="48291",
            status="processing",
            shipping_address="123 Attacker Lane",
            fulfillment_status="address_updated",
            confirmation="shipping_address_updated",
        )


def test_runtime_tool_invoker_denies_write_on_read_only_policy() -> None:
    env = build_scenario_environment_profile(WorkflowKind.SAFE_READ)
    bundle = wire_policy_bundle(env)
    registry = ToolRegistry()
    client = InProcessOrderProviderClient()
    register_scenario_tools(registry, provider_client=client)
    executor = _CountingExecutor()
    invoker = RuntimeToolInvoker(registry=registry, executor=executor, scope_policy=None)
    run_id = mint_run_id()
    state = build_runtime_state_for_tests(run_id=run_id)
    state.context.config.policy_bundle = bundle

    request = ToolExecutionRequest(
        run_id=run_id,
        tool_id=TOOL_ORDER_UPDATE_SHIPPING_ADDRESS,
        step_id="1",
        input=UpdateShippingAddressInput(
            order_id="48291",
            new_shipping_address="123 Attacker Lane",
        ),
    )

    token = bind_active_execution_identity(
        run_id=run_id,
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
    )
    try:
        with pytest.raises(DeclarativePolicyViolationError):
            invoker.invoke(state=state, agent_id="order_assistant", request=request)
    finally:
        reset_active_execution_identity(token)

    assert executor.calls == 0
    assert client.mutation_state().write_count == 0


def test_runtime_tool_invoker_allows_write_on_authorized_workflow() -> None:
    env = build_scenario_environment_profile(WorkflowKind.AUTHORIZED_WRITE)
    bundle = wire_policy_bundle(env)
    registry = ToolRegistry()
    client = InProcessOrderProviderClient()
    register_scenario_tools(registry, provider_client=client)
    executor = RegistryToolExecutor(registry)
    invoker = RuntimeToolInvoker(registry=registry, executor=executor, scope_policy=None)
    run_id = mint_run_id()
    state = build_runtime_state_for_tests(run_id=run_id)
    state.context.config.policy_bundle = bundle

    request = ToolExecutionRequest(
        run_id=run_id,
        tool_id=TOOL_ORDER_UPDATE_SHIPPING_ADDRESS,
        step_id="1",
        input=UpdateShippingAddressInput(
            order_id="48291",
            new_shipping_address="456 Oak Street",
        ),
    )

    token = bind_active_execution_identity(
        run_id=run_id,
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
    )
    try:
        result = invoker.invoke(state=state, agent_id="order_assistant", request=request)
    finally:
        reset_active_execution_identity(token)
    assert result.success is True
    assert client.mutation_state().write_count == 1
