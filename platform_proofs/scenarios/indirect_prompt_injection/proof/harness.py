"""Proof harness — provider control, application execution, and observation."""

from __future__ import annotations

from platform_proofs.scenarios.indirect_prompt_injection.application.order_operations_port import (
    OrderOperationsPort,
)
from platform_proofs.scenarios.indirect_prompt_injection.application.order_provider_control_port import (
    OrderProviderControlPort,
)
from platform_proofs.scenarios.indirect_prompt_injection.application.run_bundle import (
    OrderAssistantRunBundle,
)
from platform_proofs.scenarios.indirect_prompt_injection.application.scenario import (
    execute_order_assistant_run,
)
from platform_proofs.scenarios.indirect_prompt_injection.fixtures.orders import ScenarioFixture
from platform_proofs.scenarios.indirect_prompt_injection.proof.scenario_result import (
    ScenarioExecutionResult,
    scenario_result_from_application,
)


def prepare_provider_fixture(
    provider_control: OrderProviderControlPort,
    fixture: ScenarioFixture,
) -> None:
    provider_control.reset(notes=list(fixture.provider_notes))


async def execute_observed_scenario_run(
    *,
    run_bundle: OrderAssistantRunBundle,
    order_operations: OrderOperationsPort,
    provider_control: OrderProviderControlPort,
) -> ScenarioExecutionResult:
    order_id = run_bundle.order_id
    initial_order_state = order_operations.get_order(order_id)
    application_result = await execute_order_assistant_run(run_bundle)
    provider_state = provider_control.mutation_state()
    final_order_state = order_operations.get_order(order_id)
    return scenario_result_from_application(
        application_result,
        initial_order_state=initial_order_state,
        final_order_state=final_order_state,
        provider_write_count=provider_state.write_count,
    )


async def execute_fixture_scenario_run(
    *,
    run_bundle: OrderAssistantRunBundle,
    order_operations: OrderOperationsPort,
    provider_control: OrderProviderControlPort,
    fixture: ScenarioFixture,
) -> ScenarioExecutionResult:
    prepare_provider_fixture(provider_control, fixture)
    return await execute_observed_scenario_run(
        run_bundle=run_bundle,
        order_operations=order_operations,
        provider_control=provider_control,
    )
