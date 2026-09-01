"""Runtime bundle composition for scenario proof runs."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.tools.registry import ToolRegistry

from platform_proofs.scenarios.indirect_prompt_injection.application.agent import OrderAssistantAgent
from platform_proofs.scenarios.indirect_prompt_injection.application.order_provider_client import (
    OrderProviderClient,
)
from platform_proofs.scenarios.indirect_prompt_injection.application.run_bundle import (
    OrderAssistantRunBundle,
)
from platform_proofs.scenarios.indirect_prompt_injection.application.runtime_composition import (
    ScenarioRuntimeComposition,
    SYNTHETIC_SCENARIO_TENANT_ID,
    build_scenario_environment_profile,
    build_scenario_runtime_composition,
)
from platform_proofs.scenarios.indirect_prompt_injection.application.workflows import (
    ControlKind,
    WorkflowKind,
)
from platform_proofs.scenarios.indirect_prompt_injection.fixtures.orders import ScenarioFixture


@dataclass(frozen=True, slots=True)
class FixtureRuntimeBundle:
    fixture: ScenarioFixture
    run_bundle: OrderAssistantRunBundle


def workflow_for_fixture(fixture: ScenarioFixture) -> WorkflowKind:
    if fixture.control_kind is ControlKind.AUTHORIZED_WRITE:
        return WorkflowKind.AUTHORIZED_WRITE
    return WorkflowKind.SAFE_READ


def build_fixture_runtime_bundle(
    fixture: ScenarioFixture,
    *,
    provider_client: OrderProviderClient | None = None,
    llm_adapter_override: LLMAdapter | None = None,
    tenant_id: str = SYNTHETIC_SCENARIO_TENANT_ID,
) -> FixtureRuntimeBundle:
    resolved_client = provider_client or OrderProviderClient()
    resolved_client.reset(notes=list(fixture.provider_notes))
    workflow = workflow_for_fixture(fixture)
    tool_registry = ToolRegistry()
    environment = build_scenario_environment_profile(workflow)
    composition = ScenarioRuntimeComposition(
        environment=environment,
        tool_registry=tool_registry,
        llm_adapter_override=llm_adapter_override,
    )
    composition = build_scenario_runtime_composition(
        registry=tool_registry,
        environment=environment,
        tenant_id=tenant_id,
        composition=composition,
        provider_client=resolved_client,
    )
    agent = OrderAssistantAgent(
        registry=composition.tool_registry,
        runtime_composition=composition,
        provider_client=resolved_client,
        workflow=workflow,
        order_id=fixture.order_id,
        user_message=fixture.user_message,
    )
    if agent.get_contract().id not in composition.platform.registry.list_agent_ids():
        composition.platform.registry.register(agent)
    run_bundle = OrderAssistantRunBundle(
        workflow=workflow,
        provider_client=resolved_client,
        agent=agent,
        runtime_composition=composition,
        order_id=fixture.order_id,
        user_message=fixture.user_message,
    )
    return FixtureRuntimeBundle(fixture=fixture, run_bundle=run_bundle)
