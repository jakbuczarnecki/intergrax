# © Artur Czarnecki. All rights reserved.

"""Declarative agent roster for dispute_sim_application (Tier-3 product host)."""

from __future__ import annotations

from intergrax.applications._shared.budget_wiring import product_agent_budget_slice
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest
from intergrax.integrations.registry.profile import IntegrationProfile
from dispute_analyst.dispute_analyst_agent import DisputeAnalystAgent
from dispute_intake.dispute_intake_agent import DisputeIntakeAgent
from dispute_scenario.dispute_scenario_agent import DisputeScenarioAgent
from dispute_strategist.dispute_strategist_agent import DisputeStrategistAgent
from dispute_sim_application.host.agent_factories import (
    build_dispute_sim_dispute_analyst_from_context,
    build_dispute_sim_dispute_intake_from_context,
    build_dispute_sim_dispute_scenario_from_context,
    build_dispute_sim_dispute_strategist_from_context,
)


def _dispute_sim_environment() -> ApplicationEnvironmentProfile:
    return (
        ApplicationEnvironmentProfile.product_defaults(
            profile_id="dispute_sim.product",
            skill_bundles=["harness", "legal"],
        )
        .model_copy(
            update={
                "integration_profile": IntegrationProfile.legal_product(),
                "context_profile": ApplicationEnvironmentProfile.product_defaults()
                .context_profile.model_copy(
                    update={"enable_rag": True, "enable_websearch": False}
                ),
            }
        )
        .with_harness_memory()
        .with_reference_host_platform_defaults(multi_agent_critic=True)
    )


DISPUTE_SIM_APPLICATION_MANIFEST = ApplicationManifest.product(
    app_id="dispute_sim",
    name="Intergrax Dispute Simulation Workspace API",
    route_prefix="/v1/dispute_sim",
    env_prefix="DISPUTE_SIM_",
    default_port=8025,
    default_capability="dispute.intake",
    integration_profile=IntegrationProfile.legal_product(),
    environment=_dispute_sim_environment(),
    agents=[
        AgentBinding.mount(
            DisputeIntakeAgent,
            factory=build_dispute_sim_dispute_intake_from_context,
            capabilities=["dispute.intake"],
            default=True,
            budget_slice=product_agent_budget_slice(),
        ),
        AgentBinding.mount(
            DisputeAnalystAgent,
            factory=build_dispute_sim_dispute_analyst_from_context,
            capabilities=["dispute.analyze"],
            budget_slice=product_agent_budget_slice(),
        ),
        AgentBinding.mount(
            DisputeStrategistAgent,
            factory=build_dispute_sim_dispute_strategist_from_context,
            capabilities=["dispute.strategy"],
            budget_slice=product_agent_budget_slice(),
        ),
        AgentBinding.mount(
            DisputeScenarioAgent,
            factory=build_dispute_sim_dispute_scenario_from_context,
            capabilities=["dispute.scenario"],
            budget_slice=product_agent_budget_slice(),
        ),
    ],
    description="Dispute Simulation Workspace — multi-agent litigation prep and scenario host",
)

# Backward-compatible alias for scaffold-generated imports.
APPLICATION_MANIFEST = DISPUTE_SIM_APPLICATION_MANIFEST


def build_dispute_sim_manifest() -> ApplicationManifest:
    return DISPUTE_SIM_APPLICATION_MANIFEST
