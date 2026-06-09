# © Artur Czarnecki. All rights reserved.

"""Declarative agent roster for dispute_sim_application (Tier-3 product host)."""

from __future__ import annotations

from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest
from intergrax.fastapi_core.config import ApiEnvironment
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
from dispute_sim_application.host.environment_profile import build_dispute_sim_environment_profile
from dispute_sim_application.host.settings import DisputeSimBackendSettings


def _dispute_sim_environment() -> ApplicationEnvironmentProfile:
    return build_dispute_sim_environment_profile(
        DisputeSimBackendSettings(environment=ApiEnvironment.DEV),
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
        ),
        AgentBinding.mount(
            DisputeAnalystAgent,
            factory=build_dispute_sim_dispute_analyst_from_context,
            capabilities=["dispute.analyze", "dispute.pipeline", "dispute.correspondence"],
        ),
        AgentBinding.mount(
            DisputeStrategistAgent,
            factory=build_dispute_sim_dispute_strategist_from_context,
            capabilities=["dispute.strategy", "dispute.full_pipeline"],
        ),
        AgentBinding.mount(
            DisputeScenarioAgent,
            factory=build_dispute_sim_dispute_scenario_from_context,
            capabilities=["dispute.scenario", "dispute.pipeline", "dispute.correspondence"],
        ),
    ],
    description="Dispute Simulation Workspace — multi-agent litigation prep and scenario host",
)

# Backward-compatible alias for scaffold-generated imports.
APPLICATION_MANIFEST = DISPUTE_SIM_APPLICATION_MANIFEST


def build_dispute_sim_manifest() -> ApplicationManifest:
    return DISPUTE_SIM_APPLICATION_MANIFEST
