# © Artur Czarnecki. All rights reserved.

"""Declarative research agent roster (Tier-3 composition contract)."""

from __future__ import annotations

from intergrax.applications._shared.budget_wiring import product_agent_budget_slice
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest
from intergrax.integrations.registry.profile import IntegrationProfile
from research.research_agent import ResearchAgent
from research.summary_agent import SummaryAgent


def _research_environment() -> ApplicationEnvironmentProfile:
    return ApplicationEnvironmentProfile.product_defaults(
        profile_id="research.product",
        skill_bundles=["research"],
    ).model_copy(update={"integration_profile": IntegrationProfile.research_product()}).with_harness_memory().with_reference_host_platform_defaults()


RESEARCH_APPLICATION_MANIFEST = ApplicationManifest.product(
    app_id="research",
    name="Intergrax Research API",
    route_prefix="/v1/research",
    env_prefix="RESEARCH_",
    default_port=8010,
    integration_profile=IntegrationProfile.research_product(),
    environment=_research_environment(),
    agents=[
        AgentBinding.mount(ResearchAgent, budget_slice=product_agent_budget_slice()),
        AgentBinding.mount(SummaryAgent, budget_slice=product_agent_budget_slice()),
    ],
    description="Research → summarize multi-agent host (prototype)",
)
