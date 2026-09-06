# © Artur Czarnecki. All rights reserved.

"""Declarative research agent roster (Tier-3 composition contract)."""

from __future__ import annotations

from intergrax.applications._shared.agent_certification_wiring import apply_roster_agent_governance
from intergrax.applications._shared.budget_wiring import product_agent_budget_slice
from intergrax.applications._shared.ownership_wiring import standard_product_operational_ownership
from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest
from intergrax.integrations.registry.profile import IntegrationProfile
from research.research_agent import ResearchAgent
from research.summary_agent import SummaryAgent
from research_application.host.environment_profile import build_research_environment_profile


_RESEARCH_AGENTS = [
    AgentBinding.mount(
        ResearchAgent,
        contract_id="research",
        budget_slice=product_agent_budget_slice(),
    ),
    AgentBinding.mount(
        SummaryAgent,
        contract_id="research-summary",
        budget_slice=product_agent_budget_slice(),
    ),
]


def _research_environment():
    base = build_research_environment_profile()
    return apply_roster_agent_governance(base, agents=_RESEARCH_AGENTS, app_id="research")


RESEARCH_APPLICATION_MANIFEST = ApplicationManifest.product(
    app_id="research",
    name="Intergrax Research API",
    route_prefix="/v1/research",
    env_prefix="RESEARCH_",
    default_port=8010,
    integration_profile=IntegrationProfile.research_product(),
    environment=_research_environment(),
    agents=list(_RESEARCH_AGENTS),
    description="Research → summarize multi-agent host (prototype)",
    ownership=standard_product_operational_ownership("research"),
)
