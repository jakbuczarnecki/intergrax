# © Artur Czarnecki. All rights reserved.

"""Declarative research agent roster (Tier-3 composition contract)."""

from __future__ import annotations

from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest
from research.research_agent import ResearchAgent
from research.summary_agent import SummaryAgent


RESEARCH_APPLICATION_MANIFEST = ApplicationManifest.product(
    app_id="research",
    name="Intergrax Research API",
    route_prefix="/v1/research",
    env_prefix="RESEARCH_",
    default_port=8010,
    agents=[
        AgentBinding.mount(ResearchAgent),
        AgentBinding.mount(SummaryAgent),
    ],
    description="Research → summarize multi-agent host (prototype)",
)
