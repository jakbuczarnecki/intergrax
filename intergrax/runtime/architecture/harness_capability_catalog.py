# © Artur Czarnecki. All rights reserved.

"""Harness application topology compatibility data — NOT agent inventory authority.

Maps reference application hosts to mounted agent contract ids for application→agent
edges only. Agent existence, versions, and capability metadata come from
``AgentCapabilityMetadataProvider`` (AGENT-CONSOLIDATION-2).
"""

from __future__ import annotations

from intergrax.contracts.capability_graph_catalog import ApplicationCapabilityCatalogEntry

HARNESS_CAPABILITY_CATALOG: tuple[ApplicationCapabilityCatalogEntry, ...] = (
    ApplicationCapabilityCatalogEntry(app_id="lab", agent_contract_ids=["echo"]),
    ApplicationCapabilityCatalogEntry(app_id="legal", agent_contract_ids=["legal"]),
    ApplicationCapabilityCatalogEntry(
        app_id="research",
        agent_contract_ids=["research", "research-summary"],
    ),
    ApplicationCapabilityCatalogEntry(app_id="poc_template", agent_contract_ids=["echo"]),
)
