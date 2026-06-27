# © Artur Czarnecki. All rights reserved.

"""Static harness reference catalog for runtime capability graph edges."""

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
