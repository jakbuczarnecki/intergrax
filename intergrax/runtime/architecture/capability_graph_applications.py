# © Artur Czarnecki. All rights reserved.

"""Per-application capability graph edge mapping (Phase V-REM-CG.1)."""

from __future__ import annotations

from collections.abc import Sequence

from intergrax.contracts.capability_graph_catalog import ApplicationCapabilityCatalogEntry
from intergrax.runtime.architecture.capability_graph import CapabilityEdge, CapabilityEdgeType


def application_capability_node_id(app_id: str) -> str:
    """Stable application node id aligned with ``_system_nodes`` naming."""
    return f"application:{app_id}_application"


def build_application_agent_edges(
    *,
    agent_node_ids: frozenset[str],
    catalog: Sequence[ApplicationCapabilityCatalogEntry],
) -> list[CapabilityEdge]:
    """Map each application host to only its roster agents (not global union)."""
    edges: list[CapabilityEdge] = []
    for entry in catalog:
        application_node = application_capability_node_id(entry.app_id)
        for contract_id in entry.agent_contract_ids:
            agent_node = f"agent:{contract_id}"
            if agent_node not in agent_node_ids:
                continue
            edges.append(
                CapabilityEdge(
                    source_node_id=application_node,
                    target_node_id=agent_node,
                    edge_type=CapabilityEdgeType.DEPENDS_ON,
                )
            )
    return edges
