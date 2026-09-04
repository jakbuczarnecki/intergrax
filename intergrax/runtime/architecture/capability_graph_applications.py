# © Artur Czarnecki. All rights reserved.

"""Per-application capability graph projection (Phase V-REM-CG.1 / Stage 13)."""

from __future__ import annotations

from collections.abc import Sequence

from intergrax.contracts.application_capability_metadata import (
    ApplicationCapabilityDescriptor,
    ApplicationCapabilityProjectionConflict,
)
from intergrax.runtime.architecture.capability_graph import (
    CapabilityEdge,
    CapabilityEdgeType,
    CapabilityNode,
    CapabilityNodeType,
)


def application_capability_node_id(app_id: str) -> str:
    """Stable application node id aligned with harness capability graph naming."""
    return f"application:{app_id}_application"


def application_nodes_from_descriptors(
    descriptors: Sequence[ApplicationCapabilityDescriptor],
) -> list[CapabilityNode]:
    """Build application inventory nodes from projected descriptors."""
    nodes: list[CapabilityNode] = []
    for descriptor in descriptors:
        metadata: dict[str, str] = {}
        if descriptor.default_capability is not None:
            metadata["default_capability"] = descriptor.default_capability
        nodes.append(
            CapabilityNode(
                node_id=application_capability_node_id(descriptor.application_id),
                node_type=CapabilityNodeType.APPLICATION,
                version=descriptor.application_version,
                metadata=metadata,
            )
        )
    return nodes


def build_application_agent_edges(
    *,
    agent_node_ids: frozenset[str],
    descriptors: Sequence[ApplicationCapabilityDescriptor],
) -> list[CapabilityEdge]:
    """Map each application host to its roster agents; fail closed on missing agent metadata."""
    edges: list[CapabilityEdge] = []
    for descriptor in descriptors:
        application_node = application_capability_node_id(descriptor.application_id)
        for contract_id in descriptor.agent_contract_ids:
            agent_node = f"agent:{contract_id}"
            if agent_node not in agent_node_ids:
                raise ApplicationCapabilityProjectionConflict(
                    f"application {descriptor.application_id!r} references agent contract "
                    f"{contract_id!r} missing from agent metadata projection",
                )
            edges.append(
                CapabilityEdge(
                    source_node_id=application_node,
                    target_node_id=agent_node,
                    edge_type=CapabilityEdgeType.DEPENDS_ON,
                )
            )
    return edges
