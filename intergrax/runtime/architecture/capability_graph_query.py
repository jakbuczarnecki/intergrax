# © Artur Czarnecki. All rights reserved.

"""Read-only query layer over immutable capability graphs (Stage 13)."""

from __future__ import annotations

from intergrax.runtime.architecture.capability_graph import (
    CapabilityEdgeType,
    CapabilityGraph,
    CapabilityNodeType,
)
from intergrax.runtime.architecture.capability_graph_applications import application_capability_node_id


class CapabilityGraphQuery:
    """Deterministic, read-only capability graph queries."""

    def __init__(self, graph: CapabilityGraph) -> None:
        self._graph = graph
        self._agent_nodes = {
            node.node_id: node
            for node in graph.nodes
            if node.node_type == CapabilityNodeType.AGENT
        }
        self._application_nodes = {
            node.node_id: node
            for node in graph.nodes
            if node.node_type == CapabilityNodeType.APPLICATION
        }
        self._depends_on_edges = [
            edge
            for edge in graph.edges
            if edge.edge_type == CapabilityEdgeType.DEPENDS_ON
        ]

    def agents_for_application(self, application_id: str) -> tuple[str, ...]:
        application_node = application_capability_node_id(application_id)
        if application_node not in self._application_nodes:
            return ()
        agents = [
            edge.target_node_id.removeprefix("agent:")
            for edge in self._depends_on_edges
            if edge.source_node_id == application_node
            and edge.target_node_id.startswith("agent:")
        ]
        return tuple(sorted(agents))

    def applications_for_agent(self, contract_id: str) -> tuple[str, ...]:
        agent_node = f"agent:{contract_id}"
        if agent_node not in self._agent_nodes:
            return ()
        applications = [
            edge.source_node_id.removeprefix("application:").removesuffix("_application")
            for edge in self._depends_on_edges
            if edge.target_node_id == agent_node
            and edge.source_node_id.startswith("application:")
        ]
        return tuple(sorted(applications))

    def capabilities_for_agent(self, contract_id: str) -> tuple[str, ...]:
        agent_node = f"agent:{contract_id}"
        node = self._agent_nodes.get(agent_node)
        if node is None:
            return ()
        raw = node.metadata.get("capabilities", "")
        if not raw:
            return ()
        return tuple(sorted(item.strip() for item in raw.split(",") if item.strip()))

    def agents_with_capability(self, capability_id: str) -> tuple[str, ...]:
        capability = capability_id.strip()
        if not capability:
            return ()
        agents = [
            node.node_id.removeprefix("agent:")
            for node in self._agent_nodes.values()
            if capability in self.capabilities_for_agent(node.node_id.removeprefix("agent:"))
        ]
        return tuple(sorted(agents))
