# © Artur Czarnecki. All rights reserved.

"""Capability graph lineage and blast-radius reporting (Phase V-CG.2/V-CG.3)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from intergrax.runtime.architecture.capability_graph import CapabilityGraph


class CapabilityLineageRecord(BaseModel):
    node_id: str
    upstream_node_ids: list[str] = Field(default_factory=list)
    downstream_node_ids: list[str] = Field(default_factory=list)


class CapabilityLineageReport(BaseModel):
    schema_version: str = "1.0.0"
    records: list[CapabilityLineageRecord] = Field(default_factory=list)


class CapabilityImpactRecord(BaseModel):
    node_id: str
    blast_radius_node_ids: list[str] = Field(default_factory=list)


class CapabilityImpactReport(BaseModel):
    schema_version: str = "1.0.0"
    impacts: list[CapabilityImpactRecord] = Field(default_factory=list)


def build_capability_lineage_report(graph: CapabilityGraph) -> CapabilityLineageReport:
    upstream: dict[str, set[str]] = {node.node_id: set() for node in graph.nodes}
    downstream: dict[str, set[str]] = {node.node_id: set() for node in graph.nodes}

    for edge in graph.edges:
        upstream[edge.target_node_id].add(edge.source_node_id)
        downstream[edge.source_node_id].add(edge.target_node_id)

    records = [
        CapabilityLineageRecord(
            node_id=node.node_id,
            upstream_node_ids=sorted(upstream[node.node_id]),
            downstream_node_ids=sorted(downstream[node.node_id]),
        )
        for node in sorted(graph.nodes, key=lambda item: item.node_id)
    ]
    return CapabilityLineageReport(records=records)


def build_capability_impact_report(graph: CapabilityGraph) -> CapabilityImpactReport:
    adjacency: dict[str, set[str]] = {node.node_id: set() for node in graph.nodes}
    for edge in graph.edges:
        adjacency[edge.source_node_id].add(edge.target_node_id)

    impacts: list[CapabilityImpactRecord] = []
    for node_id in sorted(adjacency):
        visited: set[str] = set()
        queue: list[str] = sorted(adjacency[node_id])
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            for child in sorted(adjacency[current]):
                if child not in visited:
                    queue.append(child)
        impacts.append(
            CapabilityImpactRecord(
                node_id=node_id,
                blast_radius_node_ids=sorted(visited),
            )
        )
    return CapabilityImpactReport(impacts=impacts)
