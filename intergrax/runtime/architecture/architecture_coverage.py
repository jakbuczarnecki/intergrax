# © Artur Czarnecki. All rights reserved.

"""Governance and observability coverage measurement for Phase V-AM.3."""

from __future__ import annotations

from pydantic import BaseModel, Field

from intergrax.runtime.architecture.capability_graph import CapabilityEdgeType, CapabilityGraph


class ArchitectureCoverageSummary(BaseModel):
    nodes_total: int
    governed_nodes: int
    observed_nodes: int
    governance_coverage: float
    observability_coverage: float


class ArchitectureCoverageReport(BaseModel):
    schema_version: str = "1.0.0"
    summary: ArchitectureCoverageSummary
    uncovered_governance_node_ids: list[str] = Field(default_factory=list)
    uncovered_observability_node_ids: list[str] = Field(default_factory=list)


def compute_architecture_coverage(graph: CapabilityGraph) -> ArchitectureCoverageReport:
    nodes_total = len(graph.nodes)
    if nodes_total == 0:
        summary = ArchitectureCoverageSummary(
            nodes_total=0,
            governed_nodes=0,
            observed_nodes=0,
            governance_coverage=0.0,
            observability_coverage=0.0,
        )
        return ArchitectureCoverageReport(summary=summary)

    governed_targets = {
        edge.source_node_id
        for edge in graph.edges
        if edge.edge_type == CapabilityEdgeType.CONSTRAINED_BY
    }
    observed_targets = {
        edge.target_node_id
        for edge in graph.edges
        if edge.edge_type == CapabilityEdgeType.EVALUATES
    }
    node_ids = {node.node_id for node in graph.nodes}
    uncovered_governance = sorted(node_ids - governed_targets)
    uncovered_observability = sorted(node_ids - observed_targets)
    governed_nodes = len(node_ids) - len(uncovered_governance)
    observed_nodes = len(node_ids) - len(uncovered_observability)

    summary = ArchitectureCoverageSummary(
        nodes_total=nodes_total,
        governed_nodes=governed_nodes,
        observed_nodes=observed_nodes,
        governance_coverage=float(governed_nodes) / float(nodes_total),
        observability_coverage=float(observed_nodes) / float(nodes_total),
    )
    return ArchitectureCoverageReport(
        summary=summary,
        uncovered_governance_node_ids=uncovered_governance,
        uncovered_observability_node_ids=uncovered_observability,
    )
