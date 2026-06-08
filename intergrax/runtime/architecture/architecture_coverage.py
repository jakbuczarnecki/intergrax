# © Artur Czarnecki. All rights reserved.

"""Governance and observability coverage measurement for Phase V-AM.3."""

from __future__ import annotations

from pydantic import BaseModel, Field

from intergrax.runtime.architecture.capability_graph import (
    CapabilityEdgeType,
    CapabilityGraph,
    CapabilityNodeType,
)

GOVERNANCE_COVERAGE_NODE_TYPES = frozenset(
    {
        CapabilityNodeType.AGENT,
        CapabilityNodeType.APPLICATION,
        CapabilityNodeType.SKILL,
        CapabilityNodeType.PRODUCT,
    }
)
OBSERVABILITY_COVERAGE_NODE_TYPES = frozenset(
    {
        CapabilityNodeType.AGENT,
        CapabilityNodeType.APPLICATION,
    }
)


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

    governance_scope_ids = {
        node.node_id
        for node in graph.nodes
        if node.node_type in GOVERNANCE_COVERAGE_NODE_TYPES
    }
    observability_scope_ids = {
        node.node_id
        for node in graph.nodes
        if node.node_type in OBSERVABILITY_COVERAGE_NODE_TYPES
    }

    if not governance_scope_ids and not observability_scope_ids:
        summary = ArchitectureCoverageSummary(
            nodes_total=0,
            governed_nodes=0,
            observed_nodes=0,
            governance_coverage=0.0,
            observability_coverage=0.0,
        )
        return ArchitectureCoverageReport(summary=summary)

    governed_nodes = len(governance_scope_ids & governed_targets)
    observed_nodes = len(observability_scope_ids & observed_targets)
    uncovered_governance = sorted(governance_scope_ids - governed_targets)
    uncovered_observability = sorted(observability_scope_ids - observed_targets)
    nodes_total = len(governance_scope_ids | observability_scope_ids)

    governance_coverage = (
        float(governed_nodes) / float(len(governance_scope_ids))
        if governance_scope_ids
        else 0.0
    )
    observability_coverage = (
        float(observed_nodes) / float(len(observability_scope_ids))
        if observability_scope_ids
        else 0.0
    )

    summary = ArchitectureCoverageSummary(
        nodes_total=nodes_total,
        governed_nodes=governed_nodes,
        observed_nodes=observed_nodes,
        governance_coverage=governance_coverage,
        observability_coverage=observability_coverage,
    )
    return ArchitectureCoverageReport(
        summary=summary,
        uncovered_governance_node_ids=uncovered_governance,
        uncovered_observability_node_ids=uncovered_observability,
    )
