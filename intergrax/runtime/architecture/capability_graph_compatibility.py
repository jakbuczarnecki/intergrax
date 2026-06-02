# © Artur Czarnecki. All rights reserved.

"""Capability graph compatibility validation (Phase V-CG.4)."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from intergrax.runtime.architecture.capability_graph import (
    CapabilityEdge,
    CapabilityGraph,
    CapabilityNode,
    CapabilityNodeType,
)


class CompatibilitySeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class CompatibilityIssue(BaseModel):
    severity: CompatibilitySeverity
    message: str
    node_id: str | None = None


class CapabilityCompatibilityReport(BaseModel):
    schema_version: str = "1.0.0"
    compatible: bool
    issues: list[CompatibilityIssue] = Field(default_factory=list)


_CRITICAL_NODE_TYPES: set[CapabilityNodeType] = {
    CapabilityNodeType.INTEGRATION,
    CapabilityNodeType.TOOL,
    CapabilityNodeType.SKILL,
    CapabilityNodeType.AGENT,
}


def evaluate_capability_graph_compatibility(
    *,
    previous: CapabilityGraph,
    current: CapabilityGraph,
) -> CapabilityCompatibilityReport:
    issues: list[CompatibilityIssue] = []

    previous_nodes = {node.node_id: node for node in previous.nodes}
    current_nodes = {node.node_id: node for node in current.nodes}
    previous_edges = {_edge_key(edge) for edge in previous.edges}
    current_edges = {_edge_key(edge) for edge in current.edges}

    removed_node_ids = sorted(set(previous_nodes) - set(current_nodes))
    added_node_ids = sorted(set(current_nodes) - set(previous_nodes))
    removed_edges = sorted(previous_edges - current_edges)

    for node_id in removed_node_ids:
        prev_node = previous_nodes[node_id]
        if prev_node.node_type in _CRITICAL_NODE_TYPES:
            issues.append(
                CompatibilityIssue(
                    severity=CompatibilitySeverity.ERROR,
                    message=f"Removed critical capability node: {node_id}",
                    node_id=node_id,
                )
            )
        else:
            issues.append(
                CompatibilityIssue(
                    severity=CompatibilitySeverity.WARNING,
                    message=f"Removed non-critical capability node: {node_id}",
                    node_id=node_id,
                )
            )

    for node_id, prev_node in previous_nodes.items():
        current_node = current_nodes.get(node_id)
        if current_node is None:
            continue
        if current_node.node_type != prev_node.node_type:
            issues.append(
                CompatibilityIssue(
                    severity=CompatibilitySeverity.ERROR,
                    message=(
                        f"Node type changed for {node_id}: "
                        f"{prev_node.node_type.value} -> {current_node.node_type.value}"
                    ),
                    node_id=node_id,
                )
            )

    for source_node_id, edge_type_value, target_node_id in removed_edges:
        issues.append(
            CompatibilityIssue(
                severity=CompatibilitySeverity.WARNING,
                message=(
                    "Removed dependency relation: "
                    f"{source_node_id} -[{edge_type_value}]-> {target_node_id}"
                ),
                node_id=source_node_id,
            )
        )

    for node_id in added_node_ids:
        issues.append(
            CompatibilityIssue(
                severity=CompatibilitySeverity.INFO,
                message=f"Added capability node: {node_id}",
                node_id=node_id,
            )
        )

    compatible = not any(issue.severity == CompatibilitySeverity.ERROR for issue in issues)
    return CapabilityCompatibilityReport(compatible=compatible, issues=issues)


def _edge_key(edge: CapabilityEdge) -> tuple[str, str, str]:
    return (
        edge.source_node_id,
        edge.edge_type.value,
        edge.target_node_id,
    )
