from __future__ import annotations

from intergrax.runtime.architecture.capability_graph import (
    CapabilityEdge,
    CapabilityEdgeType,
    CapabilityGraph,
    CapabilityNode,
    CapabilityNodeType,
)
from intergrax.runtime.architecture.capability_graph_compatibility import (
    CompatibilitySeverity,
    evaluate_capability_graph_compatibility,
)


def _graph_with_tool() -> CapabilityGraph:
    return CapabilityGraph(
        nodes=[
            CapabilityNode(node_id="integration:sqlite", node_type=CapabilityNodeType.INTEGRATION),
            CapabilityNode(node_id="tool:rag.retrieve", node_type=CapabilityNodeType.TOOL),
        ],
        edges=[
            CapabilityEdge(
                source_node_id="tool:rag.retrieve",
                target_node_id="integration:sqlite",
                edge_type=CapabilityEdgeType.DEPENDS_ON,
            )
        ],
    )


def test_compatibility_detects_removed_critical_node_as_error() -> None:
    previous = _graph_with_tool()
    current = CapabilityGraph(
        nodes=[CapabilityNode(node_id="integration:sqlite", node_type=CapabilityNodeType.INTEGRATION)],
        edges=[],
    )
    report = evaluate_capability_graph_compatibility(previous=previous, current=current)
    assert report.compatible is False
    assert any(issue.severity == CompatibilitySeverity.ERROR for issue in report.issues)


def test_compatibility_detects_type_change_as_error() -> None:
    previous = _graph_with_tool()
    current = CapabilityGraph(
        nodes=[
            CapabilityNode(node_id="integration:sqlite", node_type=CapabilityNodeType.INTEGRATION),
            CapabilityNode(node_id="tool:rag.retrieve", node_type=CapabilityNodeType.SKILL),
        ],
        edges=[],
    )
    report = evaluate_capability_graph_compatibility(previous=previous, current=current)
    assert report.compatible is False
    assert any("Node type changed" in issue.message for issue in report.issues)


def test_compatibility_marks_added_nodes_as_info() -> None:
    previous = _graph_with_tool()
    current = CapabilityGraph(
        nodes=[
            CapabilityNode(node_id="integration:sqlite", node_type=CapabilityNodeType.INTEGRATION),
            CapabilityNode(node_id="tool:rag.retrieve", node_type=CapabilityNodeType.TOOL),
            CapabilityNode(node_id="skill:research.scan", node_type=CapabilityNodeType.SKILL),
        ],
        edges=[
            CapabilityEdge(
                source_node_id="tool:rag.retrieve",
                target_node_id="integration:sqlite",
                edge_type=CapabilityEdgeType.DEPENDS_ON,
            )
        ],
    )
    report = evaluate_capability_graph_compatibility(previous=previous, current=current)
    assert any(issue.severity == CompatibilitySeverity.INFO for issue in report.issues)
