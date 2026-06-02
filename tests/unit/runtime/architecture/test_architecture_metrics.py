from __future__ import annotations

from intergrax.runtime.architecture.architecture_metrics import compute_architecture_metrics
from intergrax.runtime.architecture.capability_graph import (
    CapabilityEdge,
    CapabilityEdgeType,
    CapabilityGraph,
    CapabilityNode,
    CapabilityNodeType,
)


def test_metrics_report_for_empty_graph() -> None:
    report = compute_architecture_metrics(CapabilityGraph(nodes=[], edges=[]))
    assert report.mode == "report-only"
    assert report.summary.nodes_total == 0
    assert report.summary.architecture_debt_index == 1.0


def test_metrics_report_for_valid_graph() -> None:
    graph = CapabilityGraph(
        nodes=[
            CapabilityNode(node_id="integration:sqlite", node_type=CapabilityNodeType.INTEGRATION),
            CapabilityNode(node_id="tool:rag.retrieve", node_type=CapabilityNodeType.TOOL),
            CapabilityNode(node_id="skill:research.literature_scan", node_type=CapabilityNodeType.SKILL),
            CapabilityNode(node_id="agent:research", node_type=CapabilityNodeType.AGENT),
            CapabilityNode(node_id="policy:runtime_policy_bundle", node_type=CapabilityNodeType.POLICY),
            CapabilityNode(node_id="evaluation:runtime_quality", node_type=CapabilityNodeType.EVALUATION),
        ],
        edges=[
            CapabilityEdge(
                source_node_id="tool:rag.retrieve",
                target_node_id="integration:sqlite",
                edge_type=CapabilityEdgeType.DEPENDS_ON,
            ),
            CapabilityEdge(
                source_node_id="skill:research.literature_scan",
                target_node_id="tool:rag.retrieve",
                edge_type=CapabilityEdgeType.DEPENDS_ON,
            ),
            CapabilityEdge(
                source_node_id="agent:research",
                target_node_id="skill:research.literature_scan",
                edge_type=CapabilityEdgeType.DEPENDS_ON,
            ),
            CapabilityEdge(
                source_node_id="agent:research",
                target_node_id="policy:runtime_policy_bundle",
                edge_type=CapabilityEdgeType.CONSTRAINED_BY,
            ),
            CapabilityEdge(
                source_node_id="evaluation:runtime_quality",
                target_node_id="agent:research",
                edge_type=CapabilityEdgeType.EVALUATES,
            ),
        ],
    )
    report = compute_architecture_metrics(graph)
    assert report.summary.nodes_total == 6
    assert report.summary.edges_total == 5
    assert 0.0 <= report.summary.modularity_score <= 1.0
    assert 0.0 <= report.summary.dependency_health_score <= 1.0
    assert 0.0 <= report.summary.architecture_debt_index <= 1.0
