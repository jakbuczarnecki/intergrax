from __future__ import annotations

from intergrax.runtime.architecture.architecture_coverage import compute_architecture_coverage
from intergrax.runtime.architecture.capability_graph import (
    CapabilityEdge,
    CapabilityEdgeType,
    CapabilityGraph,
    CapabilityNode,
    CapabilityNodeType,
)


def test_architecture_coverage_reports_governance_and_observability() -> None:
    graph = CapabilityGraph(
        nodes=[
            CapabilityNode(node_id="agent:research", node_type=CapabilityNodeType.AGENT),
            CapabilityNode(node_id="policy:runtime", node_type=CapabilityNodeType.POLICY),
            CapabilityNode(node_id="evaluation:runtime", node_type=CapabilityNodeType.EVALUATION),
        ],
        edges=[
            CapabilityEdge(
                source_node_id="agent:research",
                target_node_id="policy:runtime",
                edge_type=CapabilityEdgeType.CONSTRAINED_BY,
            ),
            CapabilityEdge(
                source_node_id="evaluation:runtime",
                target_node_id="agent:research",
                edge_type=CapabilityEdgeType.EVALUATES,
            ),
        ],
    )
    report = compute_architecture_coverage(graph)
    assert report.summary.nodes_total == 3
    assert 0.0 <= report.summary.governance_coverage <= 1.0
    assert 0.0 <= report.summary.observability_coverage <= 1.0
