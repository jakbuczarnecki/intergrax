from __future__ import annotations

from intergrax.runtime.architecture.capability_graph import (
    CapabilityEdge,
    CapabilityEdgeType,
    CapabilityGraph,
    CapabilityNode,
    CapabilityNodeType,
)
from intergrax.runtime.architecture.capability_graph_lineage import (
    build_capability_impact_report,
    build_capability_lineage_report,
)


def _sample_graph() -> CapabilityGraph:
    return CapabilityGraph(
        nodes=[
            CapabilityNode(node_id="integration:sqlite", node_type=CapabilityNodeType.INTEGRATION),
            CapabilityNode(node_id="tool:rag.retrieve", node_type=CapabilityNodeType.TOOL),
            CapabilityNode(node_id="skill:research.scan", node_type=CapabilityNodeType.SKILL),
            CapabilityNode(node_id="agent:research", node_type=CapabilityNodeType.AGENT),
            CapabilityNode(node_id="application:lab", node_type=CapabilityNodeType.APPLICATION),
            CapabilityNode(node_id="product:intergrax_harness", node_type=CapabilityNodeType.PRODUCT),
        ],
        edges=[
            CapabilityEdge(
                source_node_id="tool:rag.retrieve",
                target_node_id="integration:sqlite",
                edge_type=CapabilityEdgeType.DEPENDS_ON,
            ),
            CapabilityEdge(
                source_node_id="skill:research.scan",
                target_node_id="tool:rag.retrieve",
                edge_type=CapabilityEdgeType.DEPENDS_ON,
            ),
            CapabilityEdge(
                source_node_id="agent:research",
                target_node_id="skill:research.scan",
                edge_type=CapabilityEdgeType.DEPENDS_ON,
            ),
            CapabilityEdge(
                source_node_id="application:lab",
                target_node_id="agent:research",
                edge_type=CapabilityEdgeType.DEPENDS_ON,
            ),
            CapabilityEdge(
                source_node_id="product:intergrax_harness",
                target_node_id="application:lab",
                edge_type=CapabilityEdgeType.DEPENDS_ON,
            ),
        ],
    )


def test_build_capability_lineage_report_includes_upstream_and_downstream() -> None:
    report = build_capability_lineage_report(_sample_graph())
    record_by_node = {record.node_id: record for record in report.records}
    assert "tool:rag.retrieve" in record_by_node
    assert "integration:sqlite" in record_by_node["tool:rag.retrieve"].downstream_node_ids
    assert "agent:research" in record_by_node["skill:research.scan"].upstream_node_ids


def test_build_capability_impact_report_includes_transitive_blast_radius() -> None:
    report = build_capability_impact_report(_sample_graph())
    record_by_node = {record.node_id: record for record in report.impacts}
    assert "product:intergrax_harness" in record_by_node
    assert "application:lab" in record_by_node["product:intergrax_harness"].blast_radius_node_ids
    assert "agent:research" in record_by_node["product:intergrax_harness"].blast_radius_node_ids
