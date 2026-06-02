from __future__ import annotations

from intergrax.runtime.architecture.graph_provenance import build_graph_provenance_trace
from intergrax.runtime.architecture.graph_rag import (
    GraphRagEdge,
    GraphRagEdgeType,
    GraphRagNode,
    GraphRagNodeType,
)


def test_graph_provenance_trace_builds_path_records() -> None:
    nodes = [
        GraphRagNode(node_id="doc-1", node_type=GraphRagNodeType.DOCUMENT, label="Doc"),
        GraphRagNode(node_id="ent-1", node_type=GraphRagNodeType.ENTITY, label="Entity"),
    ]
    edges = [
        GraphRagEdge(
            source_node_id="doc-1",
            target_node_id="ent-1",
            edge_type=GraphRagEdgeType.DERIVED_FROM,
        )
    ]
    bundle = build_graph_provenance_trace(
        trace_id="t1",
        graph_id="g1",
        nodes=nodes,
        edges=edges,
        target_node_id="ent-1",
    )
    assert bundle.provenance_records
    assert "doc-1" in bundle.explainability_summary
