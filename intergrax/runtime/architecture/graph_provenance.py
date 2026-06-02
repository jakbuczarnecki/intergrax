# © Artur Czarnecki. All rights reserved.

"""Graph-backed explainability trace field contracts (Phase V-KG.3)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from intergrax.runtime.architecture.graph_rag import GraphRagEdge, GraphRagNode


class GraphProvenanceRecord(BaseModel):
    node_id: str
    edge_path: list[str] = Field(default_factory=list)
    explanation: str


class GraphTraceFieldBundle(BaseModel):
    schema_version: str = "1.0.0"
    trace_id: str
    graph_id: str
    provenance_records: list[GraphProvenanceRecord] = Field(default_factory=list)
    explainability_summary: str = ""


def build_graph_provenance_trace(
    *,
    trace_id: str,
    graph_id: str,
    nodes: list[GraphRagNode],
    edges: list[GraphRagEdge],
    target_node_id: str,
) -> GraphTraceFieldBundle:
    node_by_id = {node.node_id: node for node in nodes}
    incoming = {edge.target_node_id: edge for edge in edges}
    records: list[GraphProvenanceRecord] = []
    current_id = target_node_id
    path: list[str] = [current_id]
    while current_id in incoming:
        edge = incoming[current_id]
        path.insert(0, edge.source_node_id)
        current_id = edge.source_node_id
    for node_id in path:
        node = node_by_id[node_id]
        records.append(
            GraphProvenanceRecord(
                node_id=node.node_id,
                edge_path=path,
                explanation=f"Traversed {node.node_type.value} node `{node.label}`",
            )
        )
    return GraphTraceFieldBundle(
        trace_id=trace_id,
        graph_id=graph_id,
        provenance_records=records,
        explainability_summary=f"Graph provenance path: {' -> '.join(path)}",
    )
