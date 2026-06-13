# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

"""Structured graph provenance for GraphRagRetriever traces (M-RAG.44, M-RAG.54) — Tier-0 only."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

from intergrax.rag.graph.contracts.graph_store import GraphNode


@dataclass(frozen=True)
class GraphRetrievalProvenanceRecord:
    node_id: str
    edge_path: List[str] = field(default_factory=list)
    explanation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "edge_path": list(self.edge_path),
            "explanation": self.explanation,
        }


@dataclass(frozen=True)
class GraphRetrievalProvenanceBundle:
    trace_id: str
    graph_id: str
    provenance_records: List[GraphRetrievalProvenanceRecord] = field(default_factory=list)
    explainability_summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": "1.0.0",
            "trace_id": self.trace_id,
            "graph_id": self.graph_id,
            "provenance_records": [record.to_dict() for record in self.provenance_records],
            "explainability_summary": self.explainability_summary,
        }


def build_graph_retrieval_provenance(
    *,
    trace_id: str,
    graph_id: str,
    seed_node_ids: Sequence[str],
    expanded_nodes: Sequence[GraphNode],
) -> GraphRetrievalProvenanceBundle:
    """Build structured provenance records aligned with ``GraphTraceFieldBundle`` shape."""
    ordered_ids = [node.id for node in expanded_nodes] or list(seed_node_ids)
    records: List[GraphRetrievalProvenanceRecord] = []
    for node in expanded_nodes:
        path_index = ordered_ids.index(node.id) if node.id in ordered_ids else 0
        edge_path = ordered_ids[: path_index + 1] if ordered_ids else [node.id]
        records.append(
            GraphRetrievalProvenanceRecord(
                node_id=node.id,
                edge_path=edge_path,
                explanation=f"Expanded {node.node_type} node `{node.label}`",
            )
        )
    for seed_id in seed_node_ids:
        if any(record.node_id == seed_id for record in records):
            continue
        records.append(
            GraphRetrievalProvenanceRecord(
                node_id=seed_id,
                edge_path=[seed_id],
                explanation=f"Seed entity `{seed_id.replace('ent:', '').replace('_', ' ')}`",
            )
        )
    summary = (
        f"Graph expansion nodes: {' -> '.join(node.label for node in expanded_nodes[:5])}"
        if expanded_nodes
        else (
            f"Graph seed nodes: {' -> '.join(seed.replace('ent:', '') for seed in seed_node_ids[:5])}"
            if seed_node_ids
            else ""
        )
    )
    return GraphRetrievalProvenanceBundle(
        trace_id=trace_id,
        graph_id=graph_id,
        provenance_records=records,
        explainability_summary=summary,
    )
