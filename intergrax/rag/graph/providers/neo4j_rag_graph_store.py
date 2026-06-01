# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

"""Neo4j-backed RAG GraphStore — adapts Integration Library graph_store to GraphRAG contract."""

from __future__ import annotations

from typing import Any, Dict, List, Set

from intergrax.rag.graph.contracts.graph_store import GraphEdge, GraphNode, GraphStore

_ENTITY_LABEL = "RagEntity"
_CHUNK_LABEL = "RagChunk"


class Neo4jRagGraphStore(GraphStore):
    """
    GraphRAG store over Neo4j via the Integration Library ``GraphStore.run_query``.

    Uses ``RagEntity`` / ``RagChunk`` labels and ``RAG_REL`` / ``HAS_CHUNK`` relationships.
    """

    def __init__(self, integration_store: Any) -> None:
        self._store = integration_store

    def _run(self, statement: str, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        result = self._store.run_query(statement, parameters=parameters)
        return list(result.records or [])

    def upsert_node(self, node: GraphNode) -> None:
        self._run(
            f"""
            MERGE (n:{_ENTITY_LABEL} {{id: $id}})
            SET n.label = $label,
                n.node_type = $node_type,
                n.metadata = $metadata
            """,
            {
                "id": node.id,
                "label": node.label,
                "node_type": node.node_type,
                "metadata": dict(node.metadata or {}),
            },
        )

    def upsert_edge(self, edge: GraphEdge) -> None:
        self._run(
            f"""
            MATCH (a:{_ENTITY_LABEL} {{id: $source_id}})
            MATCH (b:{_ENTITY_LABEL} {{id: $target_id}})
            MERGE (a)-[r:RAG_REL {{relation: $relation}}]->(b)
            SET r.weight = $weight,
                r.metadata = $metadata
            """,
            {
                "source_id": edge.source_id,
                "target_id": edge.target_id,
                "relation": edge.relation,
                "weight": float(edge.weight),
                "metadata": dict(edge.metadata or {}),
            },
        )

    def link_chunk(self, node_id: str, chunk_id: str) -> None:
        self._run(
            f"""
            MATCH (n:{_ENTITY_LABEL} {{id: $node_id}})
            MERGE (c:{_CHUNK_LABEL} {{id: $chunk_id}})
            MERGE (n)-[:HAS_CHUNK]->(c)
            """,
            {"node_id": node_id, "chunk_id": chunk_id},
        )

    def neighbors(self, node_id: str, *, max_hops: int = 1) -> List[GraphNode]:
        hops = max(1, int(max_hops))
        rows = self._run(
            f"""
            MATCH (n:{_ENTITY_LABEL} {{id: $node_id}})-[*1..{hops}]-(m:{_ENTITY_LABEL})
            WHERE m.id <> $node_id
            RETURN DISTINCT m.id AS id, m.label AS label, m.node_type AS node_type, m.metadata AS metadata
            """,
            {"node_id": node_id},
        )
        return [_row_to_node(row) for row in rows]

    def find_nodes(self, *, label_contains: str, limit: int = 20) -> List[GraphNode]:
        rows = self._run(
            f"""
            MATCH (n:{_ENTITY_LABEL})
            WHERE toLower(n.label) CONTAINS toLower($needle)
            RETURN n.id AS id, n.label AS label, n.node_type AS node_type, n.metadata AS metadata
            LIMIT $limit
            """,
            {"needle": label_contains or "", "limit": int(limit)},
        )
        return [_row_to_node(row) for row in rows]

    def chunk_ids_for_nodes(self, node_ids: Set[str]) -> List[str]:
        if not node_ids:
            return []
        rows = self._run(
            f"""
            MATCH (n:{_ENTITY_LABEL})-[:HAS_CHUNK]->(c:{_CHUNK_LABEL})
            WHERE n.id IN $node_ids
            RETURN DISTINCT c.id AS chunk_id
            """,
            {"node_ids": sorted(node_ids)},
        )
        return [str(row["chunk_id"]) for row in rows if row.get("chunk_id")]


def _row_to_node(row: Dict[str, Any]) -> GraphNode:
    return GraphNode(
        id=str(row.get("id", "")),
        label=str(row.get("label", "")),
        node_type=str(row.get("node_type") or "entity"),
        metadata=dict(row.get("metadata") or {}),
    )
