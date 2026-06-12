# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Cypher/Bolt-backed RAG GraphStore — shared by neo4j, memgraph, falkordb (M-RAG.39)."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Set

from intergrax.rag.graph.contracts.graph_store import GraphEdge, GraphNode, GraphStore

_ENTITY_LABEL = "RagEntity"
_CHUNK_LABEL = "RagChunk"


class CypherRagGraphStore(GraphStore):
    """
    GraphRAG store over Bolt-compatible integration ``GraphStore.run_query``.

    Uses ``RagEntity`` / ``RagChunk`` labels and ``RAG_REL`` / ``HAS_CHUNK`` relationships.
    """

    def __init__(
        self,
        integration_store: Any,
        *,
        tenant_id: str | None = None,
    ) -> None:
        self._store = integration_store
        self._tenant_id = tenant_id.strip() if tenant_id else None

    @property
    def tenant_id(self) -> str | None:
        return self._tenant_id

    def _run(self, statement: str, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        result = self._store.run_query(statement, parameters=parameters)
        return list(result.records or [])

    def _tenant_clause(self, alias: str = "n") -> str:
        if self._tenant_id is None:
            return ""
        return f" AND {alias}.tenant_id = $tenant_id"

    def _tenant_params(self) -> Dict[str, str]:
        if self._tenant_id is None:
            return {}
        return {"tenant_id": self._tenant_id}

    def upsert_node(self, node: GraphNode) -> None:
        metadata = dict(node.metadata or {})
        if self._tenant_id is not None:
            metadata.setdefault("tenant_id", self._tenant_id)
        params: Dict[str, Any] = {
            "id": node.id,
            "label": node.label,
            "node_type": node.node_type,
            "metadata": metadata,
            **self._tenant_params(),
        }
        tenant_set = ", n.tenant_id = $tenant_id" if self._tenant_id is not None else ""
        self._run(
            f"""
            MERGE (n:{_ENTITY_LABEL} {{id: $id}})
            SET n.label = $label,
                n.node_type = $node_type,
                n.metadata = $metadata{tenant_set}
            """,
            params,
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
        params: Dict[str, Any] = {"node_id": node_id, "chunk_id": chunk_id, **self._tenant_params()}
        tenant_chunk = ", c.tenant_id = $tenant_id" if self._tenant_id is not None else ""
        self._run(
            f"""
            MATCH (n:{_ENTITY_LABEL} {{id: $node_id}})
            MERGE (c:{_CHUNK_LABEL} {{id: $chunk_id}})
            SET c.id = $chunk_id{tenant_chunk}
            MERGE (n)-[:HAS_CHUNK]->(c)
            """,
            params,
        )

    def neighbors(self, node_id: str, *, max_hops: int = 1) -> List[GraphNode]:
        hops = max(1, int(max_hops))
        params: Dict[str, Any] = {"node_id": node_id, **self._tenant_params()}
        tenant = self._tenant_clause("m")
        rows = self._run(
            f"""
            MATCH (n:{_ENTITY_LABEL} {{id: $node_id}})-[*1..{hops}]-(m:{_ENTITY_LABEL})
            WHERE m.id <> $node_id{tenant}
            RETURN DISTINCT m.id AS id, m.label AS label, m.node_type AS node_type, m.metadata AS metadata
            """,
            params,
        )
        return [_row_to_node(row) for row in rows]

    def find_nodes(self, *, label_contains: str, limit: int = 20) -> List[GraphNode]:
        params: Dict[str, Any] = {
            "needle": label_contains or "",
            "limit": int(limit),
            **self._tenant_params(),
        }
        tenant = self._tenant_clause("n")
        rows = self._run(
            f"""
            MATCH (n:{_ENTITY_LABEL})
            WHERE toLower(n.label) CONTAINS toLower($needle){tenant}
            RETURN n.id AS id, n.label AS label, n.node_type AS node_type, n.metadata AS metadata
            LIMIT $limit
            """,
            params,
        )
        return [_row_to_node(row) for row in rows]

    def chunk_ids_for_nodes(self, node_ids: Set[str]) -> List[str]:
        if not node_ids:
            return []
        params: Dict[str, Any] = {"node_ids": sorted(node_ids), **self._tenant_params()}
        tenant = self._tenant_clause("n")
        rows = self._run(
            f"""
            MATCH (n:{_ENTITY_LABEL})-[:HAS_CHUNK]->(c:{_CHUNK_LABEL})
            WHERE n.id IN $node_ids{tenant}
            RETURN DISTINCT c.id AS chunk_id
            """,
            params,
        )
        return [str(row["chunk_id"]) for row in rows if row.get("chunk_id")]

    def node_ids_for_chunks(self, chunk_ids: Set[str]) -> Set[str]:
        ids = [cid.strip() for cid in chunk_ids if cid.strip()]
        if not ids:
            return set()
        params: Dict[str, Any] = {"chunk_ids": ids, **self._tenant_params()}
        tenant = self._tenant_clause("n")
        rows = self._run(
            f"""
            MATCH (n:{_ENTITY_LABEL})-[:HAS_CHUNK]->(c:{_CHUNK_LABEL})
            WHERE c.id IN $chunk_ids{tenant}
            RETURN DISTINCT n.id AS node_id
            """,
            params,
        )
        return {str(row["node_id"]) for row in rows if row.get("node_id")}

    def unlink_chunks(self, chunk_ids: Sequence[str]) -> int:
        ids = [cid.strip() for cid in chunk_ids if cid.strip()]
        if not ids:
            return 0
        params: Dict[str, Any] = {"chunk_ids": ids, **self._tenant_params()}
        tenant_chunk = self._tenant_clause("c")
        tenant_entity = self._tenant_clause("n")
        rows = self._run(
            f"""
            MATCH (c:{_CHUNK_LABEL})
            WHERE c.id IN $chunk_ids{tenant_chunk}
            OPTIONAL MATCH (n:{_ENTITY_LABEL})-[r:HAS_CHUNK]->(c)
            DELETE r
            WITH DISTINCT c
            DETACH DELETE c
            RETURN count(c) AS removed_chunks
            """,
            params,
        )
        removed_chunks = int(rows[0].get("removed_chunks", 0)) if rows else 0
        orphan_rows = self._run(
            f"""
            MATCH (n:{_ENTITY_LABEL})
            WHERE NOT (n)-[:HAS_CHUNK]->(){tenant_entity}
            WITH n
            DETACH DELETE n
            RETURN count(n) AS pruned_entities
            """,
            params,
        )
        pruned = int(orphan_rows[0].get("pruned_entities", 0)) if orphan_rows else 0
        return removed_chunks + pruned

    def purge_graph(self, *, tenant_id: str | None = None) -> int:
        scope = (tenant_id or self._tenant_id or "").strip()
        if scope:
            params = {"tenant_id": scope}
            entity_rows = self._run(
                f"""
                MATCH (n:{_ENTITY_LABEL})
                WHERE n.tenant_id = $tenant_id
                DETACH DELETE n
                RETURN count(n) AS removed
                """,
                params,
            )
            chunk_rows = self._run(
                f"""
                MATCH (c:{_CHUNK_LABEL})
                WHERE c.tenant_id = $tenant_id
                DETACH DELETE c
                RETURN count(c) AS removed
                """,
                params,
            )
        else:
            entity_rows = self._run(
                f"MATCH (n:{_ENTITY_LABEL}) DETACH DELETE n RETURN count(n) AS removed",
                {},
            )
            chunk_rows = self._run(
                f"MATCH (c:{_CHUNK_LABEL}) DETACH DELETE c RETURN count(c) AS removed",
                {},
            )
        removed_entities = int(entity_rows[0].get("removed", 0)) if entity_rows else 0
        removed_chunks = int(chunk_rows[0].get("removed", 0)) if chunk_rows else 0
        return removed_entities + removed_chunks


def _row_to_node(row: Dict[str, Any]) -> GraphNode:
    return GraphNode(
        id=str(row.get("id", "")),
        label=str(row.get("label", "")),
        node_type=str(row.get("node_type") or "entity"),
        metadata=dict(row.get("metadata") or {}),
    )
