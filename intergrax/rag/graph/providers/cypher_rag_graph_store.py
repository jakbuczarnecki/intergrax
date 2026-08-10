# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Cypher/Bolt-backed source-scoped GraphRAG store."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Sequence, Set

from intergrax.distributed.source_operation import (
    SOURCE_PUBLICATION_GENERATION_METADATA_KEY,
)
from intergrax.rag.graph.contracts.graph_store import (
    GraphEdge,
    GraphNode,
    GraphScope,
    GraphStore,
)

_ENTITY_LABEL = "RagEntity"
_CHUNK_LABEL = "RagChunk"


class CypherRagGraphStore(GraphStore):
    """GraphRAG store using scope-local entities, chunks and evidence nodes."""

    def __init__(
        self,
        integration_store: Any,
        *,
        tenant_id: str | None = None,
        namespace: str | None = None,
        workspace_id: str | None = None,
    ) -> None:
        self._store = integration_store
        self._tenant_id = tenant_id.strip() if tenant_id else None
        self._bound_scope: GraphScope | None = None
        self._node_metadata_cache: Dict[tuple[str, str], Dict[str, Any]] = {}
        if namespace is not None or workspace_id is not None:
            if self._tenant_id is None:
                raise ValueError("namespace/workspace require tenant_id")
            self._bound_scope = GraphScope(
                self._tenant_id, namespace=namespace, workspace_id=workspace_id
            )

    @property
    def tenant_id(self) -> str | None:
        return self._tenant_id

    @property
    def scope(self) -> GraphScope | None:
        return self._bound_scope

    def bind_scope(self, scope: GraphScope) -> None:
        normalized = GraphScope.from_object(scope)
        if self._tenant_id is not None and normalized.tenant_id != self._tenant_id:
            raise ValueError("document tenant_id differs from bound graph store")
        if self._bound_scope is not None and self._bound_scope != normalized:
            raise ValueError("graph scope cannot change after binding")
        self._bound_scope = normalized

    def _run(self, statement: str, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        result = self._store.run_query(statement, parameters=parameters)
        records = getattr(result, "records", None)
        if records is None:
            raise RuntimeError("graph query returned malformed result")
        return [dict(record) for record in records]

    def _scope_for_metadata(
        self,
        metadata: Dict[str, Any] | None = None,
        scope: GraphScope | None = None,
    ) -> GraphScope:
        values = metadata or {}
        if self._bound_scope is not None:
            resolved = self._bound_scope
            for name in ("tenant_id", "namespace", "workspace_id"):
                supplied = values.get(name)
                expected = getattr(resolved, name)
                if supplied is not None and supplied != expected:
                    raise ValueError(f"graph {name} differs from bound scope")
        elif scope is not None:
            resolved = GraphScope.from_object(scope)
        else:
            resolved = GraphScope(
                str(values.get("tenant_id") or self._tenant_id or "__legacy__"),
                namespace=values.get("namespace")
                if isinstance(values.get("namespace"), str)
                else None,
                workspace_id=values.get("workspace_id")
                if isinstance(values.get("workspace_id"), str)
                else None,
            )
        self.bind_scope(resolved)
        return resolved

    @staticmethod
    def _scope_values(scope: GraphScope) -> Dict[str, str]:
        return {
            "scope_key": scope.key,
            "tenant_id": scope.tenant_id,
            "namespace": scope.namespace or "",
            "workspace_id": scope.workspace_id or "",
        }

    @staticmethod
    def _metadata(metadata: Dict[str, Any] | None, scope: GraphScope) -> Dict[str, Any]:
        result = dict(metadata or {})
        result.update(
            {
                "tenant_id": scope.tenant_id,
                "namespace": scope.namespace,
                "workspace_id": scope.workspace_id,
            }
        )
        return result

    def upsert_node(self, node: GraphNode) -> None:
        scope = self._scope_for_metadata(node.metadata)
        metadata = self._metadata(node.metadata, scope)
        self._node_metadata_cache[(scope.key, node.id)] = metadata
        self._run(
            f"""
            MERGE (n:{_ENTITY_LABEL} {{scope_key: $scope_key, id: $id}})
            SET n.label = $label,
                n.node_type = $node_type,
                n.tenant_id = $tenant_id,
                n.namespace = $namespace,
                n.workspace_id = $workspace_id,
                n.metadata_json = $metadata_json
            """,
            {
                "id": node.id,
                "label": node.label,
                "node_type": node.node_type,
                "metadata": metadata,
                "metadata_json": json.dumps(metadata, default=str, sort_keys=True),
                **self._scope_values(scope),
            },
        )

    def upsert_edge(self, edge: GraphEdge) -> None:
        metadata = dict(edge.metadata or {})
        scope = self._scope_for_metadata(metadata)
        edge_key = self._edge_key(edge, scope)
        params = {
            "source_id": edge.source_id,
            "target_id": edge.target_id,
            "relation": edge.relation,
            "weight": float(edge.weight),
            "metadata": metadata,
            "metadata_json": json.dumps(metadata, default=str, sort_keys=True),
            "edge_key": edge_key,
            **self._scope_values(scope),
        }
        self._run(
            f"""
            MATCH (a:{_ENTITY_LABEL} {{scope_key: $scope_key, id: $source_id}})
            MATCH (b:{_ENTITY_LABEL} {{scope_key: $scope_key, id: $target_id}})
            MERGE (a)-[r:RAG_REL {{scope_key: $scope_key, edge_key: $edge_key}}]->(b)
            SET r.relation = $relation, r.weight = $weight,
                r.metadata_json = $metadata_json
            """,
            params,
        )
        raw_chunks = metadata.get("chunk_ids")
        if not metadata.get("source_id") and not isinstance(
            raw_chunks, (list, tuple, set)
        ):
            return
        chunk_ids = (
            [str(item).strip() for item in raw_chunks if str(item).strip()]
            if isinstance(raw_chunks, (list, tuple, set))
            else [""]
        )
        source_id = str(metadata.get("source_id") or "__legacy__")
        self._run(
            f"""
            MATCH (a:{_ENTITY_LABEL} {{scope_key: $scope_key, id: $entity_source_id}})
            MATCH (b:{_ENTITY_LABEL} {{scope_key: $scope_key, id: $target_id}})
            UNWIND $chunk_ids AS chunk_id
            MERGE (c:{_CHUNK_LABEL} {{scope_key: $scope_key, id: chunk_id}})
            MERGE (e:RagEvidence {{
                evidence_key: $evidence_prefix + chunk_id,
                scope_key: $scope_key,
                source_id: $source_id,
                chunk_id: chunk_id,
                edge_key: $edge_key
            }})
            SET e.tenant_id = $tenant_id, e.namespace = $namespace,
                e.workspace_id = $workspace_id, e.generation = $generation
            MERGE (e)-[:SUPPORTS_FROM]->(a)
            MERGE (e)-[:SUPPORTS_TO]->(b)
            MERGE (e)-[:EVIDENCES_CHUNK]->(c)
            """,
            {
                **params,
                "entity_source_id": edge.source_id,
                "source_id": source_id,
                "chunk_ids": chunk_ids,
                "evidence_prefix": (
                    f"{scope.key}|edge|{source_id}|{edge_key}|"
                ),
                "generation": metadata.get(SOURCE_PUBLICATION_GENERATION_METADATA_KEY),
            },
        )

    @staticmethod
    def _edge_key(edge: GraphEdge, scope: GraphScope) -> str:
        return f"{scope.key}|{edge.source_id}|{edge.relation}|{edge.target_id}"

    def link_chunk(self, node_id: str, chunk_id: str) -> None:
        normalized_chunk_id = str(chunk_id).strip()
        if not normalized_chunk_id:
            return
        scope = self._scope_for_metadata()
        metadata = self._node_metadata_cache.get((scope.key, node_id))
        if metadata is None:
            rows = self._run(
                f"""
                MATCH (n:{_ENTITY_LABEL} {{scope_key: $scope_key, id: $node_id}})
                RETURN n.metadata_json AS metadata_json
                """,
                {"scope_key": scope.key, "node_id": node_id},
            )
            if not rows or not isinstance(rows[0].get("metadata_json"), str):
                raise LookupError(f"graph node not found in scope: {node_id}")
            try:
                metadata = json.loads(rows[0]["metadata_json"])
            except (TypeError, ValueError) as exc:
                raise RuntimeError("graph node metadata is malformed") from exc
            if not isinstance(metadata, dict):
                raise RuntimeError("graph node metadata is malformed")
        source_id = str(metadata.get("source_id") or "__legacy__")
        self._run(
            f"""
            MATCH (n:{_ENTITY_LABEL} {{scope_key: $scope_key, id: $node_id}})
            MERGE (c:{_CHUNK_LABEL} {{scope_key: $scope_key, id: $chunk_id}})
            SET c.tenant_id = $tenant_id, c.namespace = $namespace,
                c.workspace_id = $workspace_id
            MERGE (e:RagEvidence {{
                evidence_key: $evidence_key, scope_key: $scope_key,
                source_id: $source_id, chunk_id: $chunk_id, edge_key: ""
            }})
            SET e.tenant_id = $tenant_id, e.namespace = $namespace,
                e.workspace_id = $workspace_id, e.generation = $generation
            MERGE (e)-[:EVIDENCES_NODE]->(n)
            MERGE (e)-[:EVIDENCES_CHUNK]->(c)
            """,
            {
                "node_id": node_id,
                "chunk_id": normalized_chunk_id,
                "source_id": source_id,
                "evidence_key": (
                    f"{scope.key}|node|{source_id}|{node_id}|{normalized_chunk_id}"
                ),
                "generation": metadata.get(SOURCE_PUBLICATION_GENERATION_METADATA_KEY),
                **self._scope_values(scope),
            },
        )

    def _read_scope(self) -> GraphScope:
        return self._scope_for_metadata()

    def neighbors(self, node_id: str, *, max_hops: int = 1) -> List[GraphNode]:
        scope = self._read_scope()
        hops = max(1, int(max_hops))
        rows = self._run(
            f"""
            MATCH (n:{_ENTITY_LABEL} {{scope_key: $scope_key, id: $node_id}})
            MATCH p=(n)-[:RAG_REL*1..{hops}]-(m:{_ENTITY_LABEL} {{scope_key: $scope_key}})
            WHERE m.id <> $node_id
              AND ALL(rel IN relationships(p) WHERE EXISTS {{
                  MATCH (e:RagEvidence {{scope_key: $scope_key}})
                        -[:SUPPORTS_FROM]->(a:{_ENTITY_LABEL})
                  MATCH (e)-[:SUPPORTS_TO]->(b:{_ENTITY_LABEL})
                  WHERE e.edge_key = rel.edge_key
              }})
            RETURN DISTINCT m.id AS id, m.label AS label,
                   m.node_type AS node_type, m.metadata_json AS metadata_json
            """,
            {"scope_key": scope.key, "node_id": node_id},
        )
        return [_row_to_node(row) for row in rows]

    def find_nodes(self, *, label_contains: str, limit: int = 20) -> List[GraphNode]:
        scope = self._read_scope()
        rows = self._run(
            f"""
            MATCH (n:{_ENTITY_LABEL} {{scope_key: $scope_key}})
            WHERE toLower(n.label) CONTAINS toLower($needle)
            RETURN n.id AS id, n.label AS label, n.node_type AS node_type,
                   n.metadata_json AS metadata_json
            LIMIT $limit
            """,
            {"needle": label_contains or "", "limit": int(limit), "scope_key": scope.key},
        )
        return [_row_to_node(row) for row in rows]

    def chunk_ids_for_nodes(self, node_ids: Set[str]) -> List[str]:
        if not node_ids:
            return []
        scope = self._read_scope()
        rows = self._run(
            f"""
            MATCH (e:RagEvidence {{scope_key: $scope_key}})-[:EVIDENCES_NODE]->(n:{_ENTITY_LABEL})
            MATCH (e)-[:EVIDENCES_CHUNK]->(c:{_CHUNK_LABEL})
            WHERE n.id IN $node_ids
            RETURN DISTINCT c.id AS chunk_id
            """,
            {"node_ids": sorted(node_ids), "scope_key": scope.key},
        )
        return [str(row["chunk_id"]) for row in rows if row.get("chunk_id")]

    def node_ids_for_chunks(self, chunk_ids: Set[str]) -> Set[str]:
        ids = [cid.strip() for cid in chunk_ids if cid.strip()]
        if not ids:
            return set()
        scope = self._read_scope()
        rows = self._run(
            f"""
            MATCH (e:RagEvidence {{scope_key: $scope_key}})-[:EVIDENCES_NODE]->(n:{_ENTITY_LABEL})
            MATCH (e)-[:EVIDENCES_CHUNK]->(c:{_CHUNK_LABEL})
            WHERE c.id IN $chunk_ids
            RETURN DISTINCT n.id AS node_id
            """,
            {"chunk_ids": ids, "scope_key": scope.key},
        )
        return {str(row["node_id"]) for row in rows if row.get("node_id")}

    def unlink_chunks(self, chunk_ids: Sequence[str]) -> int:
        ids = [cid.strip() for cid in chunk_ids if cid.strip()]
        if not ids:
            return 0
        scope = self._read_scope()
        rows = self._run(
            """
            MATCH (e:RagEvidence {scope_key: $scope_key})-[:EVIDENCES_CHUNK]->(c:RagChunk)
            WHERE c.id IN $chunk_ids
            DETACH DELETE e
            RETURN count(e) AS removed
            """,
            {"chunk_ids": ids, "scope_key": scope.key},
        )
        return self._cleanup(scope, _counter(rows, "removed"))

    def unlink_source(self, source_id: str, *, scope: GraphScope | None = None) -> int:
        source_id = source_id.strip()
        if not source_id:
            return 0
        resolved = self._scope_for_metadata(scope=scope)
        rows = self._run(
            """
            MATCH (e:RagEvidence {scope_key: $scope_key, source_id: $source_id})
            DETACH DELETE e
            RETURN count(e) AS removed
            """,
            {"scope_key": resolved.key, "source_id": source_id},
        )
        return self._cleanup(resolved, _counter(rows, "removed"))

    def _cleanup(self, scope: GraphScope, removed: int) -> int:
        params = {"scope_key": scope.key}
        relation_rows = self._run(
            """
            MATCH (a:RagEntity {scope_key: $scope_key})
                  -[r:RAG_REL {scope_key: $scope_key}]->(b:RagEntity)
            WHERE NOT EXISTS {
                MATCH (e:RagEvidence {scope_key: $scope_key})
                WHERE e.edge_key = r.edge_key
            }
            DELETE r
            RETURN count(r) AS removed
            """,
            params,
        )
        entity_rows = self._run(
            """
            MATCH (n:RagEntity {scope_key: $scope_key})
            WHERE NOT EXISTS {
                MATCH (:RagEvidence {scope_key: $scope_key})-[:EVIDENCES_NODE]->(n)
            }
            AND NOT (n)-[:RAG_REL {scope_key: $scope_key}]-()
            DETACH DELETE n
            RETURN count(n) AS removed
            """,
            params,
        )
        chunk_rows = self._run(
            """
            MATCH (c:RagChunk {scope_key: $scope_key})
            WHERE NOT EXISTS {
                MATCH (:RagEvidence {scope_key: $scope_key})-[:EVIDENCES_CHUNK]->(c)
            }
            DETACH DELETE c
            RETURN count(c) AS removed
            """,
            params,
        )
        return removed + sum(
            _counter(rows, "removed")
            for rows in (relation_rows, entity_rows, chunk_rows)
        )

    def purge_graph(self, *, tenant_id: str | None = None) -> int:
        if tenant_id is None and self._bound_scope is not None:
            entity_where = "n.scope_key = $scope_key"
            chunk_where = "c.scope_key = $scope_key"
            params: Dict[str, Any] = {"scope_key": self._bound_scope.key}
        else:
            scope = (tenant_id or self._tenant_id or "").strip()
            entity_where = "n.tenant_id = $tenant_id" if scope else "true"
            chunk_where = "c.tenant_id = $tenant_id" if scope else "true"
            params = {"tenant_id": scope} if scope else {}
        entity_rows = self._run(
            f"MATCH (n:{_ENTITY_LABEL}) WHERE {entity_where} DETACH DELETE n "
            "RETURN count(n) AS removed",
            params,
        )
        chunk_rows = self._run(
            f"MATCH (c:{_CHUNK_LABEL}) WHERE {chunk_where} DETACH DELETE c "
            "RETURN count(c) AS removed",
            params,
        )
        return _counter(entity_rows, "removed") + _counter(chunk_rows, "removed")


def _row_to_node(row: Dict[str, Any]) -> GraphNode:
    raw_metadata = row.get("metadata")
    if raw_metadata is None and isinstance(row.get("metadata_json"), str):
        try:
            raw_metadata = json.loads(row["metadata_json"])
        except (TypeError, ValueError) as exc:
            raise RuntimeError("graph node metadata is malformed") from exc
    if raw_metadata is not None and not isinstance(raw_metadata, dict):
        raise RuntimeError("graph node metadata is malformed")
    return GraphNode(
        id=str(row.get("id", "")),
        label=str(row.get("label", "")),
        node_type=str(row.get("node_type") or "entity"),
        metadata=dict(raw_metadata or {}),
    )


def _counter(rows: List[Dict[str, Any]], key: str) -> int:
    if not rows:
        return 0
    value = rows[0].get(key)
    if value is None:
        raise RuntimeError(f"graph mutation result missing {key}")
    return int(value)
