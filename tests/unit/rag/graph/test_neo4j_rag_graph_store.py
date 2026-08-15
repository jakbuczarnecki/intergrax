# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

import pytest

from intergrax.rag.graph.contracts.graph_store import GraphEdge, GraphNode
from intergrax.rag.graph.providers.neo4j_rag_graph_store import Neo4jRagGraphStore

pytestmark = pytest.mark.unit


class _FakeIntegrationGraphStore:
    def __init__(self) -> None:
        self._nodes: Dict[str, GraphNode] = {}
        self._edges: List[GraphEdge] = []
        self._chunks: Dict[str, Set[str]] = {}

    def run_query(self, statement: str, *, parameters: Optional[Dict[str, Any]] = None) -> Any:
        params = dict(parameters or {})
        stmt = statement.strip().lower()
        if "merge (n:ragentity" in stmt and "set n.label" in stmt:
            node = GraphNode(
                id=str(params["id"]),
                label=str(params["label"]),
                node_type=str(params.get("node_type", "entity")),
                metadata=dict(params.get("metadata") or {}),
            )
            self._nodes[node.id] = node
            return type("R", (), {"records": []})()
        if "match (a:ragentity" in stmt and "merge (a)-[r:rag_rel" in stmt:
            self._edges.append(
                GraphEdge(
                    source_id=str(params["source_id"]),
                    target_id=str(params["target_id"]),
                    relation=str(params.get("relation", "related_to")),
                    weight=float(params.get("weight", 1.0)),
                )
            )
            return type("R", (), {"records": []})()
        if "merge (a)-[r:rag_rel" in stmt:
            self._edges.append(
                GraphEdge(
                    source_id=str(params["source_id"]),
                    target_id=str(params["target_id"]),
                    relation=str(params.get("relation", "related_to")),
                    weight=float(params.get("weight", 1.0)),
                )
            )
            return type("R", (), {"records": []})()
        if "merge (e:ragevidence" in stmt:
            if "node_id" in params:
                self._chunks.setdefault(str(params["node_id"]), set()).add(
                    str(params["chunk_id"])
                )
            return type("R", (), {"records": []})()
        if "merge (c:ragchunk" in stmt:
            nid = str(params["node_id"])
            cid = str(params["chunk_id"])
            self._chunks.setdefault(nid, set()).add(cid)
            return type("R", (), {"records": []})()
        if "contains tolower($needle)" in stmt:
            needle = str(params.get("needle", "")).lower()
            found = [
                {
                    "id": n.id,
                    "label": n.label,
                    "node_type": n.node_type,
                    "metadata": n.metadata,
                }
                for n in self._nodes.values()
                if needle in n.label.lower()
            ][: int(params.get("limit", 20))]
            return type("R", (), {"records": found})()
        if "match p=" in stmt or "return distinct m.id" in stmt:
            start = str(params["node_id"])
            out = []
            for edge in self._edges:
                nxt = None
                if edge.source_id == start:
                    nxt = edge.target_id
                elif edge.target_id == start:
                    nxt = edge.source_id
                if nxt and nxt in self._nodes:
                    n = self._nodes[nxt]
                    out.append(
                        {
                            "id": n.id,
                            "label": n.label,
                            "node_type": n.node_type,
                            "metadata": n.metadata,
                        }
                    )
            return type("R", (), {"records": out})()
        if "evidences_node" in stmt and "return distinct c.id" in stmt:
            ids = set(params.get("node_ids") or [])
            chunk_ids: List[str] = []
            for nid, chunks in self._chunks.items():
                if nid in ids:
                    chunk_ids.extend(sorted(chunks))
            return type("R", (), {"records": [{"chunk_id": c} for c in chunk_ids]})()
        return type("R", (), {"records": []})()


def test_neo4j_rag_graph_store_neighbors_and_chunks() -> None:
    integration = _FakeIntegrationGraphStore()
    store = Neo4jRagGraphStore(integration)
    store.upsert_node(GraphNode(id="ent:a", label="Alpha", node_type="entity"))
    store.upsert_node(GraphNode(id="ent:b", label="Beta", node_type="entity"))
    store.upsert_edge(
        GraphEdge(
            source_id="ent:a",
            target_id="ent:b",
            relation="related_to",
            metadata={"source_id": "source-a", "chunk_ids": ["chunk-1"]},
        )
    )
    store.link_chunk("ent:a", "chunk-1")

    found = store.find_nodes(label_contains="Alpha", limit=5)
    assert found and found[0].id == "ent:a"
    neighbors = store.neighbors("ent:a", max_hops=1)
    assert any(n.id == "ent:b" for n in neighbors)
    assert store.chunk_ids_for_nodes({"ent:a"}) == ["chunk-1"]


def test_create_rag_graph_store_inmemory_default() -> None:
    from intergrax.rag.graph.bootstrap.graph_store_bootstrap import create_rag_graph_store
    from intergrax.rag.graph.providers.inmemory_graph_store import InMemoryGraphStore

    store = create_rag_graph_store()
    assert isinstance(store, InMemoryGraphStore)
