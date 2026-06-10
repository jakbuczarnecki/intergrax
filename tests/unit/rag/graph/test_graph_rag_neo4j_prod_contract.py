# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

import pytest
from langchain_core.documents import Document

from intergrax.integrations.providers.vector_store.inmemory.rag_store import InMemoryVectorStore
from intergrax.rag.graph.bootstrap.graph_store_bootstrap import create_rag_graph_store
from intergrax.rag.graph.indexer.heuristic_graph_indexer import HeuristicGraphIndexer
from intergrax.rag.graph.providers.neo4j_rag_graph_store import Neo4jRagGraphStore
from intergrax.rag.profiles.rag_profile import production_graph_rag_profile
from intergrax.rag.retrievers.contracts.base_retriever import RetrieverQuery
from intergrax.rag.retrievers.providers.graph_rag_retriever import GraphRagRetriever
from intergrax.rag.vectorstore.vectorstore_manager import VectorstoreManager

pytestmark = [pytest.mark.unit, pytest.mark.gate]


class _FakeNeo4jIntegrationGraphStore:
    """Minimal durable-graph stand-in for Neo4jRagGraphStore contract tests."""

    def __init__(self) -> None:
        self._nodes: Dict[str, Any] = {}
        self._edges: List[Any] = []
        self._chunks: Dict[str, Set[str]] = {}

    def run_query(self, statement: str, *, parameters: Optional[Dict[str, Any]] = None) -> Any:
        from intergrax.rag.graph.contracts.graph_store import GraphEdge, GraphNode

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
        if "merge (a)-[r:rag_rel" in stmt or (
            "match (a:ragentity" in stmt and "merge (a)-[r:rag_rel" in stmt
        ):
            self._edges.append(
                GraphEdge(
                    source_id=str(params["source_id"]),
                    target_id=str(params["target_id"]),
                    relation=str(params.get("relation", "related_to")),
                    weight=float(params.get("weight", 1.0)),
                )
            )
            return type("R", (), {"records": []})()
        if "merge (c:ragchunk" in stmt:
            self._chunks.setdefault(str(params["node_id"]), set()).add(str(params["chunk_id"]))
            return type("R", (), {"records": []})()
        if "contains tolower($needle)" in stmt:
            needle = str(params.get("needle", "")).lower()
            found = [
                {
                    "id": node.id,
                    "label": node.label,
                    "node_type": node.node_type,
                    "metadata": node.metadata,
                }
                for node in self._nodes.values()
                if needle in node.label.lower()
            ][: int(params.get("limit", 20))]
            return type("R", (), {"records": found})()
        if "-[*1.." in stmt and "return distinct m.id" in stmt:
            start = str(params["node_id"])
            out = []
            for edge in self._edges:
                nxt = edge.target_id if edge.source_id == start else (
                    edge.source_id if edge.target_id == start else None
                )
                if nxt and nxt in self._nodes:
                    node = self._nodes[nxt]
                    out.append(
                        {
                            "id": node.id,
                            "label": node.label,
                            "node_type": node.node_type,
                            "metadata": node.metadata,
                        }
                    )
            return type("R", (), {"records": out})()
        if "has_chunk" in stmt and "return distinct c.id" in stmt:
            ids = set(params.get("node_ids") or [])
            chunk_ids: List[str] = []
            for node_id, chunks in self._chunks.items():
                if node_id in ids:
                    chunk_ids.extend(sorted(chunks))
            return type("R", (), {"records": [{"chunk_id": c} for c in chunk_ids]})()
        return type("R", (), {"records": []})()


class _Emb:
    def embed_one(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]


def test_create_rag_graph_store_uses_neo4j_integration_instance() -> None:
    integration = _FakeNeo4jIntegrationGraphStore()
    store = create_rag_graph_store(
        profile=production_graph_rag_profile(),
        integration_graph_store=integration,
    )
    assert isinstance(store, Neo4jRagGraphStore)


def test_graph_rag_retrieve_through_neo4j_prod_profile() -> None:
    vector = VectorstoreManager(store=InMemoryVectorStore(tenant_id="prod-graph"))
    doc = Document(
        page_content="Intergrax legal corpus references contract clause alpha.",
        metadata={"tenant_id": "prod-graph"},
    )
    vector.add_documents([doc], [[0.1, 0.2, 0.3]], ids=["chunk-legal-1"])

    graph = create_rag_graph_store(
        profile=production_graph_rag_profile(),
        integration_graph_store=_FakeNeo4jIntegrationGraphStore(),
    )
    HeuristicGraphIndexer(graph).index_documents([doc], chunk_ids=["chunk-legal-1"])

    retriever = GraphRagRetriever(vector, _Emb(), graph, seed_top_k=1)
    hits = retriever.retrieve(
        RetrieverQuery(query_text="contract clause", query_embedding=None, top_k=3)
    )

    assert hits
    assert any("contract" in hit.content.lower() for hit in hits)
