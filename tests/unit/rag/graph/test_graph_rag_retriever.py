# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.integrations.providers.vector_store.inmemory.rag_store import (
    InMemoryVectorStore,
)
from intergrax.rag.graph.indexer.heuristic_graph_indexer import HeuristicGraphIndexer
from intergrax.rag.graph.providers.inmemory_graph_store import InMemoryGraphStore
from intergrax.rag.retrievers.contracts.base_retriever import (
    RetrievalHit,
    RetrieverQuery,
)
from intergrax.rag.retrievers.providers.graph_rag_retriever import GraphRagRetriever
from intergrax.rag.vectorstore.vectorstore_manager import VectorstoreManager
from tests.unit.rag.graph.fixtures import knowledge_document

pytestmark = pytest.mark.unit


class _Emb:
    def embed_one(self, text: str):
        return [0.1, 0.2, 0.3]


def test_graph_rag_retriever_returns_seeded_chunks() -> None:
    store = InMemoryVectorStore(tenant_id="g1")
    manager = VectorstoreManager(store=store)
    doc = knowledge_document(
        "Intergrax Harness connects Legal Agent and Research Agent.",
        tenant_id="g1",
    )
    manager.add_documents([doc], [[0.1, 0.2, 0.3]], ids=["chunk-1"])

    graph = InMemoryGraphStore()
    HeuristicGraphIndexer(graph).index_documents([doc], chunk_ids=["chunk-1"])

    retriever = GraphRagRetriever(manager, _Emb(), graph, seed_top_k=1)
    hits = retriever.retrieve(
        RetrieverQuery(query_text="Intergrax Agent", query_embedding=None, top_k=3)
    )
    assert hits
    assert all(isinstance(hit, RetrievalHit) for hit in hits)
    assert all(hit.document.scope.tenant_id == "g1" for hit in hits)
    assert all(hit.channel in {"graph", "hybrid"} for hit in hits)
    assert any("Intergrax" in h.content for h in hits)
