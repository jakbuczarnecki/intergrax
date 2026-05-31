# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from intergrax.integrations.providers.vector_store.inmemory.rag_store import InMemoryVectorStore
from intergrax.rag.graph.indexer.heuristic_graph_indexer import HeuristicGraphIndexer
from intergrax.rag.graph.providers.inmemory_graph_store import InMemoryGraphStore
from intergrax.rag.retrievers.providers.graph_rag_retriever import GraphRagRetriever
from intergrax.rag.retrievers.contracts.base_retriever import RetrieverQuery
from intergrax.rag.vectorstore.vectorstore_manager import VectorstoreManager

pytestmark = pytest.mark.unit


class _Emb:
    def embed_one(self, text: str):
        return [0.1, 0.2, 0.3]


def test_graph_rag_retriever_returns_seeded_chunks() -> None:
    store = InMemoryVectorStore(tenant_id="g1")
    manager = VectorstoreManager(store=store)
    doc = Document(
        page_content="Intergrax Harness connects Legal Agent and Research Agent.",
        metadata={"tenant_id": "g1"},
    )
    manager.add_documents([doc], [[0.1, 0.2, 0.3]], ids=["chunk-1"])

    graph = InMemoryGraphStore()
    HeuristicGraphIndexer(graph).index_documents([doc], chunk_ids=["chunk-1"])

    retriever = GraphRagRetriever(manager, _Emb(), graph, seed_top_k=1)
    hits = retriever.retrieve(
        RetrieverQuery(query_text="Intergrax Agent", query_embedding=None, top_k=3)
    )
    assert hits
    assert any("Intergrax" in h.content for h in hits)
