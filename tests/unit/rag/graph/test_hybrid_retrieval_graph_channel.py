# © Artur Czarnecki. All rights reserved.

import pytest
from langchain_core.documents import Document

from intergrax.integrations.providers.vector_store.inmemory.rag_store import InMemoryVectorStore
from intergrax.rag.graph.indexer.heuristic_graph_indexer import HeuristicGraphIndexer
from intergrax.rag.graph.providers.inmemory_graph_store import InMemoryGraphStore
from intergrax.rag.retrievers.contracts.base_retriever import RetrieverQuery
from intergrax.rag.retrievers.providers.graph_rag_retriever import GraphRagRetriever
from intergrax.rag.vectorstore.vectorstore_manager import VectorstoreManager


class _Emb:
    def embed_one(self, text: str):
        return [0.1, 0.2, 0.3]


@pytest.mark.gate
def test_graph_rag_exposes_channel_contributions() -> None:
    store = InMemoryVectorStore(tenant_id="hybrid")
    manager = VectorstoreManager(store=store)
    doc = Document(
        page_content="Nimbus Analytics uses Intergrax Harness for retrieval fusion.",
        metadata={"tenant_id": "hybrid"},
    )
    manager.add_documents([doc], [[0.1, 0.2, 0.3]], ids=["chunk-nimbus"])
    graph = InMemoryGraphStore()
    HeuristicGraphIndexer(graph).index_documents([doc], chunk_ids=["chunk-nimbus"])

    retriever = GraphRagRetriever(manager, _Emb(), graph, seed_top_k=1, hybrid_fusion_enabled=True)
    retriever.retrieve(
        RetrieverQuery(query_text="Nimbus Analytics", query_embedding=None, top_k=2)
    )
    trace = retriever.last_graph_trace
    assert trace is not None
    assert "vector" in trace.channel_contributions
    assert trace.channel_contributions["vector"]
