# © Artur Czarnecki. All rights reserved.

import pytest
from langchain_core.documents import Document

from intergrax.integrations.providers.vector_store.inmemory.rag_store import InMemoryVectorStore
from intergrax.rag.graph.indexer.heuristic_graph_indexer import HeuristicGraphIndexer
from intergrax.rag.graph.providers.inmemory_graph_store import InMemoryGraphStore
from intergrax.rag.profiles.rag_profile import RagProfile
from intergrax.rag.retrieval.retrieval_request import RetrievalRequest
from intergrax.rag.retrieval.retrieval_service import RetrievalService
from intergrax.rag.retrievers.bootstrap.retriever_bootstrap import create_default_retriever_manager
from intergrax.rag.vectorstore.vectorstore_manager import VectorstoreManager


class _Emb:
    def embed_one(self, text: str):
        return [0.1, 0.2, 0.3]


@pytest.mark.gate
def test_retrieval_trace_includes_structured_graph_provenance() -> None:
    store = InMemoryVectorStore(tenant_id="trace")
    manager = VectorstoreManager(store=store)
    doc = Document(
        page_content="Vertex Corp deploys Intergrax GraphRAG on Neo4j.",
        metadata={"tenant_id": "trace"},
    )
    manager.add_documents([doc], [[0.1, 0.2, 0.3]], ids=["chunk-vertex"])
    graph = InMemoryGraphStore()
    HeuristicGraphIndexer(graph).index_documents([doc], chunk_ids=["chunk-vertex"])

    profile = RagProfile(
        retriever_id="graph_rag",
        route_mode="off",
        graph_rag_enabled=True,
        enable_rerank=False,
    )
    retriever_manager = create_default_retriever_manager(
        vector_store=manager,
        embedding_manager=_Emb(),
        graph_store=graph,
        profile=profile,
    )
    service = RetrievalService(retriever_manager=retriever_manager, profile=profile)
    result = service.retrieve(RetrievalRequest(query="Vertex Corp", top_k=2))
    assert result.used is True
    assert result.trace.graph_expanded_node_ids or result.trace.graph_provenance_summary
    assert result.trace.graph_provenance_records
    record = result.trace.graph_provenance_records[0]
    assert "node_id" in record
    assert "explanation" in record
