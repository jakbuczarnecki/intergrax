# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.integrations.providers.vector_store.inmemory.rag_store import (
    InMemoryVectorStore,
)
from intergrax.rag.graph.indexer.heuristic_graph_indexer import HeuristicGraphIndexer
from intergrax.rag.graph.providers.inmemory_graph_store import InMemoryGraphStore
from intergrax.rag.retrievers.contracts.base_retriever import RetrieverQuery
from intergrax.rag.retrievers.providers.graph_rag_retriever import GraphRagRetriever
from intergrax.rag.vectorstore.vectorstore_manager import VectorstoreManager
from tests.unit.rag.graph.fixtures import knowledge_document


class _Emb:
    def embed_one(self, text: str):
        return [0.1, 0.2, 0.3]


@pytest.mark.gate
def test_graph_rag_uses_chunk_linked_entities_not_substring_only() -> None:
    store = InMemoryVectorStore(tenant_id="g2")
    manager = VectorstoreManager(store=store)
    doc = knowledge_document(
        "Zephyr Labs partners with Orion Systems for cloud RAG.",
        tenant_id="g2",
    )
    manager.add_documents([doc], [[0.1, 0.2, 0.3]], ids=["chunk-zephyr"])

    graph = InMemoryGraphStore()
    indexer = HeuristicGraphIndexer(graph)
    indexer.index_documents([doc], chunk_ids=["chunk-zephyr"])
    from intergrax.rag.graph.contracts.graph_store import GraphNode

    graph.upsert_node(GraphNode(id="ent:zephyr_labs", label="Zephyr Labs", node_type="entity"))
    graph.link_chunk("ent:zephyr_labs", "chunk-zephyr")

    retriever = GraphRagRetriever(manager, _Emb(), graph, seed_top_k=1, graph_hops=1)
    hits = retriever.retrieve(
        RetrieverQuery(query_text="unrelated query token", query_embedding=None, top_k=2)
    )
    assert hits
    assert retriever.last_graph_trace is not None
    assert "ent:zephyr_labs" in retriever.last_graph_trace.expanded_node_ids


@pytest.mark.gate
def test_graph_rag_retriever_is_stable() -> None:
    assert GraphRagRetriever.STABILITY == "stable"


@pytest.mark.gate
def test_graph_rag_empty_result_is_native_tuple() -> None:
    retriever = GraphRagRetriever(object(), object(), object())  # type: ignore[arg-type]

    assert (
        retriever.retrieve(
            RetrieverQuery(query_text="", query_embedding=None, top_k=2)
        )
        == ()
    )
