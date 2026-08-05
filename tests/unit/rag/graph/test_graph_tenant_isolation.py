# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.rag.graph.indexer.heuristic_graph_indexer import HeuristicGraphIndexer
from intergrax.rag.graph.providers.inmemory_graph_store import InMemoryGraphStore
from intergrax.rag.graph.tenant.graph_isolation_contract import (
    run_graph_isolation_contract,
)
from tests.unit.rag.graph.fixtures import knowledge_document


@pytest.mark.gate
def test_inmemory_graph_tenant_isolation_contract() -> None:
    result = run_graph_isolation_contract(
        lambda tenant: InMemoryGraphStore(tenant_id=tenant),
        slug="inmemory",
    )
    assert result.cross_query_isolated is True
    assert result.reason == "ok"


def test_graph_rejects_mixed_tenants_and_preserves_chunk_isolation() -> None:
    store_a = InMemoryGraphStore(tenant_id="tenant-a")
    store_b = InMemoryGraphStore(tenant_id="tenant-b")
    doc_a = knowledge_document(
        "Acme Corp uses Intergrax Harness.",
        tenant_id="tenant-a",
        document_id="same-document-id",
    )
    doc_b = knowledge_document(
        "Acme Corp uses Intergrax Harness.",
        tenant_id="tenant-b",
        document_id="same-document-id",
    )

    with pytest.raises(ValueError, match="tenant"):
        HeuristicGraphIndexer(store_a).index_documents(
            [doc_a, doc_b],
            chunk_ids=["chunk-a", "chunk-b"],
        )
    assert store_a.find_nodes(label_contains="Acme", limit=5) == []

    HeuristicGraphIndexer(store_a).index_documents([doc_a], chunk_ids=["chunk-a"])
    HeuristicGraphIndexer(store_b).index_documents([doc_b], chunk_ids=["chunk-b"])
    assert store_a.node_ids_for_chunks({"chunk-b"}) == set()
    assert store_b.node_ids_for_chunks({"chunk-a"}) == set()


def test_tenant_metadata_is_not_an_authoritative_graph_scope() -> None:
    with pytest.raises(ValueError):
        knowledge_document(
            "Spoofed tenant metadata must be rejected.",
            tenant_id="tenant-a",
            metadata={"tenant_id": "tenant-b"},
        )
    with pytest.raises(ValueError):
        knowledge_document(
            "Spoofed workspace metadata must be rejected.",
            tenant_id="tenant-a",
            metadata={"workspace_id": "workspace-b"},
        )

    document = knowledge_document(
        "Acme Corp trusts Beta Labs; user metadata cannot change graph scope.",
        tenant_id="tenant-a",
        workspace_id="workspace-a",
        metadata={"tenant_hint": "tenant-b", "workspace_hint": "workspace-b"},
    )
    store = InMemoryGraphStore(tenant_id="tenant-a")
    HeuristicGraphIndexer(store).index_documents([document], chunk_ids=["chunk-scope"])

    assert document.scope.tenant_id == "tenant-a"
    assert document.scope.workspace_id == "workspace-a"
    assert store.node_ids_for_chunks({"chunk-scope"})
