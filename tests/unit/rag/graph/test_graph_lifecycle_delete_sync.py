# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.rag.graph.indexer.heuristic_graph_indexer import HeuristicGraphIndexer
from intergrax.rag.graph.lifecycle.graph_lifecycle_sync import (
    sync_graph_purge_collection,
)
from intergrax.rag.graph.providers.inmemory_graph_store import InMemoryGraphStore
from intergrax.tools.providers.rag.index_lifecycle_contracts import (
    RagPurgeCollectionInput,
)
from intergrax.tools.providers.rag.index_lifecycle_service import (
    perform_rag_purge_collection,
)
from intergrax.tools.providers.rag.lifecycle_contracts import RagDeleteDocumentsInput
from intergrax.tools.providers.rag.lifecycle_service import perform_rag_delete_documents
from intergrax.tools.registry.wiring import ToolWiringContext
from tests.unit.rag.graph.fixtures import knowledge_document


class _FakeVectorstore:
    def __init__(self) -> None:
        self.deleted: list[str] = []

    def delete(self, ids: list[str]) -> None:
        self.deleted.extend(ids)


class _PurgeVectorstore:
    def list_document_ids(self, *, limit: int = 100, offset: int = 0) -> list[str]:
        return []

    def get_document(self, document_id: str) -> dict | None:
        return None

    def list_collections(self) -> list[str]:
        return ["default"]

    def count(self) -> int:
        return 0

    def search_by_metadata(self, *, conditions: dict, limit: int = 50) -> list[dict]:
        return []

    def purge_collection(self, *, dry_run: bool = True, tenant_id: str = "") -> dict:
        return {"dry_run": dry_run, "would_delete": 1, "deleted": 0 if dry_run else 1, "tenant_id": tenant_id}


@pytest.mark.gate
def test_unlink_chunks_prunes_orphan_entities() -> None:
    store = InMemoryGraphStore()
    indexer = HeuristicGraphIndexer(store)
    doc = knowledge_document("Acme Corp partners with Beta Labs for enterprise RAG.")
    indexer.index_documents([doc], chunk_ids=["chunk-acme"])
    assert store.find_nodes(label_contains="Acme", limit=5)

    removed = store.unlink_chunks(["chunk-acme"])
    assert removed >= 1
    assert not store.find_nodes(label_contains="Acme", limit=5)


@pytest.mark.gate
def test_delete_documents_tool_syncs_graph() -> None:
    graph = InMemoryGraphStore()
    indexer = HeuristicGraphIndexer(graph)
    doc = knowledge_document("Gamma Industries uses Intergrax Harness GraphRAG.")
    indexer.index_documents([doc], chunk_ids=["chunk-gamma"])
    ctx = ToolWiringContext(
        vectorstore_manager=_FakeVectorstore(),
        rag_graph_store=graph,
    )
    out = perform_rag_delete_documents(ctx, RagDeleteDocumentsInput(document_ids=["chunk-gamma"]))
    assert out.used is True
    assert "graph_unlinked" in out.reason
    assert not graph.find_nodes(label_contains="Gamma", limit=5)


@pytest.mark.gate
def test_purge_collection_tool_syncs_graph() -> None:
    graph = InMemoryGraphStore(tenant_id="tenant-purge")
    indexer = HeuristicGraphIndexer(graph)
    doc = knowledge_document(
        "Delta Systems signed with Intergrax Harness.",
        tenant_id="tenant-purge",
    )
    indexer.index_documents([doc], chunk_ids=["chunk-delta"])
    ctx = ToolWiringContext(
        vectorstore_manager=_PurgeVectorstore(),
        rag_graph_store=graph,
    )
    out = perform_rag_purge_collection(
        ctx,
        RagPurgeCollectionInput(tenant_id="tenant-purge", dry_run=False),
    )
    assert out.used is True
    assert "graph_purged" in out.reason
    assert sync_graph_purge_collection(graph, tenant_id="tenant-purge") == 0
