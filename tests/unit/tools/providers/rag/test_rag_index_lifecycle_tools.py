# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.knowledge.contracts import KnowledgeDocument
from intergrax.rag.vectorstore.contracts.native_vectorstore import (
    VectorStoreRecord,
    VectorStoreScope,
)
from intergrax.integrations.providers.vector_store.inmemory.rag_store import InMemoryVectorStore
from intergrax.rag.vectorstore.vectorstore_manager import VectorstoreManager
from intergrax.tools.providers.rag.index_lifecycle_contracts import (
    RagCheckIndexStatusInput,
    RagGetDocumentInput,
    RagListDocumentsInput,
    RagPurgeCollectionInput,
    RagSearchByMetadataInput,
)
from intergrax.tools.providers.rag.index_lifecycle_service import (
    perform_rag_check_index_status,
    perform_rag_get_document,
    perform_rag_list_documents,
    perform_rag_purge_collection,
    perform_rag_search_by_metadata,
)
from intergrax.tools.registry.wiring import ToolWiringContext

pytestmark = pytest.mark.unit


@pytest.fixture
def vectorstore_ctx() -> ToolWiringContext:
    store = InMemoryVectorStore(tenant_id="t-1")
    document = KnowledgeDocument.model_validate(
        {
            "schema_version": 1,
            "identity": {"document_id": "doc-1", "root_document_id": "doc-1"},
            "scope": {"tenant_id": "t-1"},
            "content": "alpha text",
            "metadata": {"source": "a.md"},
            "provenance": {"source_kind": "test", "source_id": "a.md"},
        }
    )
    scope = VectorStoreScope(tenant_id="t-1")
    store.add_records(
        [VectorStoreRecord(document=document, embedding=[1.0, 0.0], vector_id="doc-1")],
        scope=scope,
    )
    return ToolWiringContext(vectorstore_manager=VectorstoreManager(store))


def test_native_workspace_isolation_and_delete_readd() -> None:
    store = InMemoryVectorStore(tenant_id="t-1")
    manager = VectorstoreManager(store)

    def _record(workspace_id: str, content: str, vector_id: str) -> VectorStoreRecord:
        document = KnowledgeDocument.model_validate(
            {
                "schema_version": 1,
                "identity": {"document_id": "logical-doc", "root_document_id": "logical-doc"},
                "scope": {
                    "tenant_id": "t-1",
                    "namespace": "namespace-a",
                    "workspace_id": workspace_id,
                },
                "content": content,
                "metadata": {},
                "provenance": {"source_kind": "test", "source_id": vector_id},
            }
        )
        return VectorStoreRecord(
            document=document,
            embedding=[1.0, 0.0],
            vector_id=vector_id,
        )

    scope_a = VectorStoreScope(
        tenant_id="t-1",
        namespace="namespace-a",
        workspace_id="workspace-a",
    )
    scope_b = VectorStoreScope(
        tenant_id="t-1",
        namespace="namespace-a",
        workspace_id="workspace-b",
    )
    record_a = _record("workspace-a", "content A", "vector-a")
    record_b = _record("workspace-b", "content B", "vector-b")

    manager.add_records([record_a], scope=scope_a)
    manager.add_records([record_b], scope=scope_b)

    assert [hit.content for hit in manager.query([1.0, 0.0], scope=scope_a, top_k=5)] == [
        "content A"
    ]
    assert [hit.content for hit in manager.query([1.0, 0.0], scope=scope_b, top_k=5)] == [
        "content B"
    ]

    manager.delete(["vector-a"], scope=scope_a)
    assert manager.query([1.0, 0.0], scope=scope_a, top_k=5) == []
    assert manager.count(scope=scope_a) == 0
    assert manager.count(scope=scope_b) == 1

    manager.add_records([record_a], scope=scope_a)
    hits_a = manager.query([1.0, 0.0], scope=scope_a, top_k=5)
    assert [hit.vector_id for hit in hits_a] == ["vector-a"]
    assert len(hits_a) == 1


def test_rag_list_documents(vectorstore_ctx: ToolWiringContext) -> None:
    out = perform_rag_list_documents(vectorstore_ctx, RagListDocumentsInput(limit=10))
    assert out.used is True
    assert out.total == 1
    assert out.documents[0].document_id == "doc-1"


def test_rag_get_document(vectorstore_ctx: ToolWiringContext) -> None:
    out = perform_rag_get_document(vectorstore_ctx, RagGetDocumentInput(document_id="doc-1"))
    assert out.used is True
    assert "alpha text" in out.text
    assert out.metadata.get("source") == "a.md"


def test_rag_check_index_status(vectorstore_ctx: ToolWiringContext) -> None:
    out = perform_rag_check_index_status(vectorstore_ctx, RagCheckIndexStatusInput())
    assert out.used is True
    assert out.ready is True
    assert out.document_count == 1


def test_rag_search_by_metadata(vectorstore_ctx: ToolWiringContext) -> None:
    out = perform_rag_search_by_metadata(
        vectorstore_ctx,
        RagSearchByMetadataInput(filters={"source": "a.md"}, limit=10),
    )
    assert out.used is True
    assert out.total == 1
    assert out.matches[0].document_id == "doc-1"


def test_rag_purge_collection_dry_run(vectorstore_ctx: ToolWiringContext) -> None:
    out = perform_rag_purge_collection(vectorstore_ctx, RagPurgeCollectionInput(dry_run=True))
    assert out.used is True
    assert out.would_delete == 1
    assert perform_rag_list_documents(vectorstore_ctx, RagListDocumentsInput()).total == 1


def test_rag_purge_collection_executes(vectorstore_ctx: ToolWiringContext) -> None:
    out = perform_rag_purge_collection(vectorstore_ctx, RagPurgeCollectionInput(dry_run=False))
    assert out.used is True
    assert out.deleted == 1
    assert perform_rag_list_documents(vectorstore_ctx, RagListDocumentsInput()).total == 0


def test_rag_list_documents_unsupported_without_lifecycle_binding() -> None:
    class MinimalManager:
        def count(self) -> int:
            return 0

    ctx = ToolWiringContext(vectorstore_manager=MinimalManager())
    out = perform_rag_list_documents(ctx, RagListDocumentsInput())
    assert out.used is False
    assert out.reason == "list_documents_not_supported"
