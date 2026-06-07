# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from intergrax.integrations.providers.vector_store.inmemory.rag_store import InMemoryVectorStore
from intergrax.rag.vectorstore.vectorstore_manager import VectorstoreManager
from intergrax.tools.providers.rag.index_lifecycle_contracts import (
    RagCheckIndexStatusInput,
    RagGetDocumentInput,
    RagListDocumentsInput,
)
from intergrax.tools.providers.rag.index_lifecycle_service import (
    perform_rag_check_index_status,
    perform_rag_get_document,
    perform_rag_list_documents,
)
from intergrax.tools.registry.wiring import ToolWiringContext

pytestmark = pytest.mark.unit


@pytest.fixture
def vectorstore_ctx() -> ToolWiringContext:
    store = InMemoryVectorStore(tenant_id="t-1")
    store.add_documents(
        [Document(page_content="alpha text", metadata={"source": "a.md"})],
        embeddings=[[1.0, 0.0]],
        ids=["doc-1"],
    )
    return ToolWiringContext(vectorstore_manager=VectorstoreManager(store))


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


def test_rag_list_documents_unsupported_without_lifecycle_binding() -> None:
    class MinimalManager:
        def count(self) -> int:
            return 0

    ctx = ToolWiringContext(vectorstore_manager=MinimalManager())
    out = perform_rag_list_documents(ctx, RagListDocumentsInput())
    assert out.used is False
    assert out.reason == "list_documents_not_supported"
