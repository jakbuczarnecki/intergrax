# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from intergrax.integrations.providers.vector_store.inmemory.rag_store import InMemoryVectorStore
from intergrax.rag.vectorstore.vectorstore_manager import VectorstoreManager
from intergrax.tools.providers.vector_store.contracts import (
    VectorStoreCountInput,
    VectorStoreDeleteInput,
    VectorStoreHealthInput,
    VectorStoreListCollectionsInput,
)
from intergrax.tools.providers.vector_store.service import (
    vector_store_count,
    vector_store_delete,
    vector_store_health,
    vector_store_list_collections,
)
from intergrax.tools.registry.wiring import ToolWiringContext

pytestmark = pytest.mark.unit


@pytest.fixture
def vectorstore_ctx() -> ToolWiringContext:
    store = InMemoryVectorStore(tenant_id="t-1")
    store.add_documents(
        [Document(page_content="alpha", metadata={"source": "a.md"})],
        embeddings=[[1.0, 0.0]],
        ids=["doc-1"],
    )
    return ToolWiringContext(vectorstore_manager=VectorstoreManager(store))


def test_vector_store_count(vectorstore_ctx: ToolWiringContext) -> None:
    out = vector_store_count(vectorstore_ctx, VectorStoreCountInput())
    assert out.used is True
    assert out.document_count == 1


def test_vector_store_delete(vectorstore_ctx: ToolWiringContext) -> None:
    out = vector_store_delete(vectorstore_ctx, VectorStoreDeleteInput(document_ids=["doc-1"]))
    assert out.used is True
    assert out.deleted_count == 1
    count_out = vector_store_count(vectorstore_ctx, VectorStoreCountInput())
    assert count_out.document_count == 0


def test_vector_store_list_collections(vectorstore_ctx: ToolWiringContext) -> None:
    out = vector_store_list_collections(vectorstore_ctx, VectorStoreListCollectionsInput())
    assert out.used is True
    assert out.collections


def test_vector_store_health(vectorstore_ctx: ToolWiringContext) -> None:
    out = vector_store_health(vectorstore_ctx, VectorStoreHealthInput())
    assert out.used is True
    assert out.healthy is True
    assert out.document_count == 1


def test_vector_store_not_configured() -> None:
    out = vector_store_count(ToolWiringContext(), VectorStoreCountInput())
    assert out.used is False
    assert out.reason == "vectorstore_manager_not_configured"
