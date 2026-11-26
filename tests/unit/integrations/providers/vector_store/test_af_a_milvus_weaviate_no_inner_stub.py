# © Artur Czarnecki. All rights reserved.

"""ARCH-FREEZE package A: Milvus and Weaviate must not fall back to in-memory stubs."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.vector_store.milvus.rag_store import MilvusConfig, MilvusVectorStore
from intergrax.integrations.providers.vector_store.weaviate.rag_store import (
    WeaviateConfig,
    WeaviateVectorStore,
)
from intergrax.knowledge.contracts import KnowledgeDocument
from intergrax.rag.vectorstore.contracts.native_vectorstore import VectorStoreRecord, VectorStoreScope

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _record() -> VectorStoreRecord:
    document = KnowledgeDocument.model_validate(
        {
            "schema_version": 1,
            "identity": {"document_id": "doc-1", "root_document_id": "doc-1"},
            "scope": {"tenant_id": "tenant-a"},
            "content": "sample",
            "metadata": {},
            "provenance": {"source_kind": "unit", "source_id": "doc-1"},
        }
    )
    return VectorStoreRecord(
        document=document,
        embedding=np.array([1.0, 0.0], dtype=np.float32),
        vector_id="doc-1",
    )


def test_milvus_add_records_does_not_write_inmemory_stub() -> None:
    store = MilvusVectorStore(MilvusConfig(collection_name="c", tenant_id="tenant-a"), client=object())
    assert not hasattr(store, "_inner")
    inmemory_spy = MagicMock()
    with patch(
        "intergrax.integrations.providers.vector_store.inmemory.rag_store.InMemoryVectorStore.add_records",
        inmemory_spy,
    ):
        with pytest.raises(IntegrationConfigurationError):
            store.add_records(
                [_record()],
                scope=VectorStoreScope(tenant_id="tenant-a"),
            )
    inmemory_spy.assert_not_called()


class _WeaviateData:
    def __init__(self) -> None:
        self.insert = MagicMock(side_effect=RuntimeError("insert failed"))

    def delete_by_id(self, _doc_id: str) -> None:
        return None


class _WeaviateCollection:
    def __init__(self) -> None:
        self.data = _WeaviateData()
        self.query = SimpleNamespace(hybrid=MagicMock(return_value=SimpleNamespace(objects=[])))
        self.aggregate = SimpleNamespace(over_all=MagicMock(return_value=SimpleNamespace(total_count=0)))

    def with_tenant(self, _tenant_id: str) -> _WeaviateCollection:
        return self


class _WeaviateClient:
    def __init__(self) -> None:
        self.collections = SimpleNamespace(list_all=MagicMock(return_value={}))


@patch(
    "intergrax.integrations.providers.vector_store.weaviate.rag_store.ensure_weaviate_collection",
    return_value=_WeaviateCollection(),
)
def test_weaviate_insert_error_does_not_call_inmemory_add_records(
    _ensure: MagicMock,
) -> None:
    store = WeaviateVectorStore(
        WeaviateConfig(collection_name="coll", tenant_id="tenant-a"),
        client=_WeaviateClient(),
    )
    inmemory_spy = MagicMock()
    with patch(
        "intergrax.integrations.providers.vector_store.inmemory.rag_store.InMemoryVectorStore.add_records",
        inmemory_spy,
    ):
        with pytest.raises(RuntimeError, match="insert failed"):
            store.add_records(
                [_record()],
                scope=VectorStoreScope(tenant_id="tenant-a"),
            )
    inmemory_spy.assert_not_called()


@patch(
    "intergrax.integrations.providers.vector_store.weaviate.rag_store.ensure_weaviate_collection",
    return_value=_WeaviateCollection(),
)
def test_weaviate_scoped_delete_does_not_touch_inmemory_stub(_ensure: MagicMock) -> None:
    store = WeaviateVectorStore(
        WeaviateConfig(collection_name="coll", tenant_id="tenant-a"),
        client=_WeaviateClient(),
    )
    inmemory_delete = MagicMock()
    with patch(
        "intergrax.integrations.providers.vector_store.inmemory.rag_store.InMemoryVectorStore.delete",
        inmemory_delete,
    ):
        with pytest.raises(RuntimeError, match="scoped delete is unsupported"):
            store.delete(
                ["doc-1"],
                scope=VectorStoreScope(tenant_id="tenant-a", namespace="ns"),
            )
    inmemory_delete.assert_not_called()
