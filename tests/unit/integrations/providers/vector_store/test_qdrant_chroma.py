# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unit tests for Qdrant and Chroma integration providers (Phase M.6 P2)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from intergrax.integrations._shared.conformance import assert_vector_store
from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationConfigurationError
from intergrax.integrations.contracts.vector_store import MetadataFilter, VectorStore, VectorStoreHit
from intergrax.integrations.providers.vector_store.chroma.adapter import ChromaVectorStoreIntegration
from intergrax.integrations.providers.vector_store.chroma.bundle import create_chroma_vector_store
from intergrax.integrations.providers.vector_store.chroma.config import ChromaIntegrationConfig
from intergrax.integrations.providers.vector_store.chroma.register import register_chroma_integration
from intergrax.integrations.providers.vector_store.qdrant.adapter import QdrantVectorStoreIntegration
from intergrax.integrations.providers.vector_store.qdrant.bundle import create_qdrant_vector_store
from intergrax.integrations.providers.vector_store.qdrant.config import QdrantIntegrationConfig
from intergrax.integrations.providers.vector_store.qdrant.register import register_qdrant_integration
from intergrax.integrations.registry.bootstrap import register_default_integrations, reset_default_integrations_state
from intergrax.integrations.registry.catalog import clear_catalog
from intergrax.integrations.registry.factory import resolve
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.integrations.registry.slugs import IntegrationSlug

pytestmark = pytest.mark.unit

_PROJECT_ROOT = Path(__file__).resolve().parents[3]


class _FakeVectorStore(VectorStore):
    def __init__(self) -> None:
        self.documents: list[Document] = []

    def add_documents(
        self,
        documents: Sequence[Document],
        embeddings: Sequence[Sequence[float]],
        *,
        ids: Optional[Sequence[str]] = None,
    ) -> None:
        self.documents.extend(documents)

    def query(
        self,
        query_embedding: Sequence[float],
        *,
        top_k: int,
        metadata_filter: Optional[MetadataFilter] = None,
        include_embeddings: bool = False,
    ) -> list[VectorStoreHit]:
        return [
            VectorStoreHit(
                id="doc-1",
                content="hello",
                metadata={},
                similarity_score=0.9,
                rank=0,
            )
        ]

    def delete(self, ids: Sequence[str]) -> None:
        return None

    def count(self) -> int:
        return len(self.documents)


@pytest.fixture(autouse=True)
def _clean_catalog() -> None:
    clear_catalog()
    reset_default_integrations_state()
    yield
    clear_catalog()
    reset_default_integrations_state()


def _store_factory(store: _FakeVectorStore | None = None):
    fake = store or _FakeVectorStore()

    def _factory() -> _FakeVectorStore:
        return fake

    return _factory, fake


def test_qdrant_opens_imports_client_and_builds_rag_store() -> None:
    config = QdrantIntegrationConfig(collection_name="coll", tenant_id="t1")
    mock_rag_store = MagicMock()

    with patch("intergrax.integrations.providers.vector_store.qdrant.opens._import_qdrant_client") as import_mock:
        with patch(
            "intergrax.rag.vectorstore.providers.qdrant_vector_store.QdrantVectorStore",
            return_value=mock_rag_store,
        ) as rag_cls:
            from intergrax.integrations.providers.vector_store.qdrant.opens import open_qdrant_vector_store

            result = open_qdrant_vector_store(config)

    import_mock.assert_called_once()
    rag_cls.assert_called_once()
    assert isinstance(result, QdrantVectorStoreIntegration)
    assert result.rag_store is mock_rag_store


def test_chroma_opens_imports_chromadb_and_builds_rag_store() -> None:
    config = ChromaIntegrationConfig(collection_name="coll", tenant_id="t1")
    mock_rag_store = MagicMock()

    with patch("intergrax.integrations.providers.vector_store.chroma.opens._import_chromadb") as import_mock:
        with patch(
            "intergrax.rag.vectorstore.providers.chroma_vector_store.ChromaVectorStore",
            return_value=mock_rag_store,
        ) as rag_cls:
            from intergrax.integrations.providers.vector_store.chroma.opens import open_chroma_vector_store

            result = open_chroma_vector_store(config)

    import_mock.assert_called_once()
    rag_cls.assert_called_once()
    assert isinstance(result, ChromaVectorStoreIntegration)
    assert result.rag_store is mock_rag_store


def test_qdrant_register_and_resolve() -> None:
    register_qdrant_integration()
    factory, _ = _store_factory()
    store = resolve(
        IntegrationCategory.VECTOR_STORE,
        profile=IntegrationProfile(vector_store=IntegrationSlug.QDRANT),
        config={"collection_name": "c1", "tenant_id": "t1", "store_factory": factory},
    )
    assert_vector_store(store)
    assert isinstance(store, QdrantVectorStoreIntegration)


def test_chroma_register_and_resolve() -> None:
    register_chroma_integration()
    factory, _ = _store_factory()
    store = resolve(
        IntegrationCategory.VECTOR_STORE,
        profile=IntegrationProfile(vector_store=IntegrationSlug.CHROMA),
        config={"collection_name": "c1", "tenant_id": "t1", "store_factory": factory},
    )
    assert_vector_store(store)
    assert isinstance(store, ChromaVectorStoreIntegration)


def test_register_default_integrations_includes_qdrant_and_chroma() -> None:
    register_default_integrations()
    factory, _ = _store_factory()
    for slug in (IntegrationSlug.QDRANT, IntegrationSlug.CHROMA):
        store = resolve(
            IntegrationCategory.VECTOR_STORE,
            profile=IntegrationProfile(vector_store=slug),
            config={"collection_name": "c1", "tenant_id": "t1", "store_factory": factory},
        )
        assert_vector_store(store)


def test_chroma_missing_package_raises_configuration_error() -> None:
    config = ChromaIntegrationConfig(collection_name="c1", tenant_id="t1")
    with patch(
        "intergrax.integrations.providers.vector_store.chroma.opens._import_chromadb",
        side_effect=IntegrationConfigurationError("missing chromadb"),
    ):
        from intergrax.integrations.providers.vector_store.chroma.opens import open_chroma_vector_store

        with pytest.raises(IntegrationConfigurationError, match="missing chromadb"):
            open_chroma_vector_store(config)


def test_qdrant_sdk_only_in_opens_module() -> None:
    pkg = _PROJECT_ROOT / "intergrax" / "integrations" / "providers" / "qdrant"
    violations = [
        path.name
        for path in pkg.glob("*.py")
        if path.name != "opens.py" and "qdrant_client" in path.read_text(encoding="utf-8")
    ]
    assert violations == []


def test_chromadb_only_in_opens_module() -> None:
    pkg = _PROJECT_ROOT / "intergrax" / "integrations" / "providers" / "chroma"
    violations = [
        path.name
        for path in pkg.glob("*.py")
        if path.name != "opens.py" and "chromadb" in path.read_text(encoding="utf-8")
    ]
    assert violations == []


def test_create_qdrant_vector_store_delegates() -> None:
    factory, inner = _store_factory()
    store = create_qdrant_vector_store(collection_name="c1", tenant_id="t1", store_factory=factory)
    store.add_documents([Document(page_content="x")], [[0.1, 0.2]])
    assert inner.documents


def test_create_chroma_vector_store_delegates() -> None:
    factory, inner = _store_factory()
    store = create_chroma_vector_store(collection_name="c1", tenant_id="t1", store_factory=factory)
    store.add_documents([Document(page_content="x")], [[0.1, 0.2]])
    assert inner.documents
