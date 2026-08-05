# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unit tests for RAG vector store bootstrap via Integration Library."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.rag.vectorstore.bootstrap.integration_vectorstore import (
    create_default_vectorstore_manager,
    create_vectorstore_from_integration,
    create_vectorstore_manager,
)
from intergrax.integrations.providers.vector_store.inmemory.integration import (
    InmemoryVectorStoreIntegration,
)
from intergrax.integrations.providers.vector_store.inmemory.rag_store import InMemoryVectorStore
from intergrax.rag.vectorstore.vectorstore_manager import VectorstoreManager

pytestmark = pytest.mark.unit


def test_create_vectorstore_from_integration_falls_back_to_inmemory() -> None:
    store = create_vectorstore_from_integration(profile=IntegrationProfile())
    assert isinstance(store, InmemoryVectorStoreIntegration)
    assert isinstance(store.rag_store, InMemoryVectorStore)


def test_create_vectorstore_manager_wraps_resolved_store() -> None:
    mock_store = MagicMock()
    profile = IntegrationProfile(vector_store="qdrant")

    with patch(
        "intergrax.rag.vectorstore.bootstrap.integration_vectorstore.create_vectorstore_from_integration",
        return_value=mock_store,
    ):
        manager = create_vectorstore_manager(profile=profile, tenant_id="tenant-a")

    assert isinstance(manager, VectorstoreManager)
    assert manager._store is mock_store


def test_create_default_vectorstore_manager_uses_integration_path() -> None:
    with patch(
        "intergrax.rag.vectorstore.bootstrap.integration_vectorstore.create_vectorstore_manager",
        return_value=VectorstoreManager(store=InMemoryVectorStore(tenant_id="t")),
    ) as factory_mock:
        manager = create_default_vectorstore_manager(tenant_id="t")

    factory_mock.assert_called_once_with(tenant_id="t")
    assert isinstance(manager, VectorstoreManager)
