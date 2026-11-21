# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unit tests for RAG vector store bootstrap via Integration Library."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from intergrax.integrations.contracts.base import IntegrationConfigurationError, LabSelectorOptIn
from intergrax.integrations.providers.vector_store.inmemory.integration import (
    InmemoryVectorStoreIntegration,
)
from intergrax.integrations.providers.vector_store.inmemory.rag_store import InMemoryVectorStore
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.rag.vectorstore.bootstrap.integration_vectorstore import (
    create_default_vectorstore_manager,
    create_vectorstore_from_integration,
    create_vectorstore_manager,
    lab_inmemory_vector_store_slug,
)
from intergrax.rag.vectorstore.contracts.native_vectorstore import VectorStoreScope
from intergrax.rag.vectorstore.vectorstore_manager import VectorstoreManager

pytestmark = pytest.mark.unit


def test_create_vectorstore_from_integration_requires_slug_or_lab_opt_in() -> None:
    with pytest.raises(IntegrationConfigurationError, match="vector_store slug"):
        create_vectorstore_from_integration(
            profile=IntegrationProfile(),
            tenant_id="tenant-test",
        )


def test_create_vectorstore_from_integration_lab_opt_in_inmemory() -> None:
    lab_opt_in = LabSelectorOptIn(
        selector=lab_inmemory_vector_store_slug(),
        reason="integration vectorstore bootstrap unit test",
    )
    store = create_vectorstore_from_integration(
        profile=IntegrationProfile(),
        tenant_id="tenant-test",
        lab_opt_in=lab_opt_in,
    )
    assert isinstance(store, InmemoryVectorStoreIntegration)
    assert isinstance(store.rag_store, InMemoryVectorStore)
    assert store.rag_store.list_collections() == ["inmemory:tenant-test"]


def test_inmemory_integration_delegates_source_record_lookup() -> None:
    inner = InMemoryVectorStore(tenant_id="tenant-a")
    integration = InmemoryVectorStoreIntegration.from_store(object(), inner)

    assert (
        integration.list_source_record_ids(
            source_id="source-a",
            scope=VectorStoreScope(tenant_id="tenant-a"),
        )
        == []
    )


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
        manager = create_default_vectorstore_manager(
            tenant_id="t",
            lab_opt_in=LabSelectorOptIn(
                selector=lab_inmemory_vector_store_slug(),
                reason="default vectorstore manager wrapper test",
            ),
        )

    factory_mock.assert_called_once()
    assert factory_mock.call_args.kwargs["tenant_id"] == "t"
    assert isinstance(manager, VectorstoreManager)
