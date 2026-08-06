from __future__ import annotations

import pytest

from intergrax.integrations.registry.catalog_manifests import INMEMORY
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.rag.vectorstore.bootstrap.integration_vectorstore import (
    create_default_vectorstore_manager,
    create_vectorstore_manager,
)
from intergrax.rag.vectorstore.contracts.native_vectorstore import (
    VectorStoreContractError,
    VectorStoreScope,
)
from intergrax.rag.vectorstore.vectorstore_manager import VectorstoreManager
from intergrax.integrations.providers.vector_store.inmemory.rag_store import (
    InMemoryVectorStore,
)


def test_explicit_tenant_binds_in_memory_fallback() -> None:
    manager = create_default_vectorstore_manager(tenant_id="tenant-explicit")

    assert isinstance(manager, VectorstoreManager)
    assert manager._bound_scope == VectorStoreScope(tenant_id="tenant-explicit")


def test_typed_integration_tenant_preserves_scope_context() -> None:
    profile = IntegrationProfile(
        vector_store=INMEMORY,
        options={
            INMEMORY.slug: {
                "tenant_id": "tenant-configured",
                "namespace": " knowledge ",
                "workspace_id": " workspace-a ",
            }
        },
    )

    manager = create_vectorstore_manager(profile=profile)

    assert isinstance(manager, VectorstoreManager)
    assert manager._bound_scope == VectorStoreScope(
        tenant_id="tenant-configured",
        namespace="knowledge",
        workspace_id="workspace-a",
    )


def test_scope_tenant_and_overrides_are_canonicalized() -> None:
    profile = IntegrationProfile(
        vector_store=INMEMORY,
        options={INMEMORY.slug: {"tenant_id": " tenant-a "}},
    )

    manager = create_vectorstore_manager(
        tenant_id=" tenant-a ",
        profile=profile,
        namespace=" knowledge ",
        workspace_id=" workspace-a ",
    )

    assert manager._bound_scope == VectorStoreScope(
        tenant_id="tenant-a",
        namespace="knowledge",
        workspace_id="workspace-a",
    )


def test_explicit_and_configured_tenant_mismatch_fails_closed() -> None:
    profile = IntegrationProfile(
        vector_store=INMEMORY,
        options={INMEMORY.slug: {"tenant_id": "tenant-configured"}},
    )

    with pytest.raises(ValueError, match="tenant_id sources disagree"):
        create_vectorstore_manager(tenant_id="tenant-explicit", profile=profile)


def test_missing_tenant_does_not_create_default_scope() -> None:
    with pytest.raises(ValueError, match="explicit tenant_id"):
        create_vectorstore_manager(profile=IntegrationProfile())


def test_provider_scope_mismatch_fails_closed() -> None:
    provider = InMemoryVectorStore(tenant_id="tenant-provider")

    with pytest.raises(ValueError, match="does not match provider tenant"):
        VectorstoreManager(
            provider,
            scope=VectorStoreScope(tenant_id="tenant-other"),
        )


def test_bound_manager_rejects_unrelated_operation_tenant() -> None:
    manager = create_vectorstore_manager(tenant_id="tenant-a")

    with pytest.raises(VectorStoreContractError, match="differs from bound scope"):
        manager.count(scope=VectorStoreScope(tenant_id="tenant-b"))
