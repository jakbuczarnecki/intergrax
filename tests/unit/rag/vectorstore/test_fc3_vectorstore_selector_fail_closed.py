# © Artur Czarnecki. All rights reserved.

"""FC-3 Part B: vector store selector fail-closed (NT-FC3-V1..V4)."""

from __future__ import annotations

import pytest
from intergrax.integrations.contracts.base import IntegrationConfigurationError, LabSelectorOptIn
from intergrax.integrations.providers.vector_store.inmemory.integration import (
    InmemoryVectorStoreIntegration,
)
from intergrax.integrations.providers.vector_store.inmemory.rag_store import InMemoryVectorStore
from intergrax.integrations.registry.bootstrap import (
    register_default_integrations,
    reset_default_integrations_state,
)
from intergrax.integrations.registry.catalog import clear_catalog
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.rag.vectorstore.bootstrap.integration_vectorstore import (
    create_default_vectorstore_manager,
    create_vectorstore_from_integration,
    create_vectorstore_manager,
)

pytestmark = pytest.mark.unit

_LAB_VECTOR_STORE = "inmemory"
_LAB_OPT_IN = LabSelectorOptIn(
    selector=_LAB_VECTOR_STORE,
    reason="FC-3 vector store unit negative-test positive control",
)


@pytest.fixture(autouse=True)
def _bootstrap_integration_catalog() -> None:
    clear_catalog()
    reset_default_integrations_state()
    register_default_integrations(preset="full")
    yield
    clear_catalog()
    reset_default_integrations_state()


def test_nt_fc3_v1_empty_profile_requires_binding() -> None:
    with pytest.raises(IntegrationConfigurationError):
        create_vectorstore_from_integration(
            profile=IntegrationProfile(),
            tenant_id="tenant-a",
        )


def test_nt_fc3_v2_lab_opt_in_yields_tenant_scoped_inmemory() -> None:
    store = create_vectorstore_from_integration(
        profile=IntegrationProfile(),
        tenant_id="tenant-a",
        lab_opt_in=_LAB_OPT_IN,
    )
    assert isinstance(store, InmemoryVectorStoreIntegration)
    assert isinstance(store.rag_store, InMemoryVectorStore)
    assert store.rag_store.list_collections() == [f"{_LAB_VECTOR_STORE}:tenant-a"]


def test_nt_fc3_v3_lab_opt_in_still_requires_tenant() -> None:
    with pytest.raises(ValueError, match="explicit tenant_id"):
        create_vectorstore_from_integration(
            profile=IntegrationProfile(),
            lab_opt_in=_LAB_OPT_IN,
        )


def test_nt_fc3_v4_manager_without_profile_or_env_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("INTERGRAX_INTEGRATION_VECTOR_STORE", raising=False)
    with pytest.raises(IntegrationConfigurationError):
        create_vectorstore_manager()
    with pytest.raises(IntegrationConfigurationError):
        create_default_vectorstore_manager()
