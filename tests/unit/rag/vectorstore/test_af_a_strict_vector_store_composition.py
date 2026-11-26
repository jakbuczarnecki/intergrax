# © Artur Czarnecki. All rights reserved.

"""ARCH-FREEZE package A: strict host vector store catalog admission."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from intergrax.integrations.contracts.base import (
    IntegrationCategory,
    IntegrationConfigurationError,
    IntegrationStatus,
)
from intergrax.integrations.contracts.integration_profile import IntegrationProfile
from intergrax.integrations.core.binding import IntegrationBinding
from intergrax.integrations.providers.vector_store.inmemory.rag_store import InMemoryVectorStore
from intergrax.integrations.registry.bootstrap import (
    register_default_integrations,
    reset_default_integrations_state,
)
from intergrax.integrations.registry.catalog import clear_catalog, iter_entries
from intergrax.integrations.registry.catalog_manifests import QDRANT
from intergrax.rag.vectorstore.bootstrap.integration_vectorstore import create_vectorstore_manager

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _catalog() -> None:
    clear_catalog()
    reset_default_integrations_state()
    register_default_integrations(preset="full")
    yield
    clear_catalog()
    reset_default_integrations_state()


def _beta_vector_slugs() -> list[str]:
    return sorted(
        entry.slug
        for entry in iter_entries()
        if IntegrationCategory.VECTOR_STORE in entry.categories and entry.status is not IntegrationStatus.STABLE
    )


@pytest.mark.parametrize("slug", _beta_vector_slugs())
def test_strict_composition_refuses_non_stable_catalog_slug(slug: str) -> None:
    profile = IntegrationProfile(
        vector_store=IntegrationBinding.from_slug(slug),
        options={slug: {"tenant_id": "tenant-a"}},
    )
    inmemory_ctor = InMemoryVectorStore.__init__
    with patch.object(InMemoryVectorStore, "__init__", autospec=True) as mocked_init:
        mocked_init.side_effect = lambda self, *args, **kwargs: inmemory_ctor(self, *args, **kwargs)
        with pytest.raises(IntegrationConfigurationError, match="STABLE"):
            create_vectorstore_manager(
                profile=profile,
                tenant_id="tenant-a",
                composition_fails_closed=True,
            )
    mocked_init.assert_not_called()


def test_strict_composition_stable_slug_follows_existing_resolution_path() -> None:
    profile = IntegrationProfile(
        vector_store=QDRANT,
        options={QDRANT.slug: {"tenant_id": "tenant-q"}},
    )
    sentinel = MagicMock(name="vector_store")
    with patch(
        "intergrax.rag.vectorstore.bootstrap.integration_vectorstore.resolve",
        return_value=sentinel,
    ) as resolve_mock:
        manager = create_vectorstore_manager(
            profile=profile,
            tenant_id="tenant-q",
            composition_fails_closed=True,
        )
    resolve_mock.assert_called_once()
    assert manager._store is sentinel
