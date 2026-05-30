# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""
Resolve vector stores via the Integration Library catalog.

RAG Tier-0 code should prefer these factories over direct imports of
``intergrax.rag.vectorstore.providers.*`` when wiring production backends.
"""

from __future__ import annotations

from typing import Optional

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.contracts.vector_store import VectorStore
from intergrax.integrations.registry.bootstrap import register_default_integrations
from intergrax.integrations.registry.factory import build_profile_from_env, resolve
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.rag.vectorstore.contracts.base_vectorstore_manager import BaseVectorstoreManager
from intergrax.rag.vectorstore.providers.inmemory_vectorstore import InMemoryVectorStore
from intergrax.rag.vectorstore.vectorstore_manager import VectorstoreManager


def create_vectorstore_from_integration(
    *,
    profile: Optional[IntegrationProfile] = None,
    tenant_id: Optional[str] = None,
    **config_overrides: object,
) -> VectorStore:
    """
    Resolve ``IntegrationCategory.VECTOR_STORE`` from profile/env.

    Falls back to ``InMemoryVectorStore`` when no vector_store slug is configured.
    """
    register_default_integrations()
    resolved_profile = profile or build_profile_from_env()
    slug = resolved_profile.slug_for_category(IntegrationCategory.VECTOR_STORE)
    if not slug:
        return InMemoryVectorStore(tenant_id=tenant_id or "in_memory_tenant_id")

    from intergrax.integrations.registry.slugs import coerce_slug

    slug_enum = coerce_slug(slug)
    config = dict(resolved_profile.options_for_slug(slug_enum))
    if tenant_id is not None:
        config.setdefault("tenant_id", tenant_id)
    config.update(config_overrides)
    store = resolve(
        IntegrationCategory.VECTOR_STORE,
        profile=resolved_profile,
        config=config,
    )
    assert isinstance(store, VectorStore)
    return store


def create_vectorstore_manager(
    *,
    tenant_id: Optional[str] = None,
    profile: Optional[IntegrationProfile] = None,
    **config_overrides: object,
) -> BaseVectorstoreManager:
    """Composition root for RAG — wraps catalog-resolved store in ``VectorstoreManager``."""
    store = create_vectorstore_from_integration(
        profile=profile,
        tenant_id=tenant_id,
        **config_overrides,
    )
    return VectorstoreManager(store=store)


def create_default_vectorstore_manager(*, tenant_id: Optional[str] = None) -> BaseVectorstoreManager:
    """
    Backward-compatible alias.

    Uses ``IntegrationProfile`` / env when ``vector_store`` is set; otherwise in-memory.
    """
    return create_vectorstore_manager(tenant_id=tenant_id)
