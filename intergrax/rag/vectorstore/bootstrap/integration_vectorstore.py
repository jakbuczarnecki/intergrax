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
from intergrax.rag.vectorstore.contracts.native_vectorstore import VectorStoreScope
from intergrax.rag.vectorstore.vectorstore_manager import VectorstoreManager


def _validated_tenant_id(value: object, *, source: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{source} tenant_id must be a non-empty string")
    return value


def _resolve_tenant_id(
    *,
    explicit_tenant_id: object | None,
    configured_tenant_id: object | None = None,
    override_tenant_id: object | None = None,
) -> str | None:
    candidates = (
        ("explicit", explicit_tenant_id),
        ("integration profile", configured_tenant_id),
        ("integration override", override_tenant_id),
    )
    resolved: list[tuple[str, str]] = [
        (source, _validated_tenant_id(value, source=source))
        for source, value in candidates
        if value is not None
    ]
    if not resolved:
        return None
    values = {value for _, value in resolved}
    if len(values) > 1:
        sources = ", ".join(source for source, _ in resolved)
        raise ValueError(f"tenant_id sources disagree ({sources})")
    return next(iter(values))


def _profile_tenant_id(
    profile: IntegrationProfile,
    slug: object | None,
) -> object | None:
    if slug is None:
        return None
    return profile.options_for_slug(slug).get("tenant_id")  # type: ignore[arg-type]


def _validated_scope_part(value: object | None, *, source: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{source} must be a non-empty string")
    return value


def _scope_from_integration_config(
    *,
    tenant_id: str | None,
    profile: IntegrationProfile,
    slug: object | None,
    config_overrides: dict[str, object],
) -> VectorStoreScope | None:
    if tenant_id is None:
        return None
    config = profile.options_for_slug(slug) if slug is not None else {}
    config.update(config_overrides)
    return VectorStoreScope(
        tenant_id=tenant_id,
        namespace=_validated_scope_part(
            config.get("namespace"),
            source="vectorstore namespace",
        ),
        workspace_id=_validated_scope_part(
            config.get("workspace_id"),
            source="vectorstore workspace_id",
        ),
    )


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
    slug_enum = None
    configured_tenant_id: object | None = None
    if slug:
        from intergrax.integrations.registry.slugs import coerce_slug

        slug_enum = coerce_slug(slug)
        configured_tenant_id = _profile_tenant_id(resolved_profile, slug_enum)
    resolved_tenant_id = _resolve_tenant_id(
        explicit_tenant_id=tenant_id,
        configured_tenant_id=configured_tenant_id,
        override_tenant_id=config_overrides.get("tenant_id"),
    )
    if not slug:
        from intergrax.integrations.providers.vector_store.inmemory.bundle import create_inmemory_vector_store

        if resolved_tenant_id is None:
            raise ValueError("in-memory vectorstore requires an explicit tenant_id")
        return create_inmemory_vector_store(tenant_id=resolved_tenant_id)

    config = dict(resolved_profile.options_for_slug(slug_enum))
    if resolved_tenant_id is not None:
        config["tenant_id"] = resolved_tenant_id
    config.update(config_overrides)
    if resolved_tenant_id is not None:
        config["tenant_id"] = resolved_tenant_id
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
    register_default_integrations()
    resolved_profile = profile or build_profile_from_env()
    slug = resolved_profile.slug_for_category(IntegrationCategory.VECTOR_STORE)
    slug_enum = None
    configured_tenant_id: object | None = None
    if slug:
        from intergrax.integrations.registry.slugs import coerce_slug

        slug_enum = coerce_slug(slug)
        configured_tenant_id = _profile_tenant_id(
            resolved_profile,
            slug_enum,
        )
    resolved_tenant_id = _resolve_tenant_id(
        explicit_tenant_id=tenant_id,
        configured_tenant_id=configured_tenant_id,
        override_tenant_id=config_overrides.get("tenant_id"),
    )
    store = create_vectorstore_from_integration(
        profile=resolved_profile,
        tenant_id=resolved_tenant_id,
        **config_overrides,
    )
    scope = _scope_from_integration_config(
        tenant_id=resolved_tenant_id,
        profile=resolved_profile,
        slug=slug_enum,
        config_overrides=config_overrides,
    )
    return VectorstoreManager(store=store, scope=scope)


def create_default_vectorstore_manager(*, tenant_id: Optional[str] = None) -> BaseVectorstoreManager:
    """
    Backward-compatible alias.

    Uses ``IntegrationProfile`` / env when ``vector_store`` is set; otherwise in-memory.
    """
    return create_vectorstore_manager(tenant_id=tenant_id)
