# © Artur Czarnecki. All rights reserved.

"""Typed Memory store plugin materialization (ENTERPRISE-5 / BLOCK D)."""

from __future__ import annotations

from intergrax.memory.resolver.classifier import (
    ClassifiedMemoryStorePlugin,
    MemoryStorePluginKind,
)
from intergrax.memory.resolver.discovery import (
    MemoryStorePluginCatalog,
    find_failed_entry_point_for_plugin_id,
    find_rejected_entry_point_for_plugin_id,
)
from intergrax.memory.resolver.errors import MemoryStorePluginResolutionError
from intergrax.memory.resolver.materialization import MemoryStoreMaterializationContext
from intergrax.memory.user_profile_store import UserProfileStore
from intergrax.runtime.nexus.session.session_storage import SessionStorage


def _user_profile_factory_kwargs(ctx: MemoryStoreMaterializationContext) -> dict[str, object]:
    kwargs: dict[str, object] = {}
    if ctx.tenant_id is not None:
        kwargs["tenant_id"] = ctx.tenant_id
    return kwargs


def _session_storage_factory_kwargs(ctx: MemoryStoreMaterializationContext) -> dict[str, object]:
    kwargs: dict[str, object] = {}
    if ctx.tenant_id is not None:
        kwargs["tenant_id"] = ctx.tenant_id
    return kwargs


def _select_classified_plugin(
    plugin_id: str,
    *,
    expected_kind: MemoryStorePluginKind,
    catalog: MemoryStorePluginCatalog,
) -> ClassifiedMemoryStorePlugin:
    failed = find_failed_entry_point_for_plugin_id(catalog, plugin_id)
    if failed is not None:
        message = f"Memory store plugin {plugin_id!r} failed during entry-point loading"
        if failed.error is not None:
            raise MemoryStorePluginResolutionError(message) from failed.error
        raise MemoryStorePluginResolutionError(message)

    rejected = find_rejected_entry_point_for_plugin_id(catalog, plugin_id)
    if rejected is not None:
        raise MemoryStorePluginResolutionError(
            f"Memory store plugin {plugin_id!r} was rejected: {rejected.reason}"
        )

    record = catalog.index.get(plugin_id)
    if record is None:
        raise MemoryStorePluginResolutionError(
            f"Memory store plugin {plugin_id!r} is not available"
        )
    if record.kind is not expected_kind:
        raise MemoryStorePluginResolutionError(
            f"Memory store plugin {plugin_id!r} is {record.kind.value}, "
            f"expected {expected_kind.value}"
        )
    return record


def _validate_user_profile_store(store: object, *, plugin_id: str) -> UserProfileStore:
    if not isinstance(store, UserProfileStore):
        raise MemoryStorePluginResolutionError(
            f"Memory store plugin {plugin_id!r} returned invalid UserProfileStore"
        )
    return store


def _validate_session_storage(store: object, *, plugin_id: str) -> SessionStorage:
    if not isinstance(store, SessionStorage):
        raise MemoryStorePluginResolutionError(
            f"Memory store plugin {plugin_id!r} returned invalid SessionStorage"
        )
    return store


def materialize_user_profile_store(
    plugin_id: str,
    ctx: MemoryStoreMaterializationContext,
    *,
    catalog: MemoryStorePluginCatalog,
) -> UserProfileStore:
    """Materialize one external ``UserProfileStore`` from an explicit plugin id."""
    record = _select_classified_plugin(
        plugin_id,
        expected_kind=MemoryStorePluginKind.USER_PROFILE_STORE,
        catalog=catalog,
    )
    try:
        store = record.plugin_type.create_user_profile_store(
            **_user_profile_factory_kwargs(ctx)
        )
    except Exception as exc:
        raise MemoryStorePluginResolutionError(
            f"Memory store plugin {plugin_id!r} failed to materialize user profile store"
        ) from exc
    return _validate_user_profile_store(store, plugin_id=plugin_id)


def materialize_session_storage(
    plugin_id: str,
    ctx: MemoryStoreMaterializationContext,
    *,
    catalog: MemoryStorePluginCatalog,
) -> SessionStorage:
    """Materialize one external ``SessionStorage`` from an explicit plugin id."""
    record = _select_classified_plugin(
        plugin_id,
        expected_kind=MemoryStorePluginKind.SESSION_STORAGE,
        catalog=catalog,
    )
    try:
        store = record.plugin_type.create_session_storage(
            **_session_storage_factory_kwargs(ctx)
        )
    except Exception as exc:
        raise MemoryStorePluginResolutionError(
            f"Memory store plugin {plugin_id!r} failed to materialize session storage"
        ) from exc
    return _validate_session_storage(store, plugin_id=plugin_id)
