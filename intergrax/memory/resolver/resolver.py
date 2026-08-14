# © Artur Czarnecki. All rights reserved.

"""Typed Memory store plugin materialization (ENTERPRISE-5 / BLOCK D)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from intergrax.llm.messages import ChatMessage
from intergrax.memory.resolver.classifier import (
    ClassifiedMemoryStorePlugin,
    MemoryStorePluginKind,
)
from intergrax.memory.resolver.discovery import (
    discover_classified_memory_store_plugins,
    index_classified_memory_store_plugins,
)
from intergrax.memory.resolver.errors import MemoryStorePluginResolutionError
from intergrax.memory.resolver.materialization import MemoryStoreMaterializationContext
from intergrax.memory.user_profile_memory import UserProfile
from intergrax.memory.user_profile_store import UserProfileStore
from intergrax.runtime.nexus.session.chat_session import ChatSession
from intergrax.runtime.nexus.session.session_storage import SessionStorage


@runtime_checkable
class _UserProfileStoreBoundary(Protocol):
    async def get_profile(self, *, tenant_id: str, user_id: str) -> UserProfile: ...

    async def save_profile(self, *, tenant_id: str, profile: UserProfile) -> None: ...

    async def delete_profile(self, *, tenant_id: str, user_id: str) -> None: ...


@runtime_checkable
class _SessionStorageBoundary(Protocol):
    async def get_session(self, *, tenant_id: str, session_id: str) -> ChatSession | None: ...

    async def create_session(
        self,
        *,
        tenant_id: str,
        session_id: str | None = None,
        user_id: str | None = None,
        workspace_id: str | None = None,
        metadata: dict | None = None,
    ) -> ChatSession: ...

    async def save_session(self, session: ChatSession) -> None: ...

    async def list_sessions_for_user(
        self,
        *,
        tenant_id: str,
        user_id: str,
        limit: int | None = None,
    ) -> list[ChatSession]: ...

    async def append_message(
        self,
        *,
        tenant_id: str,
        session_id: str,
        message: ChatMessage,
    ) -> ChatMessage: ...

    async def get_history(
        self,
        *,
        tenant_id: str,
        session_id: str,
        native_tools: bool = False,
    ) -> list[ChatMessage]: ...


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
    discover_entry_points: bool,
    explicit_plugins: Sequence[type] = (),
) -> ClassifiedMemoryStorePlugin:
    classified = discover_classified_memory_store_plugins(
        discover_entry_points=discover_entry_points,
        explicit_plugins=explicit_plugins,
    )
    index = index_classified_memory_store_plugins(classified)
    record = index.get(plugin_id)
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
    if not isinstance(store, _UserProfileStoreBoundary):
        raise MemoryStorePluginResolutionError(
            f"Memory store plugin {plugin_id!r} returned invalid UserProfileStore"
        )
    return store  # type: ignore[return-value]


def _validate_session_storage(store: object, *, plugin_id: str) -> SessionStorage:
    if not isinstance(store, _SessionStorageBoundary):
        raise MemoryStorePluginResolutionError(
            f"Memory store plugin {plugin_id!r} returned invalid SessionStorage"
        )
    return store  # type: ignore[return-value]


def materialize_user_profile_store(
    plugin_id: str,
    ctx: MemoryStoreMaterializationContext,
    *,
    discover_entry_points: bool,
    explicit_plugins: Sequence[type] = (),
) -> UserProfileStore:
    """Materialize one external ``UserProfileStore`` from an explicit plugin id."""
    record = _select_classified_plugin(
        plugin_id,
        expected_kind=MemoryStorePluginKind.USER_PROFILE_STORE,
        discover_entry_points=discover_entry_points,
        explicit_plugins=explicit_plugins,
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
    discover_entry_points: bool,
    explicit_plugins: Sequence[type] = (),
) -> SessionStorage:
    """Materialize one external ``SessionStorage`` from an explicit plugin id."""
    record = _select_classified_plugin(
        plugin_id,
        expected_kind=MemoryStorePluginKind.SESSION_STORAGE,
        discover_entry_points=discover_entry_points,
        explicit_plugins=explicit_plugins,
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
