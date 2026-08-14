# © Artur Czarnecki. All rights reserved.

"""Typed Memory store plugin classification (ENTERPRISE-5 / BLOCK D)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from intergrax.core.plugins.discovery import EntryPointSpec
from intergrax.memory.contracts.memory_store_plugin import (
    SessionStoragePlugin,
    UserProfileStorePlugin,
)
from intergrax.memory.contracts.session_turn_index import SessionTurnIndexStorePlugin


class MemoryStorePluginKind(StrEnum):
    USER_PROFILE_STORE = "user_profile_store"
    SESSION_STORAGE = "session_storage"
    SESSION_TURN_INDEX = "session_turn_index"


@dataclass(frozen=True, slots=True)
class ClassifiedMemoryStorePlugin:
    """One typed Memory store plugin candidate."""

    plugin_id: str
    kind: MemoryStorePluginKind
    plugin_type: type
    entry_point_name: str | None = None
    entry_point_spec: EntryPointSpec | None = None


def classify_memory_store_plugin(plugin_type: type) -> MemoryStorePluginKind | None:
    """Classify ``plugin_type`` using canonical typed Protocol conformance."""
    if isinstance(plugin_type, SessionTurnIndexStorePlugin):
        return MemoryStorePluginKind.SESSION_TURN_INDEX
    if isinstance(plugin_type, UserProfileStorePlugin):
        return MemoryStorePluginKind.USER_PROFILE_STORE
    if isinstance(plugin_type, SessionStoragePlugin):
        return MemoryStorePluginKind.SESSION_STORAGE
    return None


def classify_memory_store_plugin_record(
    plugin_type: type,
    *,
    entry_point_name: str | None = None,
    entry_point_spec: EntryPointSpec | None = None,
) -> ClassifiedMemoryStorePlugin | None:
    """Return a classified record or ``None`` when the target is unsupported."""
    kind = classify_memory_store_plugin(plugin_type)
    if kind is None:
        return None
    plugin_id = plugin_type.plugin_id()
    if not isinstance(plugin_id, str) or not plugin_id.strip():
        return None
    return ClassifiedMemoryStorePlugin(
        plugin_id=plugin_id.strip(),
        kind=kind,
        plugin_type=plugin_type,
        entry_point_name=entry_point_name,
        entry_point_spec=entry_point_spec,
    )
