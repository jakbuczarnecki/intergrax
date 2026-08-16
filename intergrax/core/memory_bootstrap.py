# © Artur Czarnecki. All rights reserved.

"""Bootstrap optional memory store plugins (Phase MEM-3.2, MEM-VEC-3.1)."""

from __future__ import annotations

from collections.abc import Sequence

from dataclasses import dataclass

from intergrax.memory.resolver.classifier import MemoryStorePluginKind
from intergrax.memory.resolver.discovery import discover_classified_memory_store_plugins


@dataclass(frozen=True, slots=True)
class MemoryStoreBootstrapResult:
    user_profile_plugins: int
    session_storage_plugins: int
    session_turn_index_plugins: int


def _count_by_kind(plugins: Sequence[object], kind: MemoryStorePluginKind) -> int:
    return sum(1 for item in plugins if item.kind is kind)


def bootstrap_memory_stores(
    *,
    discover_entry_points: bool = True,
    user_profile_plugins: Sequence[type] = (),
    session_storage_plugins: Sequence[type] = (),
    session_turn_index_plugins: Sequence[type] = (),
) -> MemoryStoreBootstrapResult:
    """Discover memory store plugins from entry points and explicit classes."""
    explicit = (
        *user_profile_plugins,
        *session_storage_plugins,
        *session_turn_index_plugins,
    )
    discovered = discover_classified_memory_store_plugins(
        discover_entry_points=discover_entry_points,
        explicit_plugins=(),
    )
    explicit_classified = discover_classified_memory_store_plugins(
        discover_entry_points=False,
        explicit_plugins=explicit,
    )
    return MemoryStoreBootstrapResult(
        user_profile_plugins=_count_by_kind(discovered.plugins, MemoryStorePluginKind.USER_PROFILE_STORE)
        + _count_by_kind(explicit_classified.plugins, MemoryStorePluginKind.USER_PROFILE_STORE),
        session_storage_plugins=_count_by_kind(discovered.plugins, MemoryStorePluginKind.SESSION_STORAGE)
        + _count_by_kind(explicit_classified.plugins, MemoryStorePluginKind.SESSION_STORAGE),
        session_turn_index_plugins=_count_by_kind(discovered.plugins, MemoryStorePluginKind.SESSION_TURN_INDEX)
        + _count_by_kind(explicit_classified.plugins, MemoryStorePluginKind.SESSION_TURN_INDEX),
    )


def discover_session_turn_index_plugin_types() -> list[type]:
    """Return plugin classes classified as session turn index stores."""
    classified = discover_classified_memory_store_plugins(discover_entry_points=True)
    return [
        item.plugin_type
        for item in classified.plugins
        if item.kind is MemoryStorePluginKind.SESSION_TURN_INDEX
    ]
