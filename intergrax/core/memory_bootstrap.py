# © Artur Czarnecki. All rights reserved.

"""Bootstrap optional memory store plugins (Phase MEM-3.2, MEM-VEC-3.1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from intergrax.core.plugins.discovery import EP_MEMORY_STORES, load_entry_point_plugins


@dataclass(frozen=True, slots=True)
class MemoryStoreBootstrapResult:
    user_profile_plugins: int
    session_storage_plugins: int
    session_turn_index_plugins: int


def _is_session_turn_index_plugin(plugin_type: type) -> bool:
    return hasattr(plugin_type, "create_session_turn_index") and callable(
        plugin_type.create_session_turn_index
    )


def _is_user_profile_store_plugin(plugin_type: type) -> bool:
    return hasattr(plugin_type, "create_user_profile_store") and callable(
        plugin_type.create_user_profile_store
    )


def _is_session_storage_plugin(plugin_type: type) -> bool:
    return hasattr(plugin_type, "create_session_storage") and callable(
        plugin_type.create_session_storage
    )


def bootstrap_memory_stores(
    *,
    discover_entry_points: bool = True,
    user_profile_plugins: Sequence[type] = (),
    session_storage_plugins: Sequence[type] = (),
    session_turn_index_plugins: Sequence[type] = (),
) -> MemoryStoreBootstrapResult:
    """Discover memory store plugins from entry points and explicit classes."""
    discovered_user = 0
    discovered_session = 0
    discovered_turn_index = 0
    if discover_entry_points:
        for loaded in load_entry_point_plugins(EP_MEMORY_STORES):
            plugin_type = loaded.plugin_type
            if _is_session_turn_index_plugin(plugin_type):
                discovered_turn_index += 1
            elif _is_user_profile_store_plugin(plugin_type):
                discovered_user += 1
            elif _is_session_storage_plugin(plugin_type):
                discovered_session += 1
    return MemoryStoreBootstrapResult(
        user_profile_plugins=discovered_user + len(user_profile_plugins),
        session_storage_plugins=discovered_session + len(session_storage_plugins),
        session_turn_index_plugins=discovered_turn_index + len(session_turn_index_plugins),
    )


def discover_session_turn_index_plugin_types() -> list[type]:
    """Return plugin classes exposing ``create_session_turn_index``."""
    explicit: list[type] = []
    for loaded in load_entry_point_plugins(EP_MEMORY_STORES):
        plugin_type = loaded.plugin_type
        if _is_session_turn_index_plugin(plugin_type):
            explicit.append(plugin_type)
    return explicit
