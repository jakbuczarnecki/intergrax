# © Artur Czarnecki. All rights reserved.

"""Bootstrap optional memory store plugins (Phase MEM-3.2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from intergrax.core.plugins.discovery import EP_MEMORY_STORES, load_entry_point_plugins
from intergrax.memory.contracts.memory_store_plugin import (
    SessionStoragePlugin,
    UserProfileStorePlugin,
)


@dataclass(frozen=True, slots=True)
class MemoryStoreBootstrapResult:
    user_profile_plugins: int
    session_storage_plugins: int


def bootstrap_memory_stores(
    *,
    discover_entry_points: bool = True,
    user_profile_plugins: Sequence[type] = (),
    session_storage_plugins: Sequence[type] = (),
) -> MemoryStoreBootstrapResult:
    """Discover memory store plugins from entry points and explicit classes."""
    discovered = 0
    if discover_entry_points:
        discovered = len(load_entry_point_plugins(EP_MEMORY_STORES))
    explicit = len(user_profile_plugins) + len(session_storage_plugins)
    return MemoryStoreBootstrapResult(
        user_profile_plugins=discovered + len(user_profile_plugins),
        session_storage_plugins=discovered + len(session_storage_plugins),
    )
