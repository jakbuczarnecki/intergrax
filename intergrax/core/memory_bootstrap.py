# © Artur Czarnecki. All rights reserved.

"""Bootstrap optional memory store plugins (Phase MEM-3.2, MEM-VEC-3.1)."""

from __future__ import annotations

from intergrax.memory.resolver.classifier import MemoryStorePluginKind
from intergrax.memory.resolver.discovery import discover_classified_memory_store_plugins


def discover_session_turn_index_plugin_types(
    *,
    discover_entry_points: bool = True,
) -> list[type]:
    """Return plugin classes classified as session turn index stores."""
    classified = discover_classified_memory_store_plugins(
        discover_entry_points=discover_entry_points,
    )
    return [
        item.plugin_type
        for item in classified.plugins
        if item.kind is MemoryStorePluginKind.SESSION_TURN_INDEX
    ]
