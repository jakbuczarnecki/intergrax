# © Artur Czarnecki. All rights reserved.

"""Typed Memory store plugin classification and materialization (ENTERPRISE-5 / BLOCK D)."""

from intergrax.memory.resolver.classifier import (
    ClassifiedMemoryStorePlugin,
    MemoryStorePluginKind,
    classify_memory_store_plugin,
)
from intergrax.memory.resolver.discovery import discover_classified_memory_store_plugins
from intergrax.memory.resolver.errors import MemoryStorePluginResolutionError
from intergrax.memory.resolver.materialization import MemoryStoreMaterializationContext
from intergrax.memory.resolver.resolver import (
    materialize_session_storage,
    materialize_user_profile_store,
)

__all__ = [
    "ClassifiedMemoryStorePlugin",
    "MemoryStoreMaterializationContext",
    "MemoryStorePluginKind",
    "MemoryStorePluginResolutionError",
    "classify_memory_store_plugin",
    "discover_classified_memory_store_plugins",
    "materialize_session_storage",
    "materialize_user_profile_store",
]
