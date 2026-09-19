# © Artur Czarnecki. All rights reserved.

"""Built-in in-memory long-horizon memory store plugin (MEM-ENT-9)."""

from __future__ import annotations

from intergrax.memory.contracts.memory_store_creation_context import (
    LongHorizonMemoryStoreCreationContext,
)
from intergrax.memory.stores.in_memory_long_horizon_memory_store import (
    InMemoryLongHorizonMemoryStore,
)

DEFAULT_IN_MEMORY_LONG_HORIZON_PLUGIN_ID = "intergrax.in_memory_long_horizon"


class InMemoryLongHorizonMemoryStorePlugin:
    """Built-in default ``LongHorizonMemoryStore`` provider."""

    @classmethod
    def plugin_id(cls) -> str:
        return DEFAULT_IN_MEMORY_LONG_HORIZON_PLUGIN_ID

    @classmethod
    def create_long_horizon_memory_store(
        cls,
        context: LongHorizonMemoryStoreCreationContext,
    ) -> InMemoryLongHorizonMemoryStore:
        _ = context
        return InMemoryLongHorizonMemoryStore()
