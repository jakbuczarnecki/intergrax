# © Artur Czarnecki. All rights reserved.

"""Default in-memory entity/temporal store plugin (MEM-ENT-7R)."""

from __future__ import annotations

from intergrax.memory.stores.in_memory_entity_temporal_memory_store import (
    InMemoryEntityTemporalMemoryStore,
)

DEFAULT_IN_MEMORY_ENTITY_TEMPORAL_PLUGIN_ID = "intergrax.in_memory_entity_temporal"


class InMemoryEntityTemporalMemoryStorePlugin:
    """Built-in default ``EntityTemporalMemoryStore`` provider."""

    @classmethod
    def plugin_id(cls) -> str:
        return DEFAULT_IN_MEMORY_ENTITY_TEMPORAL_PLUGIN_ID

    @classmethod
    def create_entity_temporal_memory_store(cls, **kwargs: object) -> InMemoryEntityTemporalMemoryStore:
        return InMemoryEntityTemporalMemoryStore()
