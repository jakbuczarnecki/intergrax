# © Artur Czarnecki. All rights reserved.

"""Built-in in-memory procedural memory store plugin (MEM-ENT-8)."""

from __future__ import annotations

from intergrax.memory.stores.in_memory_procedural_memory_store import (
    InMemoryProceduralMemoryStore,
)

DEFAULT_IN_MEMORY_PROCEDURAL_PLUGIN_ID = "intergrax.in_memory_procedural"


class InMemoryProceduralMemoryStorePlugin:
    """Built-in default ``ProcedureMemoryStore`` provider."""

    @classmethod
    def plugin_id(cls) -> str:
        return DEFAULT_IN_MEMORY_PROCEDURAL_PLUGIN_ID

    @classmethod
    def create_procedural_memory_store(cls, **kwargs: object) -> InMemoryProceduralMemoryStore:
        return InMemoryProceduralMemoryStore()
