# © Artur Czarnecki. All rights reserved.

"""Entity graph indexing from LTM entries (MEM-ENT-7 typed projection)."""

from __future__ import annotations

from intergrax.memory.contracts.entity_temporal_memory import (
    EntityMemoryIndexer,
    EntityMemoryScope,
    EntityTemporalMemoryStore,
)
from intergrax.memory.entity_memory_indexing import DefaultEntityMemoryIndexer
from intergrax.memory.user_profile_memory import UserProfileMemoryEntry


class EntityGraphMemoryService:
    """Indexes user memory entries via ``EntityMemoryIndexer`` (derived projection)."""

    def __init__(
        self,
        store: EntityTemporalMemoryStore,
        *,
        indexer: EntityMemoryIndexer | None = None,
    ) -> None:
        self._store = store
        self._indexer = indexer or DefaultEntityMemoryIndexer(store)

    @property
    def store(self) -> EntityTemporalMemoryStore:
        return self._store

    @property
    def indexer(self) -> EntityMemoryIndexer:
        return self._indexer

    def index_memory_entry(
        self,
        *,
        tenant_id: str,
        user_id: str,
        entry: UserProfileMemoryEntry,
    ) -> None:
        scope = EntityMemoryScope(tenant_id=tenant_id, user_id=user_id)
        self._indexer.index_memory_entry(scope, entry)

    def remove_memory_entry(
        self,
        *,
        tenant_id: str,
        user_id: str,
        memory_entry_id: str,
    ) -> None:
        scope = EntityMemoryScope(tenant_id=tenant_id, user_id=user_id)
        self._indexer.remove_memory_entry(scope, memory_entry_id)
