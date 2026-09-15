# © Artur Czarnecki. All rights reserved.

"""Explicit no-op entity/temporal memory backend (MEM-ENT-7)."""

from __future__ import annotations

from intergrax.memory.contracts.entity_temporal_memory import (
    EntityMemoryScope,
    EntityRecord,
    EntityRelationQuery,
    EntityRelationRecord,
    EntityRelationResult,
)


class NoOpEntityTemporalMemoryStore:
    """Configured disabled backend; operations are inert but typed."""

    def upsert_entity(self, scope: EntityMemoryScope, record: EntityRecord) -> EntityRecord:
        return record

    def get_entity(self, scope: EntityMemoryScope, entity_id: str) -> EntityRecord | None:
        return None

    def upsert_relation(
        self,
        scope: EntityMemoryScope,
        record: EntityRelationRecord,
    ) -> EntityRelationRecord:
        return record

    def query_relations(
        self,
        scope: EntityMemoryScope,
        query: EntityRelationQuery,
    ) -> EntityRelationResult:
        return EntityRelationResult(relations=())

    def delete_by_source_memory(self, scope: EntityMemoryScope, source_memory_id: str) -> int:
        return 0
