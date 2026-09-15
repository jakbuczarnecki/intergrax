# © Artur Czarnecki. All rights reserved.

"""Projection indexer from canonical memory records into entity/temporal store (MEM-ENT-7)."""

from __future__ import annotations

from intergrax.memory.contracts.entity_temporal_memory import (
    EntityMemoryIndexer,
    EntityMemoryScope,
    EntityRecord,
    EntityRelationRecord,
    EntityTemporalMemoryStore,
    EntityTypeRef,
    RelationTypeRef,
    entity_memory_entity_id_for_entry,
    entity_memory_relation_id_for_has_memory,
    entity_memory_user_entity_id,
)
from intergrax.memory.user_profile_memory import MemoryKind, UserProfileMemoryEntry


class DefaultEntityMemoryIndexer:
    """Derived projection indexer: LTM entries → entity graph records."""

    def __init__(self, store: EntityTemporalMemoryStore) -> None:
        self._store = store

    def index_memory_entry(
        self,
        scope: EntityMemoryScope,
        entry: object,
    ) -> None:
        if not isinstance(entry, UserProfileMemoryEntry):
            raise TypeError("entry must be UserProfileMemoryEntry")
        if entry.deleted:
            self.remove_memory_entry(scope, entry.entry_id)
            return
        content = (entry.content or "").strip()
        if not content:
            return

        user_id = (scope.user_id or "").strip()
        if not user_id:
            raise ValueError("scope.user_id is required for entity memory indexing")

        entity_type = (
            entry.kind.value if isinstance(entry.kind, MemoryKind) else str(entry.kind)
        )
        memory_entity_id = entity_memory_entity_id_for_entry(scope, entry.entry_id)
        existing = self._store.get_entity(scope, memory_entity_id)
        revision = entry.revision
        if existing is not None:
            revision = max(existing.revision, entry.revision)

        self._store.upsert_entity(
            scope,
            EntityRecord(
                entity_id=memory_entity_id,
                entity_type=EntityTypeRef(entity_type),
                canonical_name=content[:120],
                revision=revision,
                created_at=entry.created_at,
                updated_at=entry.updated_at,
                provenance=entry.provenance,
                trust=entry.trust,
                governance=entry.governance,
                evidence_refs=entry.evidence_refs,
                source_memory_id=entry.entry_id,
                source_memory_revision=entry.revision,
            ),
        )

        user_entity_id = entity_memory_user_entity_id(scope)
        user_existing = self._store.get_entity(scope, user_entity_id)
        self._store.upsert_entity(
            scope,
            EntityRecord(
                entity_id=user_entity_id,
                entity_type=EntityTypeRef("user"),
                canonical_name=user_id,
                revision=user_existing.revision if user_existing else 1,
                provenance=entry.provenance,
                trust=entry.trust,
                governance=entry.governance,
            ),
        )

        relation_id = entity_memory_relation_id_for_has_memory(entry.entry_id)
        self._store.upsert_relation(
            scope,
            EntityRelationRecord(
                relation_id=relation_id,
                source_entity_id=user_entity_id,
                target_entity_id=memory_entity_id,
                relation_type=RelationTypeRef("has_memory"),
                revision=entry.revision,
                valid_from=entry.valid_from,
                valid_until=entry.valid_until,
                provenance=entry.provenance,
                trust=entry.trust,
                governance=entry.governance,
                evidence_refs=entry.evidence_refs,
                lineage=entry.lineage,
                source_memory_id=entry.entry_id,
                source_memory_revision=entry.revision,
            ),
        )

    def remove_memory_entry(
        self,
        scope: EntityMemoryScope,
        memory_entry_id: str,
    ) -> None:
        self._store.delete_by_source_memory(scope, memory_entry_id)
