# © Artur Czarnecki. All rights reserved.

"""Default in-process entity/temporal memory store (MEM-ENT-7)."""

from __future__ import annotations

import threading
from dataclasses import replace

from intergrax.memory.contracts.entity_temporal_memory import (
    EntityMemoryScope,
    EntityRecord,
    EntityRelationQuery,
    EntityRelationRecord,
    EntityRelationResult,
    EntityTemporalMemoryNotFound,
    EntityTemporalMemoryStore,
    EntityTemporalMemoryViolation,
    entity_memory_entity_id_for_entry,
    entity_memory_relation_id_for_has_memory,
    is_entity_relation_active_at,
    order_entity_relations_deterministic,
)


def _source_revision_stale(stored_revision: int | None, incoming_revision: int | None) -> bool:
    if stored_revision is None or incoming_revision is None:
        return False
    return incoming_revision < stored_revision


def _entity_projection_equal(existing: EntityRecord, incoming: EntityRecord) -> bool:
    return (
        existing.entity_id == incoming.entity_id
        and existing.canonical_name == incoming.canonical_name
        and existing.entity_type == incoming.entity_type
        and existing.revision == incoming.revision
        and existing.created_at == incoming.created_at
        and existing.updated_at == incoming.updated_at
        and existing.aliases == incoming.aliases
        and existing.provenance == incoming.provenance
        and existing.trust == incoming.trust
        and existing.governance == incoming.governance
        and existing.evidence_refs == incoming.evidence_refs
        and existing.source_memory_id == incoming.source_memory_id
        and existing.source_memory_revision == incoming.source_memory_revision
    )


def _relation_projection_equal(
    existing: EntityRelationRecord,
    incoming: EntityRelationRecord,
) -> bool:
    return (
        existing.relation_id == incoming.relation_id
        and existing.source_entity_id == incoming.source_entity_id
        and existing.target_entity_id == incoming.target_entity_id
        and existing.relation_type == incoming.relation_type
        and existing.revision == incoming.revision
        and existing.valid_from == incoming.valid_from
        and existing.valid_until == incoming.valid_until
        and existing.provenance == incoming.provenance
        and existing.trust == incoming.trust
        and existing.governance == incoming.governance
        and existing.evidence_refs == incoming.evidence_refs
        and existing.lineage == incoming.lineage
        and existing.source_memory_id == incoming.source_memory_id
        and existing.source_memory_revision == incoming.source_memory_revision
    )


class InMemoryEntityTemporalMemoryStore:
    """Vendor-neutral in-memory ``EntityTemporalMemoryStore``."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._entities: dict[tuple[str, str], EntityRecord] = {}
        self._relations: dict[tuple[str, str], EntityRelationRecord] = {}

    def upsert_entity(self, scope: EntityMemoryScope, record: EntityRecord) -> EntityRecord:
        with self._lock:
            return self._upsert_entity_unlocked(scope, record)

    def _upsert_entity_unlocked(
        self,
        scope: EntityMemoryScope,
        record: EntityRecord,
    ) -> EntityRecord:
        key = (scope.tenant_id, record.entity_id)
        existing = self._entities.get(key)
        if existing is not None:
            if _source_revision_stale(
                existing.source_memory_revision,
                record.source_memory_revision,
            ):
                return existing
            if record.source_memory_id is not None and _entity_projection_equal(existing, record):
                return existing
            if record.source_memory_id is None:
                merged = replace(
                    record,
                    revision=max(existing.revision, record.revision),
                )
            else:
                merged = record
        else:
            merged = record
        self._entities[key] = merged
        return merged

    def get_entity(self, scope: EntityMemoryScope, entity_id: str) -> EntityRecord | None:
        with self._lock:
            return self._entities.get((scope.tenant_id, entity_id))

    def get_relation(self, scope: EntityMemoryScope, relation_id: str) -> EntityRelationRecord | None:
        with self._lock:
            return self._relations.get((scope.tenant_id, relation_id))

    def upsert_relation(
        self,
        scope: EntityMemoryScope,
        record: EntityRelationRecord,
    ) -> EntityRelationRecord:
        with self._lock:
            return self._upsert_relation_unlocked(scope, record)

    def _upsert_relation_unlocked(
        self,
        scope: EntityMemoryScope,
        record: EntityRelationRecord,
    ) -> EntityRelationRecord:
        self._ensure_endpoints_exist(scope, record)
        key = (scope.tenant_id, record.relation_id)
        existing = self._relations.get(key)
        if existing is not None:
            if _source_revision_stale(
                existing.source_memory_revision,
                record.source_memory_revision,
            ):
                return existing
            if record.source_memory_id is not None and _relation_projection_equal(existing, record):
                return existing
            if record.source_memory_id is None:
                merged = replace(
                    record,
                    revision=max(existing.revision, record.revision),
                )
            else:
                merged = record
        else:
            merged = record
        self._relations[key] = merged
        return merged

    def query_relations(
        self,
        scope: EntityMemoryScope,
        query: EntityRelationQuery,
    ) -> EntityRelationResult:
        with self._lock:
            return self._query_relations_unlocked(scope, query)

    def _query_relations_unlocked(
        self,
        scope: EntityMemoryScope,
        query: EntityRelationQuery,
    ) -> EntityRelationResult:
        as_of = query.as_of
        if as_of is None:
            raise EntityTemporalMemoryViolation("as_of is required for entity relation query")

        type_filter = {value.strip() for value in query.relation_types if value.strip()}
        matched: list[EntityRelationRecord] = []
        for (tenant_id, _rid), relation in self._relations.items():
            if tenant_id != scope.tenant_id:
                continue
            if type_filter and relation.relation_type.value not in type_filter:
                continue
            if query.direction.value == "outbound":
                if relation.source_entity_id != query.entity_id:
                    continue
            elif query.direction.value == "inbound":
                if relation.target_entity_id != query.entity_id:
                    continue
            else:
                if relation.source_entity_id != query.entity_id and relation.target_entity_id != query.entity_id:
                    continue
            if not is_entity_relation_active_at(relation, as_of=as_of):
                continue
            matched.append(relation)

        ordered = order_entity_relations_deterministic(tuple(matched))
        bounded = ordered[: query.limit]
        return EntityRelationResult(relations=bounded)

    def list_entities(self, scope: EntityMemoryScope) -> tuple[EntityRecord, ...]:
        with self._lock:
            return tuple(
                record
                for (tenant_id, _entity_id), record in self._entities.items()
                if tenant_id == scope.tenant_id
            )

    def delete_by_source_memory(self, scope: EntityMemoryScope, source_memory_id: str) -> int:
        with self._lock:
            return self._delete_by_source_memory_unlocked(scope, source_memory_id)

    def _delete_by_source_memory_unlocked(
        self,
        scope: EntityMemoryScope,
        source_memory_id: str,
    ) -> int:
        memory_id = (source_memory_id or "").strip()
        if not memory_id:
            return 0
        removed = 0
        relation_id = entity_memory_relation_id_for_has_memory(scope, memory_id)
        relation_key = (scope.tenant_id, relation_id)
        if relation_key in self._relations:
            del self._relations[relation_key]
            removed += 1

        memory_entity_id = entity_memory_entity_id_for_entry(scope, memory_id)
        entity_key = (scope.tenant_id, memory_entity_id)
        if entity_key in self._entities:
            del self._entities[entity_key]
            removed += 1
        return removed

    def _ensure_endpoints_exist(
        self,
        scope: EntityMemoryScope,
        record: EntityRelationRecord,
    ) -> None:
        for endpoint in (record.source_entity_id, record.target_entity_id):
            if self._entities.get((scope.tenant_id, endpoint)) is None:
                raise EntityTemporalMemoryNotFound(
                    f"entity endpoint not found for relation: {endpoint}"
                )
