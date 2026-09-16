# © Artur Czarnecki. All rights reserved.

"""Entity/temporal memory read orchestration with disclosure governance (MEM-ENT-10C)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from intergrax.contracts.agent_run import RequestIdentity
from intergrax.memory.contracts.entity_temporal_memory import (
    EntityEnumerationCapability,
    EntityMemoryScope,
    EntityRecord,
    EntityRelationQuery,
    EntityRelationRecord,
    EntityRelationResult,
    EntityTemporalMemoryStore,
    order_entity_relations_deterministic,
)
from intergrax.memory.memory_security_governance_service import MemorySecurityGovernanceService
from intergrax.memory.memory_specialized_disclosure_governance import (
    filter_memory_disclosure_candidates,
    memory_security_context_for_recall,
)
from intergrax.memory.memory_specialized_mutation_governance import (
    governance_snapshot_from_entity_record,
    governance_snapshot_from_entity_relation,
)

__all__ = ["EntityGraphDisclosureResult", "EntityTemporalMemoryService"]


@dataclass(frozen=True, slots=True)
class EntityGraphDisclosureResult:
    entities: tuple[EntityRecord, ...]
    relations: tuple[EntityRelationRecord, ...]


@dataclass(slots=True)
class EntityTemporalMemoryService:
    """Governed read surface over ``EntityTemporalMemoryStore``."""

    _store: EntityTemporalMemoryStore
    _security_governance: MemorySecurityGovernanceService

    def get_entity(
        self,
        identity: RequestIdentity,
        scope: EntityMemoryScope,
        entity_id: str,
        *,
        reference_time: datetime | None = None,
    ) -> EntityRecord | None:
        record = self._store.get_entity(scope, entity_id)
        if record is None:
            return None
        context = memory_security_context_for_recall(
            identity, scope, reference_time=reference_time
        )
        allowed = filter_memory_disclosure_candidates(
            self._security_governance,
            context,
            (record,),
            to_snapshot=governance_snapshot_from_entity_record,
        )
        return allowed[0] if allowed else None

    def query_relations(
        self,
        identity: RequestIdentity,
        scope: EntityMemoryScope,
        query: EntityRelationQuery,
    ) -> EntityRelationResult:
        raw = self._store.query_relations(scope, query)
        context = memory_security_context_for_recall(
            identity, scope, reference_time=query.as_of
        )
        filtered = filter_memory_disclosure_candidates(
            self._security_governance,
            context,
            raw.relations,
            to_snapshot=governance_snapshot_from_entity_relation,
        )
        ordered = order_entity_relations_deterministic(filtered)
        return EntityRelationResult(relations=ordered)

    def list_entities(
        self,
        identity: RequestIdentity,
        scope: EntityMemoryScope,
    ) -> tuple[EntityRecord, ...]:
        if not isinstance(self._store, EntityEnumerationCapability):
            return ()
        raw = self._store.list_entities(scope)
        context = memory_security_context_for_recall(identity, scope)
        return filter_memory_disclosure_candidates(
            self._security_governance,
            context,
            raw,
            to_snapshot=governance_snapshot_from_entity_record,
        )

    def disclose_entity_neighbors(
        self,
        identity: RequestIdentity,
        scope: EntityMemoryScope,
        entity_id: str,
        *,
        query: EntityRelationQuery,
    ) -> EntityGraphDisclosureResult:
        relation_result = self.query_relations(identity, scope, query)
        related_ids: set[str] = set()
        for relation in relation_result.relations:
            if relation.source_entity_id == entity_id:
                related_ids.add(relation.target_entity_id)
            if relation.target_entity_id == entity_id:
                related_ids.add(relation.source_entity_id)
        entities: list[EntityRecord] = []
        for related_id in sorted(related_ids):
            record = self.get_entity(
                identity,
                scope,
                related_id,
                reference_time=query.as_of,
            )
            if record is not None:
                entities.append(record)
        return EntityGraphDisclosureResult(
            entities=tuple(entities),
            relations=relation_result.relations,
        )
