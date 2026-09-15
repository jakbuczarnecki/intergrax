# © Artur Czarnecki. All rights reserved.

"""Canonical entity and temporal memory contracts (MEM-ENT-7)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Protocol, runtime_checkable

from intergrax.memory.contracts.enterprise_memory_record import (
    MemoryProvenance,
    MemoryRecordGovernance,
    MemoryRecordLineage,
    MemoryRecordTrust,
    parse_memory_record_timestamp,
)

__all__ = [
    "EntityMemoryIndexer",
    "EntityMemoryScope",
    "EntityRecord",
    "EntityRelationDirection",
    "EntityRelationQuery",
    "EntityRelationRecord",
    "EntityRelationResult",
    "EntityTemporalMemoryError",
    "EntityTemporalMemoryNotFound",
    "EntityTemporalMemoryStore",
    "EntityTemporalMemoryViolation",
    "EntityTypeRef",
    "RelationTypeRef",
    "entity_memory_entity_id_for_entry",
    "entity_memory_relation_id_for_has_memory",
    "entity_memory_user_entity_id",
    "is_entity_relation_active_at",
    "order_entity_relations_deterministic",
]


class EntityTemporalMemoryError(RuntimeError):
    """Base error for entity/temporal memory operations."""


class EntityTemporalMemoryNotFound(LookupError, EntityTemporalMemoryError):
    """Entity or relation identity is not present in scope."""


class EntityTemporalMemoryViolation(ValueError, EntityTemporalMemoryError):
    """Record or query violates entity/temporal invariants."""


@dataclass(frozen=True, slots=True)
class EntityMemoryScope:
    """Tenant-isolated scope for entity/temporal memory (canonical tenant authority)."""

    tenant_id: str
    user_id: str | None = None
    workspace_id: str | None = None

    def __post_init__(self) -> None:
        tenant = (self.tenant_id or "").strip()
        if not tenant:
            raise EntityTemporalMemoryViolation("tenant_id must be non-empty")


@dataclass(frozen=True, slots=True)
class EntityTypeRef:
    value: str

    def __post_init__(self) -> None:
        if not (self.value or "").strip():
            raise EntityTemporalMemoryViolation("entity_type must be non-empty")


@dataclass(frozen=True, slots=True)
class RelationTypeRef:
    value: str

    def __post_init__(self) -> None:
        if not (self.value or "").strip():
            raise EntityTemporalMemoryViolation("relation_type must be non-empty")


@dataclass(frozen=True, slots=True)
class EntityRecord:
    entity_id: str
    entity_type: EntityTypeRef
    canonical_name: str
    revision: int = 1
    created_at: str = ""
    updated_at: str | None = None
    aliases: tuple[str, ...] = ()
    provenance: MemoryProvenance = field(default_factory=MemoryProvenance)
    trust: MemoryRecordTrust = field(default_factory=MemoryRecordTrust)
    governance: MemoryRecordGovernance = field(default_factory=MemoryRecordGovernance)
    evidence_refs: tuple[str, ...] = ()
    source_memory_id: str | None = None
    source_memory_revision: int | None = None

    def __post_init__(self) -> None:
        if not (self.entity_id or "").strip():
            raise EntityTemporalMemoryViolation("entity_id must be non-empty")
        if self.revision < 1:
            raise EntityTemporalMemoryViolation("revision must be >= 1")
        if not (self.canonical_name or "").strip():
            raise EntityTemporalMemoryViolation("canonical_name must be non-empty")


@dataclass(frozen=True, slots=True)
class EntityRelationRecord:
    relation_id: str
    source_entity_id: str
    target_entity_id: str
    relation_type: RelationTypeRef
    revision: int = 1
    valid_from: str | None = None
    valid_until: str | None = None
    provenance: MemoryProvenance = field(default_factory=MemoryProvenance)
    trust: MemoryRecordTrust = field(default_factory=MemoryRecordTrust)
    governance: MemoryRecordGovernance = field(default_factory=MemoryRecordGovernance)
    evidence_refs: tuple[str, ...] = ()
    lineage: MemoryRecordLineage = field(default_factory=MemoryRecordLineage)
    source_memory_id: str | None = None
    source_memory_revision: int | None = None

    def __post_init__(self) -> None:
        if not (self.relation_id or "").strip():
            raise EntityTemporalMemoryViolation("relation_id must be non-empty")
        if self.revision < 1:
            raise EntityTemporalMemoryViolation("revision must be >= 1")
        if not (self.source_entity_id or "").strip() or not (self.target_entity_id or "").strip():
            raise EntityTemporalMemoryViolation("relation endpoints must be non-empty")
        if self.valid_from and self.valid_until:
            from_dt = parse_memory_record_timestamp("valid_from", self.valid_from)
            until_dt = parse_memory_record_timestamp("valid_until", self.valid_until)
            if from_dt.tzinfo is not None and until_dt.tzinfo is not None:
                if from_dt > until_dt:
                    raise EntityTemporalMemoryViolation("valid_from must not be after valid_until")
            elif from_dt.tzinfo is None and until_dt.tzinfo is None:
                if from_dt > until_dt:
                    raise EntityTemporalMemoryViolation("valid_from must not be after valid_until")


class EntityRelationDirection(str, Enum):
    OUTBOUND = "outbound"
    INBOUND = "inbound"
    BOTH = "both"


@dataclass(frozen=True, slots=True)
class EntityRelationQuery:
    entity_id: str
    relation_types: tuple[str, ...] = ()
    direction: EntityRelationDirection = EntityRelationDirection.BOTH
    as_of: datetime | None = None
    limit: int = 50

    def __post_init__(self) -> None:
        if not (self.entity_id or "").strip():
            raise EntityTemporalMemoryViolation("entity_id must be non-empty")
        if self.limit < 1:
            raise EntityTemporalMemoryViolation("limit must be >= 1")


@dataclass(frozen=True, slots=True)
class EntityRelationResult:
    relations: tuple[EntityRelationRecord, ...]


def is_entity_relation_active_at(
    relation: EntityRelationRecord,
    *,
    as_of: datetime,
) -> bool:
    """Return True when relation is active at ``as_of`` (exclusive valid_until)."""
    if as_of.tzinfo is None and (
        (relation.valid_from and parse_memory_record_timestamp("valid_from", relation.valid_from).tzinfo)
        or (relation.valid_until and parse_memory_record_timestamp("valid_until", relation.valid_until).tzinfo)
    ):
        raise EntityTemporalMemoryViolation(
            "as_of must be timezone-aware when relation bounds are aware"
        )
    if relation.valid_from:
        from_dt = parse_memory_record_timestamp("valid_from", relation.valid_from)
        if as_of < from_dt:
            return False
    if relation.valid_until:
        until_dt = parse_memory_record_timestamp("valid_until", relation.valid_until)
        if as_of >= until_dt:
            return False
    return True


def order_entity_relations_deterministic(
    relations: tuple[EntityRelationRecord, ...],
) -> tuple[EntityRelationRecord, ...]:
    def sort_key(item: EntityRelationRecord) -> tuple[float, str]:
        if item.valid_from:
            from_dt = parse_memory_record_timestamp("valid_from", item.valid_from)
            if from_dt.tzinfo is None:
                primary = from_dt.timestamp()
            else:
                primary = from_dt.timestamp()
        else:
            primary = float("-inf")
        return (-primary, item.relation_id)

    return tuple(sorted(relations, key=sort_key))


def entity_memory_user_entity_id(scope: EntityMemoryScope) -> str:
    user = (scope.user_id or "").strip()
    if not user:
        raise EntityTemporalMemoryViolation("user_id required for user entity id")
    return f"ent:user:{scope.tenant_id}:{user}"


def entity_memory_entity_id_for_entry(scope: EntityMemoryScope, memory_entry_id: str) -> str:
    entry_id = (memory_entry_id or "").strip()
    if not entry_id:
        raise EntityTemporalMemoryViolation("memory_entry_id must be non-empty")
    user = (scope.user_id or "").strip()
    if not user:
        raise EntityTemporalMemoryViolation("user_id required for memory-derived entity id")
    return f"ent:memory:{scope.tenant_id}:{user}:{entry_id}"


def entity_memory_relation_id_for_has_memory(source_memory_id: str) -> str:
    memory_id = (source_memory_id or "").strip()
    if not memory_id:
        raise EntityTemporalMemoryViolation("source_memory_id must be non-empty")
    return f"rel:has_memory:{memory_id}"


@runtime_checkable
class EntityTemporalMemoryStore(Protocol):
    """Pluggable persistence and query for entity/temporal memory."""

    def upsert_entity(self, scope: EntityMemoryScope, record: EntityRecord) -> EntityRecord: ...

    def get_entity(self, scope: EntityMemoryScope, entity_id: str) -> EntityRecord | None: ...

    def upsert_relation(
        self,
        scope: EntityMemoryScope,
        record: EntityRelationRecord,
    ) -> EntityRelationRecord: ...

    def query_relations(
        self,
        scope: EntityMemoryScope,
        query: EntityRelationQuery,
    ) -> EntityRelationResult: ...

    def delete_by_source_memory(
        self,
        scope: EntityMemoryScope,
        source_memory_id: str,
    ) -> int: ...


@runtime_checkable
class EntityMemoryIndexer(Protocol):
    """Indexes canonical memory records into entity/temporal projection."""

    def index_memory_entry(
        self,
        scope: EntityMemoryScope,
        entry: object,
    ) -> None: ...

    def remove_memory_entry(
        self,
        scope: EntityMemoryScope,
        memory_entry_id: str,
    ) -> None: ...
