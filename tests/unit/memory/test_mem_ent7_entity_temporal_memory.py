# © Artur Czarnecki. All rights reserved.

"""MEM-ENT-7 entity/temporal memory contracts and store."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from intergrax.memory.contracts.enterprise_memory_record import (
    MemoryProvenance,
    MemoryRecordSourceType,
    MemoryRecordTrust,
    MemoryTrustClass,
)
from intergrax.memory.contracts.entity_temporal_memory import (
    EntityMemoryScope,
    EntityRecord,
    EntityRelationDirection,
    EntityRelationQuery,
    EntityRelationRecord,
    EntityRelationResult,
    EntityTemporalMemoryNotFound,
    EntityTemporalMemoryStore,
    EntityTypeRef,
    RelationTypeRef,
    entity_memory_entity_id_for_entry,
    is_entity_relation_active_at,
    order_entity_relations_deterministic,
)
from intergrax.memory.entity_memory_indexing import DefaultEntityMemoryIndexer
from intergrax.memory.stores.in_memory_entity_temporal_memory_store import (
    InMemoryEntityTemporalMemoryStore,
)
from intergrax.memory.user_profile_memory import MemoryKind, UserProfileMemoryEntry

pytestmark = pytest.mark.gate


def _scope(tenant: str, user: str = "user-a") -> EntityMemoryScope:
    return EntityMemoryScope(tenant_id=tenant, user_id=user)


def _seed_entity(store: InMemoryEntityTemporalMemoryStore, scope: EntityMemoryScope, entity_id: str) -> None:
    store.upsert_entity(
        scope,
        EntityRecord(
            entity_id=entity_id,
            entity_type=EntityTypeRef("person"),
            canonical_name=entity_id,
            revision=1,
            created_at="2025-01-01T00:00:00+00:00",
        ),
    )


def test_temporal_relation_as_of_window() -> None:
    store = InMemoryEntityTemporalMemoryStore()
    scope = _scope("tenant-1")
    _seed_entity(store, scope, "a")
    _seed_entity(store, scope, "company")
    store.upsert_relation(
        scope,
        EntityRelationRecord(
            relation_id="rel:employment:1",
            source_entity_id="a",
            target_entity_id="company",
            relation_type=RelationTypeRef("employment"),
            revision=1,
            valid_from="2024-01-01T00:00:00+00:00",
            valid_until="2026-01-01T00:00:00+00:00",
        ),
    )
    present_2024 = store.query_relations(
        scope,
        EntityRelationQuery(
            entity_id="a",
            direction=EntityRelationDirection.OUTBOUND,
            as_of=datetime(2024, 6, 1, tzinfo=timezone.utc),
            limit=10,
        ),
    )
    absent_2026 = store.query_relations(
        scope,
        EntityRelationQuery(
            entity_id="a",
            direction=EntityRelationDirection.OUTBOUND,
            as_of=datetime(2026, 6, 1, tzinfo=timezone.utc),
            limit=10,
        ),
    )
    assert len(present_2024.relations) == 1
    assert absent_2026.relations == ()


def test_tenant_isolation() -> None:
    store = InMemoryEntityTemporalMemoryStore()
    scope_a = _scope("tenant-a", "shared-user")
    scope_b = _scope("tenant-b", "shared-user")
    entity_id = entity_memory_entity_id_for_entry(scope_a, "mem-1")
    store.upsert_entity(
        scope_a,
        EntityRecord(
            entity_id=entity_id,
            entity_type=EntityTypeRef("user_fact"),
            canonical_name="Tenant A fact",
            revision=1,
            created_at="2025-01-01T00:00:00+00:00",
        ),
    )
    assert store.get_entity(scope_a, entity_id) is not None
    assert store.get_entity(scope_b, entity_id) is None


def test_stable_public_identity_not_backend_internal() -> None:
    store = InMemoryEntityTemporalMemoryStore()
    scope = _scope("tenant-x")
    public_id = "ent:memory:tenant-x:user-a:abc123"
    store.upsert_entity(
        scope,
        EntityRecord(
            entity_id=public_id,
            entity_type=EntityTypeRef("semantic"),
            canonical_name="Fact",
            revision=1,
            created_at="2025-01-01T00:00:00+00:00",
        ),
    )
    loaded = store.get_entity(scope, public_id)
    assert loaded is not None
    assert loaded.entity_id == public_id


def test_relation_idempotent_upsert_bumps_revision() -> None:
    store = InMemoryEntityTemporalMemoryStore()
    scope = _scope("tenant-1")
    _seed_entity(store, scope, "src")
    _seed_entity(store, scope, "dst")
    first = store.upsert_relation(
        scope,
        EntityRelationRecord(
            relation_id="rel:dup",
            source_entity_id="src",
            target_entity_id="dst",
            relation_type=RelationTypeRef("knows"),
            revision=1,
        ),
    )
    second = store.upsert_relation(
        scope,
        EntityRelationRecord(
            relation_id="rel:dup",
            source_entity_id="src",
            target_entity_id="dst",
            relation_type=RelationTypeRef("knows"),
            revision=3,
        ),
    )
    assert first.relation_id == second.relation_id
    assert second.revision == 3
    all_rels = store.query_relations(
        scope,
        EntityRelationQuery(
            entity_id="src",
            direction=EntityRelationDirection.OUTBOUND,
            as_of=datetime(2025, 1, 1, tzinfo=timezone.utc),
            limit=10,
        ),
    )
    assert len(all_rels.relations) == 1


def test_referential_integrity_on_relation() -> None:
    store = InMemoryEntityTemporalMemoryStore()
    scope = _scope("tenant-1")
    with pytest.raises(EntityTemporalMemoryNotFound):
        store.upsert_relation(
            scope,
            EntityRelationRecord(
                relation_id="rel:missing",
                source_entity_id="missing",
                target_entity_id="dst",
                relation_type=RelationTypeRef("knows"),
                revision=1,
            ),
        )


def test_indexer_source_memory_linkage_and_revision_update() -> None:
    store = InMemoryEntityTemporalMemoryStore()
    indexer = DefaultEntityMemoryIndexer(store)
    scope = _scope("tenant-idx", "user-1")
    entry = UserProfileMemoryEntry(
        entry_id="mem-42",
        content="Likes deterministic tests",
        kind=MemoryKind.USER_FACT,
        revision=3,
        provenance=MemoryProvenance(source_type=MemoryRecordSourceType.SESSION_EXTRACTION),
        trust=MemoryRecordTrust(trust_class=MemoryTrustClass.MODEL_INFERENCE),
    )
    indexer.index_memory_entry(scope, entry)
    entity_id = entity_memory_entity_id_for_entry(scope, "mem-42")
    projected = store.get_entity(scope, entity_id)
    assert projected is not None
    assert projected.source_memory_id == "mem-42"
    assert projected.source_memory_revision == 3

    entry_v4 = UserProfileMemoryEntry(
        entry_id="mem-42",
        content="Likes deterministic tests v4",
        kind=MemoryKind.USER_FACT,
        revision=4,
        provenance=entry.provenance,
        trust=entry.trust,
    )
    indexer.index_memory_entry(scope, entry_v4)
    projected_v4 = store.get_entity(scope, entity_id)
    assert projected_v4 is not None
    assert projected_v4.source_memory_revision == 4
    assert projected_v4.revision >= 4


def test_delete_by_source_memory() -> None:
    store = InMemoryEntityTemporalMemoryStore()
    indexer = DefaultEntityMemoryIndexer(store)
    scope = _scope("tenant-del", "user-1")
    entry = UserProfileMemoryEntry(
        entry_id="mem-del",
        content="To be removed",
        kind=MemoryKind.USER_FACT,
        revision=1,
    )
    indexer.index_memory_entry(scope, entry)
    entity_id = entity_memory_entity_id_for_entry(scope, "mem-del")
    assert store.get_entity(scope, entity_id) is not None
    removed = store.delete_by_source_memory(scope, "mem-del")
    assert removed >= 1
    assert store.get_entity(scope, entity_id) is None


def test_provenance_roundtrip_on_entity() -> None:
    store = InMemoryEntityTemporalMemoryStore()
    scope = _scope("tenant-prov")
    provenance = MemoryProvenance(
        source_type=MemoryRecordSourceType.USER_EXPLICIT,
        source_id="src-1",
        run_id="run-9",
    )
    trust = MemoryRecordTrust(trust_class=MemoryTrustClass.USER_EXPLICIT, confidence=0.9)
    store.upsert_entity(
        scope,
        EntityRecord(
            entity_id="ent-1",
            entity_type=EntityTypeRef("preference"),
            canonical_name="Dark mode",
            revision=1,
            created_at="2025-01-01T00:00:00+00:00",
            provenance=provenance,
            trust=trust,
            evidence_refs=("evidence:abc",),
        ),
    )
    loaded = store.get_entity(scope, "ent-1")
    assert loaded is not None
    assert loaded.provenance == provenance
    assert loaded.trust == trust
    assert loaded.evidence_refs == ("evidence:abc",)


def test_deterministic_relation_ordering() -> None:
    rel_a = EntityRelationRecord(
        relation_id="rel:b",
        source_entity_id="x",
        target_entity_id="y",
        relation_type=RelationTypeRef("knows"),
        revision=1,
        valid_from="2024-01-01T00:00:00+00:00",
    )
    rel_b = EntityRelationRecord(
        relation_id="rel:a",
        source_entity_id="x",
        target_entity_id="z",
        relation_type=RelationTypeRef("knows"),
        revision=1,
        valid_from="2025-01-01T00:00:00+00:00",
    )
    ordered = order_entity_relations_deterministic((rel_a, rel_b))
    assert ordered[0].relation_id == "rel:a"


def test_query_limit_bounds_results() -> None:
    store = InMemoryEntityTemporalMemoryStore()
    scope = _scope("tenant-limit")
    _seed_entity(store, scope, "hub")
    for idx in range(5):
        target = f"n{idx}"
        _seed_entity(store, scope, target)
        store.upsert_relation(
            scope,
            EntityRelationRecord(
                relation_id=f"rel:{idx}",
                source_entity_id="hub",
                target_entity_id=target,
                relation_type=RelationTypeRef("link"),
                revision=1,
            ),
        )
    result = store.query_relations(
        scope,
        EntityRelationQuery(
            entity_id="hub",
            direction=EntityRelationDirection.OUTBOUND,
            as_of=datetime(2025, 1, 1, tzinfo=timezone.utc),
            limit=2,
        ),
    )
    assert len(result.relations) == 2


class _RecordingEntityStore:
    """Injectable fake backend for plugin-replacement contract tests."""

    def __init__(self) -> None:
        self.upserts: list[str] = []

    def upsert_entity(self, scope: EntityMemoryScope, record: EntityRecord) -> EntityRecord:
        self.upserts.append(record.entity_id)
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


def test_external_store_injectable_without_core_changes() -> None:
    fake = _RecordingEntityStore()
    assert isinstance(fake, EntityTemporalMemoryStore)
    indexer = DefaultEntityMemoryIndexer(fake)
    scope = _scope("tenant-fake", "u1")
    entry = UserProfileMemoryEntry(entry_id="e1", content="hello", kind=MemoryKind.USER_FACT, revision=1)
    indexer.index_memory_entry(scope, entry)
    assert fake.upserts


def test_is_entity_relation_active_at_fact_2025_only() -> None:
    relation = EntityRelationRecord(
        relation_id="rel:fact",
        source_entity_id="s",
        target_entity_id="t",
        relation_type=RelationTypeRef("fact"),
        revision=1,
        valid_from="2025-01-01T00:00:00+00:00",
        valid_until="2026-01-01T00:00:00+00:00",
    )
    assert is_entity_relation_active_at(
        relation,
        as_of=datetime(2025, 6, 1, tzinfo=timezone.utc),
    )
    assert not is_entity_relation_active_at(
        relation,
        as_of=datetime(2026, 6, 1, tzinfo=timezone.utc),
    )
