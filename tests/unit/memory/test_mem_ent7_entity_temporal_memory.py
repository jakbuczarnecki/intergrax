# © Artur Czarnecki. All rights reserved.

"""MEM-ENT-7 entity/temporal memory contracts and store."""

from __future__ import annotations

import inspect
from datetime import datetime, timezone

import pytest

from intergrax.contracts.agent_run import RequestIdentity
from intergrax.memory.memory_security_governance_service import build_default_memory_security_governance_service

from intergrax.applications._shared.entity_graph_wiring import resolve_entity_temporal_memory_store
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    MemoryProfile,
)
from intergrax.memory.resolver.discovery import (
    MemoryStorePluginCatalog,
    discover_classified_memory_store_plugins,
)
from intergrax.memory.resolver.errors import MemoryStorePluginResolutionError
from intergrax.memory.resolver.materialization import MemoryStoreMaterializationContext
from intergrax.memory.resolver.resolver import materialize_entity_temporal_memory_store
from intergrax.memory.stores.in_memory_entity_temporal_memory_plugin import (
    DEFAULT_IN_MEMORY_ENTITY_TEMPORAL_PLUGIN_ID,
    InMemoryEntityTemporalMemoryStorePlugin,
)

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
    EntityTemporalMemoryViolation,
    EntityTypeRef,
    RelationTypeRef,
    entity_memory_entity_id_for_entry,
    entity_memory_relation_id_for_has_memory,
    entity_memory_source_projection_key,
    entity_memory_user_entity_id,
    is_entity_relation_active_at,
    order_entity_relations_deterministic,
)
from intergrax.memory.entity_memory_indexing import DefaultEntityMemoryIndexer
from intergrax.memory.stores.in_memory_entity_temporal_memory_store import (
    InMemoryEntityTemporalMemoryStore,
)
from intergrax.memory.user_profile_memory import MemoryKind, UserProfileMemoryEntry

pytestmark = pytest.mark.gate


def _identity(scope: EntityMemoryScope) -> RequestIdentity:
    return RequestIdentity(tenant_id=scope.tenant_id, user_id=scope.user_id or "user-a")



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
    indexer = DefaultEntityMemoryIndexer(store, security_governance=build_default_memory_security_governance_service())
    scope = _scope("tenant-idx", "user-1")
    entry = UserProfileMemoryEntry(
        entry_id="mem-42",
        content="Likes deterministic tests",
        kind=MemoryKind.USER_FACT,
        revision=3,
        provenance=MemoryProvenance(source_type=MemoryRecordSourceType.SESSION_EXTRACTION),
        trust=MemoryRecordTrust(trust_class=MemoryTrustClass.MODEL_INFERENCE),
    )
    indexer.index_memory_entry(_identity(scope), scope, entry)
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
    indexer.index_memory_entry(_identity(scope), scope, entry_v4)
    projected_v4 = store.get_entity(scope, entity_id)
    assert projected_v4 is not None
    assert projected_v4.source_memory_revision == 4
    assert projected_v4.revision >= 4


def test_delete_by_source_memory() -> None:
    store = InMemoryEntityTemporalMemoryStore()
    indexer = DefaultEntityMemoryIndexer(store, security_governance=build_default_memory_security_governance_service())
    scope = _scope("tenant-del", "user-1")
    entry = UserProfileMemoryEntry(
        entry_id="mem-del",
        content="To be removed",
        kind=MemoryKind.USER_FACT,
        revision=1,
    )
    indexer.index_memory_entry(_identity(scope), scope, entry)
    entity_id = entity_memory_entity_id_for_entry(scope, "mem-del")
    user_id = entity_memory_user_entity_id(scope)
    rel_id = entity_memory_relation_id_for_has_memory(scope, "mem-del")
    assert store.get_entity(scope, entity_id) is not None
    removed = store.delete_by_source_memory(scope, "mem-del")
    assert removed == 2
    assert store.get_entity(scope, entity_id) is None
    assert not _relation_exists(store, scope, user_entity_id=user_id, relation_id=rel_id)
    assert store.get_entity(scope, user_id) is not None


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
        self.deletes: list[tuple[EntityMemoryScope, str]] = []

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
        self.deletes.append((scope, source_memory_id))
        return 0


def test_external_store_injectable_without_core_changes() -> None:
    fake = _RecordingEntityStore()
    assert isinstance(fake, EntityTemporalMemoryStore)
    indexer = DefaultEntityMemoryIndexer(fake, security_governance=build_default_memory_security_governance_service())
    scope = _scope("tenant-fake", "u1")
    entry = UserProfileMemoryEntry(entry_id="e1", content="hello", kind=MemoryKind.USER_FACT, revision=1)
    indexer.index_memory_entry(_identity(scope), scope, entry)
    assert fake.upserts
    indexer.remove_memory_entry(_identity(scope), scope, "e1")
    assert fake.deletes == [(scope, "e1")]


def test_external_store_can_derive_scoped_delete_from_contract_helpers() -> None:
    scope = EntityMemoryScope(tenant_id="T", user_id="A", workspace_id="ws")
    source_id = "mem-1"
    projection_key = entity_memory_source_projection_key(scope, source_id)
    relation_id = entity_memory_relation_id_for_has_memory(scope, source_id)
    entity_id = entity_memory_entity_id_for_entry(scope, source_id)
    assert relation_id == f"rel:has_memory:{projection_key}"
    assert entity_id == f"ent:memory:{projection_key}"
    assert "2:ws" in projection_key


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


def test_mixed_valid_from_valid_until_rejected() -> None:
    with pytest.raises(EntityTemporalMemoryViolation):
        EntityRelationRecord(
            relation_id="rel:mix",
            source_entity_id="s",
            target_entity_id="t",
            relation_type=RelationTypeRef("knows"),
            revision=1,
            valid_from="2025-01-01T00:00:00+00:00",
            valid_until="2026-01-01T00:00:00",
        )


def test_as_of_awareness_must_match_relation_bounds() -> None:
    aware_relation = EntityRelationRecord(
        relation_id="rel:aware",
        source_entity_id="s",
        target_entity_id="t",
        relation_type=RelationTypeRef("knows"),
        revision=1,
        valid_from="2025-01-01T00:00:00+00:00",
    )
    with pytest.raises(EntityTemporalMemoryViolation):
        is_entity_relation_active_at(aware_relation, as_of=datetime(2025, 6, 1))
    naive_relation = EntityRelationRecord(
        relation_id="rel:naive",
        source_entity_id="s",
        target_entity_id="t",
        relation_type=RelationTypeRef("knows"),
        revision=1,
        valid_from="2025-01-01T00:00:00",
    )
    with pytest.raises(EntityTemporalMemoryViolation):
        is_entity_relation_active_at(
            naive_relation,
            as_of=datetime(2025, 6, 1, tzinfo=timezone.utc),
        )


def test_naive_ordering_year_boundary() -> None:
    older = EntityRelationRecord(
        relation_id="rel:older",
        source_entity_id="s",
        target_entity_id="t",
        relation_type=RelationTypeRef("knows"),
        revision=1,
        valid_from="2025-12-31T00:00:00",
    )
    newer = EntityRelationRecord(
        relation_id="rel:newer",
        source_entity_id="s",
        target_entity_id="t",
        relation_type=RelationTypeRef("knows"),
        revision=1,
        valid_from="2026-01-01T00:00:00",
    )
    ordered = order_entity_relations_deterministic((older, newer))
    assert ordered[0].relation_id == "rel:newer"


def test_equal_valid_from_orders_by_relation_id() -> None:
    rel_b = EntityRelationRecord(
        relation_id="rel:b",
        source_entity_id="s",
        target_entity_id="t",
        relation_type=RelationTypeRef("knows"),
        revision=1,
        valid_from="2025-01-01T00:00:00+00:00",
    )
    rel_a = EntityRelationRecord(
        relation_id="rel:a",
        source_entity_id="s",
        target_entity_id="t",
        relation_type=RelationTypeRef("knows"),
        revision=1,
        valid_from="2025-01-01T00:00:00+00:00",
    )
    ordered = order_entity_relations_deterministic((rel_b, rel_a))
    assert ordered[0].relation_id == "rel:a"


def test_stale_source_revision_preserves_projection_content() -> None:
    store = InMemoryEntityTemporalMemoryStore()
    indexer = DefaultEntityMemoryIndexer(store, security_governance=build_default_memory_security_governance_service())
    scope = _scope("tenant-stale", "user-1")
    provenance = MemoryProvenance(source_type=MemoryRecordSourceType.SESSION_EXTRACTION)
    indexer.index_memory_entry(
        _identity(scope),
        scope,
        UserProfileMemoryEntry(
            entry_id="mem-stale",
            content="version four",
            kind=MemoryKind.USER_FACT,
            revision=4,
            provenance=provenance,
        ),
    )
    indexer.index_memory_entry(
        _identity(scope),
        scope,
        UserProfileMemoryEntry(
            entry_id="mem-stale",
            content="version three",
            kind=MemoryKind.USER_FACT,
            revision=3,
            provenance=provenance,
        ),
    )
    entity_id = entity_memory_entity_id_for_entry(scope, "mem-stale")
    projected = store.get_entity(scope, entity_id)
    assert projected is not None
    assert projected.source_memory_revision == 4
    assert projected.canonical_name == "version four"


def _memory_entry(entry_id: str, *, revision: int = 1, content: str = "fact") -> UserProfileMemoryEntry:
    return UserProfileMemoryEntry(
        entry_id=entry_id,
        content=content,
        kind=MemoryKind.USER_FACT,
        revision=revision,
        provenance=MemoryProvenance(source_type=MemoryRecordSourceType.SESSION_EXTRACTION),
        trust=MemoryRecordTrust(trust_class=MemoryTrustClass.MODEL_INFERENCE),
    )


def test_entity_memory_source_projection_key_stable_and_scoped() -> None:
    scope_a = EntityMemoryScope(tenant_id="T", user_id="A")
    scope_b = EntityMemoryScope(tenant_id="T", user_id="B")
    scope_ws1 = EntityMemoryScope(tenant_id="T", user_id="A", workspace_id="W1")
    scope_ws2 = EntityMemoryScope(tenant_id="T", user_id="A", workspace_id="W2")
    scope_t2 = EntityMemoryScope(tenant_id="T2", user_id="A")
    assert entity_memory_source_projection_key(scope_a, "M") == entity_memory_source_projection_key(
        scope_a,
        "M",
    )
    assert entity_memory_source_projection_key(scope_a, "M") != entity_memory_source_projection_key(
        scope_b,
        "M",
    )
    assert entity_memory_source_projection_key(scope_ws1, "M") != entity_memory_source_projection_key(
        scope_ws2,
        "M",
    )
    assert entity_memory_source_projection_key(scope_a, "M") != entity_memory_source_projection_key(
        scope_t2,
        "M",
    )


def test_relation_id_scoped_by_user_under_same_tenant() -> None:
    scope_a = EntityMemoryScope(tenant_id="T", user_id="A")
    scope_b = EntityMemoryScope(tenant_id="T", user_id="B")
    rel_a = entity_memory_relation_id_for_has_memory(scope_a, "M")
    rel_b = entity_memory_relation_id_for_has_memory(scope_b, "M")
    assert rel_a != rel_b
    assert rel_a == f"rel:has_memory:{entity_memory_source_projection_key(scope_a, 'M')}"


def _relation_exists(
    store: InMemoryEntityTemporalMemoryStore,
    scope: EntityMemoryScope,
    *,
    user_entity_id: str,
    relation_id: str,
) -> bool:
    result = store.query_relations(
        scope,
        EntityRelationQuery(
            entity_id=user_entity_id,
            direction=EntityRelationDirection.OUTBOUND,
            as_of=datetime(2025, 1, 1, tzinfo=timezone.utc),
            limit=50,
        ),
    )
    return any(rel.relation_id == relation_id for rel in result.relations)


def test_same_tenant_same_source_id_different_users_both_projections_exist() -> None:
    store = InMemoryEntityTemporalMemoryStore()
    indexer = DefaultEntityMemoryIndexer(store, security_governance=build_default_memory_security_governance_service())
    scope_a = EntityMemoryScope(tenant_id="T", user_id="A")
    scope_b = EntityMemoryScope(tenant_id="T", user_id="B")
    source_id = "M"
    indexer.index_memory_entry(_identity(scope_a), scope_a, _memory_entry(source_id, revision=4))
    indexer.index_memory_entry(_identity(scope_b), scope_b, _memory_entry(source_id, revision=3))

    entity_a = entity_memory_entity_id_for_entry(scope_a, source_id)
    entity_b = entity_memory_entity_id_for_entry(scope_b, source_id)
    rel_a = entity_memory_relation_id_for_has_memory(scope_a, source_id)
    rel_b = entity_memory_relation_id_for_has_memory(scope_b, source_id)
    user_a = entity_memory_user_entity_id(scope_a)
    user_b = entity_memory_user_entity_id(scope_b)

    assert entity_a != entity_b
    assert rel_a != rel_b
    assert store.get_entity(scope_a, entity_a) is not None
    assert store.get_entity(scope_b, entity_b) is not None
    assert _relation_exists(store, scope_a, user_entity_id=user_a, relation_id=rel_a)
    assert _relation_exists(store, scope_b, user_entity_id=user_b, relation_id=rel_b)


def test_delete_by_source_memory_removes_only_scoped_user_a() -> None:
    store = InMemoryEntityTemporalMemoryStore()
    indexer = DefaultEntityMemoryIndexer(store, security_governance=build_default_memory_security_governance_service())
    scope_a = EntityMemoryScope(tenant_id="T", user_id="A")
    scope_b = EntityMemoryScope(tenant_id="T", user_id="B")
    source_id = "M"
    indexer.index_memory_entry(_identity(scope_a), scope_a, _memory_entry(source_id, revision=5))
    indexer.index_memory_entry(_identity(scope_b), scope_b, _memory_entry(source_id, revision=3))

    entity_a = entity_memory_entity_id_for_entry(scope_a, source_id)
    entity_b = entity_memory_entity_id_for_entry(scope_b, source_id)
    rel_a = entity_memory_relation_id_for_has_memory(scope_a, source_id)
    rel_b = entity_memory_relation_id_for_has_memory(scope_b, source_id)
    user_a = entity_memory_user_entity_id(scope_a)
    user_b = entity_memory_user_entity_id(scope_b)

    removed = store.delete_by_source_memory(scope_a, source_id)
    assert removed == 2
    assert store.get_entity(scope_a, entity_a) is None
    assert not _relation_exists(store, scope_a, user_entity_id=user_a, relation_id=rel_a)
    assert store.get_entity(scope_b, entity_b) is not None
    assert _relation_exists(store, scope_b, user_entity_id=user_b, relation_id=rel_b)
    assert store.get_entity(scope_a, user_a) is not None
    assert store.get_entity(scope_b, user_b) is not None


def test_delete_by_source_memory_removes_only_scoped_user_b() -> None:
    store = InMemoryEntityTemporalMemoryStore()
    indexer = DefaultEntityMemoryIndexer(store, security_governance=build_default_memory_security_governance_service())
    scope_a = EntityMemoryScope(tenant_id="T", user_id="A")
    scope_b = EntityMemoryScope(tenant_id="T", user_id="B")
    source_id = "M"
    indexer.index_memory_entry(_identity(scope_a), scope_a, _memory_entry(source_id))
    indexer.index_memory_entry(_identity(scope_b), scope_b, _memory_entry(source_id))

    entity_a = entity_memory_entity_id_for_entry(scope_a, source_id)
    entity_b = entity_memory_entity_id_for_entry(scope_b, source_id)
    rel_b = entity_memory_relation_id_for_has_memory(scope_b, source_id)
    user_b = entity_memory_user_entity_id(scope_b)

    store.delete_by_source_memory(scope_b, source_id)
    assert store.get_entity(scope_b, entity_b) is None
    assert not _relation_exists(store, scope_b, user_entity_id=user_b, relation_id=rel_b)
    assert store.get_entity(scope_a, entity_a) is not None


def test_same_scope_double_index_is_idempotent() -> None:
    store = InMemoryEntityTemporalMemoryStore()
    indexer = DefaultEntityMemoryIndexer(store, security_governance=build_default_memory_security_governance_service())
    scope = EntityMemoryScope(tenant_id="T", user_id="A")
    entry = _memory_entry("M", revision=4)
    indexer.index_memory_entry(_identity(scope), scope, entry)
    indexer.index_memory_entry(_identity(scope), scope, entry)
    entity_id = entity_memory_entity_id_for_entry(scope, "M")
    rel_id = entity_memory_relation_id_for_has_memory(scope, "M")
    user_id = entity_memory_user_entity_id(scope)
    outbound = store.query_relations(
        scope,
        EntityRelationQuery(
            entity_id=user_id,
            direction=EntityRelationDirection.OUTBOUND,
            as_of=datetime(2025, 1, 1, tzinfo=timezone.utc),
            limit=50,
        ),
    )
    matching = [rel for rel in outbound.relations if rel.relation_id == rel_id]
    assert store.get_entity(scope, entity_id) is not None
    assert len(matching) == 1


def test_higher_revision_updates_same_scoped_ids() -> None:
    store = InMemoryEntityTemporalMemoryStore()
    indexer = DefaultEntityMemoryIndexer(store, security_governance=build_default_memory_security_governance_service())
    scope = EntityMemoryScope(tenant_id="T", user_id="A")
    indexer.index_memory_entry(_identity(scope), scope, _memory_entry("M", revision=4, content="v4"))
    entity_id = entity_memory_entity_id_for_entry(scope, "M")
    rel_id = entity_memory_relation_id_for_has_memory(scope, "M")
    indexer.index_memory_entry(_identity(scope), scope, _memory_entry("M", revision=5, content="v5"))
    projected = store.get_entity(scope, entity_id)
    assert projected is not None
    assert projected.source_memory_revision == 5
    assert _relation_exists(
        store,
        scope,
        user_entity_id=entity_memory_user_entity_id(scope),
        relation_id=rel_id,
    )


def test_stale_revision_on_one_user_does_not_affect_sibling_user() -> None:
    store = InMemoryEntityTemporalMemoryStore()
    indexer = DefaultEntityMemoryIndexer(store, security_governance=build_default_memory_security_governance_service())
    scope_a = EntityMemoryScope(tenant_id="T", user_id="A")
    scope_b = EntityMemoryScope(tenant_id="T", user_id="B")
    indexer.index_memory_entry(_identity(scope_a), scope_a, _memory_entry("M", revision=5, content="A v5"))
    indexer.index_memory_entry(_identity(scope_b), scope_b, _memory_entry("M", revision=3, content="B v3"))
    indexer.index_memory_entry(_identity(scope_a), scope_a, _memory_entry("M", revision=3, content="stale A"))

    entity_a = store.get_entity(scope_a, entity_memory_entity_id_for_entry(scope_a, "M"))
    entity_b = store.get_entity(scope_b, entity_memory_entity_id_for_entry(scope_b, "M"))
    assert entity_a is not None and entity_a.source_memory_revision == 5
    assert entity_b is not None and entity_b.source_memory_revision == 3


def test_workspace_qualifier_participates_in_relation_projection_identity() -> None:
    scope_ws1 = EntityMemoryScope(tenant_id="T", user_id="U", workspace_id="ws-1")
    scope_ws2 = EntityMemoryScope(tenant_id="T", user_id="U", workspace_id="ws-2")
    scope_no_ws = EntityMemoryScope(tenant_id="T", user_id="U")
    rel_1 = entity_memory_relation_id_for_has_memory(scope_ws1, "M")
    rel_2 = entity_memory_relation_id_for_has_memory(scope_ws2, "M")
    ent_1 = entity_memory_entity_id_for_entry(scope_ws1, "M")
    ent_2 = entity_memory_entity_id_for_entry(scope_ws2, "M")
    assert rel_1 != rel_2
    assert ent_1 != ent_2
    assert entity_memory_entity_id_for_entry(scope_no_ws, "M") == entity_memory_entity_id_for_entry(
        scope_no_ws,
        "M",
    )


def test_projection_key_delimiter_collision_resistance() -> None:
    scope_ab_c = EntityMemoryScope(tenant_id="T", user_id="a:b", workspace_id="c")
    scope_a_bc = EntityMemoryScope(tenant_id="T", user_id="a", workspace_id="b:c")
    assert entity_memory_source_projection_key(scope_ab_c, "M") != entity_memory_source_projection_key(
        scope_a_bc,
        "M",
    )
    assert entity_memory_entity_id_for_entry(scope_ab_c, "M") != entity_memory_entity_id_for_entry(
        scope_a_bc,
        "M",
    )


def test_projection_key_absent_workspace_not_colliding_with_dash_workspace() -> None:
    scope_none = EntityMemoryScope(tenant_id="T", user_id="U")
    scope_dash = EntityMemoryScope(tenant_id="T", user_id="U", workspace_id="-")
    assert entity_memory_source_projection_key(scope_none, "M") != entity_memory_source_projection_key(
        scope_dash,
        "M",
    )


def test_same_tenant_same_source_different_workspaces_both_projections_exist() -> None:
    store = InMemoryEntityTemporalMemoryStore()
    indexer = DefaultEntityMemoryIndexer(store, security_governance=build_default_memory_security_governance_service())
    scope_ws1 = EntityMemoryScope(tenant_id="T", user_id="U", workspace_id="W1")
    scope_ws2 = EntityMemoryScope(tenant_id="T", user_id="U", workspace_id="W2")
    source_id = "M"
    indexer.index_memory_entry(_identity(scope_ws1), scope_ws1, _memory_entry(source_id, revision=5))
    indexer.index_memory_entry(_identity(scope_ws2), scope_ws2, _memory_entry(source_id, revision=3))

    entity_1 = entity_memory_entity_id_for_entry(scope_ws1, source_id)
    entity_2 = entity_memory_entity_id_for_entry(scope_ws2, source_id)
    rel_1 = entity_memory_relation_id_for_has_memory(scope_ws1, source_id)
    rel_2 = entity_memory_relation_id_for_has_memory(scope_ws2, source_id)
    user_id = entity_memory_user_entity_id(scope_ws1)

    assert entity_1 != entity_2
    assert rel_1 != rel_2
    assert store.get_entity(scope_ws1, entity_1) is not None
    assert store.get_entity(scope_ws2, entity_2) is not None
    assert _relation_exists(store, scope_ws1, user_entity_id=user_id, relation_id=rel_1)
    assert _relation_exists(store, scope_ws2, user_entity_id=user_id, relation_id=rel_2)


def test_delete_by_source_memory_removes_only_scoped_workspace_w1() -> None:
    store = InMemoryEntityTemporalMemoryStore()
    indexer = DefaultEntityMemoryIndexer(store, security_governance=build_default_memory_security_governance_service())
    scope_ws1 = EntityMemoryScope(tenant_id="T", user_id="U", workspace_id="W1")
    scope_ws2 = EntityMemoryScope(tenant_id="T", user_id="U", workspace_id="W2")
    source_id = "M"
    indexer.index_memory_entry(_identity(scope_ws1), scope_ws1, _memory_entry(source_id, revision=5))
    indexer.index_memory_entry(_identity(scope_ws2), scope_ws2, _memory_entry(source_id, revision=3))

    entity_1 = entity_memory_entity_id_for_entry(scope_ws1, source_id)
    entity_2 = entity_memory_entity_id_for_entry(scope_ws2, source_id)
    rel_1 = entity_memory_relation_id_for_has_memory(scope_ws1, source_id)
    rel_2 = entity_memory_relation_id_for_has_memory(scope_ws2, source_id)
    user_id = entity_memory_user_entity_id(scope_ws1)

    removed = store.delete_by_source_memory(scope_ws1, source_id)
    assert removed == 2
    assert store.get_entity(scope_ws1, entity_1) is None
    assert not _relation_exists(store, scope_ws1, user_entity_id=user_id, relation_id=rel_1)
    assert store.get_entity(scope_ws2, entity_2) is not None
    assert _relation_exists(store, scope_ws2, user_entity_id=user_id, relation_id=rel_2)
    assert store.get_entity(scope_ws1, user_id) is not None


def test_delete_by_source_memory_removes_only_scoped_workspace_w2() -> None:
    store = InMemoryEntityTemporalMemoryStore()
    indexer = DefaultEntityMemoryIndexer(store, security_governance=build_default_memory_security_governance_service())
    scope_ws1 = EntityMemoryScope(tenant_id="T", user_id="U", workspace_id="W1")
    scope_ws2 = EntityMemoryScope(tenant_id="T", user_id="U", workspace_id="W2")
    source_id = "M"
    indexer.index_memory_entry(_identity(scope_ws1), scope_ws1, _memory_entry(source_id))
    indexer.index_memory_entry(_identity(scope_ws2), scope_ws2, _memory_entry(source_id))

    entity_2 = entity_memory_entity_id_for_entry(scope_ws2, source_id)
    rel_2 = entity_memory_relation_id_for_has_memory(scope_ws2, source_id)
    user_id = entity_memory_user_entity_id(scope_ws2)

    store.delete_by_source_memory(scope_ws2, source_id)
    assert store.get_entity(scope_ws2, entity_2) is None
    assert not _relation_exists(store, scope_ws2, user_entity_id=user_id, relation_id=rel_2)
    assert store.get_entity(scope_ws1, entity_memory_entity_id_for_entry(scope_ws1, source_id)) is not None


def test_no_dangling_relations_after_workspace_scoped_delete() -> None:
    store = InMemoryEntityTemporalMemoryStore()
    indexer = DefaultEntityMemoryIndexer(store, security_governance=build_default_memory_security_governance_service())
    scope_ws1 = EntityMemoryScope(tenant_id="T", user_id="U", workspace_id="W1")
    scope_ws2 = EntityMemoryScope(tenant_id="T", user_id="U", workspace_id="W2")
    indexer.index_memory_entry(_identity(scope_ws1), scope_ws1, _memory_entry("M", revision=1))
    indexer.index_memory_entry(_identity(scope_ws2), scope_ws2, _memory_entry("M", revision=1))
    user_id = entity_memory_user_entity_id(scope_ws1)
    store.delete_by_source_memory(scope_ws1, "M")

    for scope in (scope_ws1, scope_ws2):
        result = store.query_relations(
            scope,
            EntityRelationQuery(
                entity_id=user_id,
                direction=EntityRelationDirection.OUTBOUND,
                as_of=datetime(2025, 1, 1, tzinfo=timezone.utc),
                limit=50,
            ),
        )
        for rel in result.relations:
            assert store.get_entity(scope, rel.target_entity_id) is not None


def test_stale_revision_on_one_workspace_does_not_affect_sibling_workspace() -> None:
    store = InMemoryEntityTemporalMemoryStore()
    indexer = DefaultEntityMemoryIndexer(store, security_governance=build_default_memory_security_governance_service())
    scope_ws1 = EntityMemoryScope(tenant_id="T", user_id="U", workspace_id="W1")
    scope_ws2 = EntityMemoryScope(tenant_id="T", user_id="U", workspace_id="W2")
    indexer.index_memory_entry(_identity(scope_ws1), scope_ws1, _memory_entry("M", revision=5, content="W1 v5"))
    indexer.index_memory_entry(_identity(scope_ws2), scope_ws2, _memory_entry("M", revision=3, content="W2 v3"))
    indexer.index_memory_entry(_identity(scope_ws1), scope_ws1, _memory_entry("M", revision=4, content="stale W1"))

    entity_1 = store.get_entity(scope_ws1, entity_memory_entity_id_for_entry(scope_ws1, "M"))
    entity_2 = store.get_entity(scope_ws2, entity_memory_entity_id_for_entry(scope_ws2, "M"))
    assert entity_1 is not None and entity_1.source_memory_revision == 5
    assert entity_2 is not None and entity_2.source_memory_revision == 3


def test_user_qualifier_prevents_cross_user_entity_id_collision() -> None:
    scope_a = EntityMemoryScope(tenant_id="tenant-T", user_id="user-A")
    scope_b = EntityMemoryScope(tenant_id="tenant-T", user_id="user-B")
    assert entity_memory_entity_id_for_entry(scope_a, "mem-1") != entity_memory_entity_id_for_entry(
        scope_b,
        "mem-1",
    )


def test_indexer_entry_parameter_is_not_object() -> None:
    signature = inspect.signature(DefaultEntityMemoryIndexer.index_memory_entry)
    entry_param = signature.parameters["entry"]
    assert entry_param.annotation is not object


class _FakeEntityTemporalMemoryStorePlugin:
    @classmethod
    def plugin_id(cls) -> str:
        return "test.fake_entity_temporal"

    @classmethod
    def create_entity_temporal_memory_store(cls, **kwargs: object) -> _RecordingEntityStore:
        return _RecordingEntityStore()


def test_plugin_resolution_default_in_memory() -> None:
    env = ApplicationEnvironmentProfile.product_defaults()
    store = resolve_entity_temporal_memory_store(env)
    assert store is not None
    assert isinstance(store, InMemoryEntityTemporalMemoryStore)


def test_plugin_resolution_external_provider() -> None:
    env = ApplicationEnvironmentProfile(
        memory_profile=MemoryProfile(
            enable_entity_graph_memory=True,
            entity_temporal_memory_store_plugin_id=_FakeEntityTemporalMemoryStorePlugin.plugin_id(),
        ),
    )
    discovery = discover_classified_memory_store_plugins(
        discover_entry_points=False,
        explicit_plugins=(
            InMemoryEntityTemporalMemoryStorePlugin,
            _FakeEntityTemporalMemoryStorePlugin,
        ),
    )
    catalog = MemoryStorePluginCatalog.from_discovery(discovery)
    ctx = MemoryStoreMaterializationContext(
        tenant_id=None,
        integration_profile=env.integration_profile,
    )
    store = materialize_entity_temporal_memory_store(
        _FakeEntityTemporalMemoryStorePlugin.plugin_id(),
        ctx,
        catalog=catalog,
    )
    assert isinstance(store, _RecordingEntityStore)


def test_plugin_resolution_invalid_provider_fails() -> None:
    env = ApplicationEnvironmentProfile(
        memory_profile=MemoryProfile(
            enable_entity_graph_memory=True,
            entity_temporal_memory_store_plugin_id="plugin.does.not.exist",
        ),
    )
    discovery = discover_classified_memory_store_plugins(
        discover_entry_points=False,
        explicit_plugins=(InMemoryEntityTemporalMemoryStorePlugin,),
    )
    catalog = MemoryStorePluginCatalog.from_discovery(discovery)
    ctx = MemoryStoreMaterializationContext(
        tenant_id=None,
        integration_profile=env.integration_profile,
    )
    with pytest.raises(MemoryStorePluginResolutionError):
        materialize_entity_temporal_memory_store("plugin.does.not.exist", ctx, catalog=catalog)


def test_entity_graph_memory_disabled_returns_none() -> None:
    env = ApplicationEnvironmentProfile(
        memory_profile=MemoryProfile(enable_entity_graph_memory=False),
    )
    assert resolve_entity_temporal_memory_store(env) is None
