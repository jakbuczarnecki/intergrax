# © Artur Czarnecki. All rights reserved.

"""MEM-ENT-14: deterministic concurrency and revision contention proofs."""

from __future__ import annotations

from dataclasses import replace

import pytest

from intergrax.memory.contracts.entity_temporal_memory import (
    EntityMemoryScope,
    EntityRecord,
    EntityTypeRef,
    entity_memory_entity_id_for_entry,
)

from intergrax.memory.contracts.long_horizon_memory import (
    LongHorizonMemoryScope,
    LongHorizonMemoryViolation,
)
from intergrax.memory.contracts.procedural_memory import (
    ProceduralMemoryScope,
    ProcedureMemoryViolation,
    ProcedureStatus,
    procedure_id_for_source_memory,
)
from intergrax.memory.stores.in_memory_entity_temporal_memory_store import (
    InMemoryEntityTemporalMemoryStore,
)
from intergrax.memory.stores.in_memory_long_horizon_memory_store import (
    InMemoryLongHorizonMemoryStore,
)
from intergrax.memory.stores.in_memory_procedural_memory_store import (
    InMemoryProceduralMemoryStore,
)
from tests.unit.memory.resilience.interleaving_stores import (
    InterleavingEntityTemporalMemoryStore,
    InterleavingProceduralMemoryStore,
)
from tests.unit.memory.test_mem_ent8_procedural_memory import _procedure
from tests.unit.memory.test_mem_ent9_long_horizon_memory import _leaf

pytestmark = pytest.mark.unit

_TENANT_A = "tenant-ent14-a"
_TENANT_B = "tenant-ent14-b"
_USER = "user-ent14"


def _entity_scope(tenant: str = _TENANT_A) -> EntityMemoryScope:
    return EntityMemoryScope(tenant_id=tenant, user_id=_USER)


def _entity_projection(
    scope: EntityMemoryScope,
    *,
    memory_id: str,
    revision: int,
    name: str,
) -> EntityRecord:
    entity_id = entity_memory_entity_id_for_entry(scope, memory_id)
    return EntityRecord(
        entity_id=entity_id,
        entity_type=EntityTypeRef("fact"),
        canonical_name=name,
        revision=revision,
        created_at="2025-01-01T00:00:00+00:00",
        source_memory_id=memory_id,
        source_memory_revision=revision,
    )


def test_entity_higher_revision_wins_over_stale() -> None:
    store = InMemoryEntityTemporalMemoryStore()
    scope = _entity_scope()
    memory_id = "mem-rev-race"
    store.upsert_entity(scope, _entity_projection(scope, memory_id=memory_id, revision=5, name="v5"))
    store.upsert_entity(scope, _entity_projection(scope, memory_id=memory_id, revision=6, name="v6"))
    store.upsert_entity(scope, _entity_projection(scope, memory_id=memory_id, revision=4, name="v4"))
    entity_id = entity_memory_entity_id_for_entry(scope, memory_id)
    final = store.get_entity(scope, entity_id)
    assert final is not None
    assert final.source_memory_revision == 6
    assert final.canonical_name == "v6"


def test_entity_gated_interleaving_stale_writer_during_higher_commit() -> None:
    store = InterleavingEntityTemporalMemoryStore()
    scope = _entity_scope()
    memory_id = "mem-gated"
    store.upsert_entity(scope, _entity_projection(scope, memory_id=memory_id, revision=5, name="v5"))

    def _stale_during_higher_commit() -> None:
        store.upsert_entity(
            scope,
            _entity_projection(scope, memory_id=memory_id, revision=4, name="stale"),
        )

    store.pre_commit = _stale_during_higher_commit
    store.upsert_entity(scope, _entity_projection(scope, memory_id=memory_id, revision=6, name="v6"))
    entity_id = entity_memory_entity_id_for_entry(scope, memory_id)
    final = store.get_entity(scope, entity_id)
    assert final is not None
    assert final.source_memory_revision == 6


def test_entity_same_revision_idempotent_under_repeated_upsert() -> None:
    store = InMemoryEntityTemporalMemoryStore()
    scope = _entity_scope()
    record = _entity_projection(scope, memory_id="mem-idem", revision=3, name="same")
    first = store.upsert_entity(scope, record)
    second = store.upsert_entity(scope, record)
    assert first == second


def test_procedure_higher_revision_wins_and_same_revision_conflict_raises() -> None:
    store = InMemoryProceduralMemoryStore()
    scope = ProceduralMemoryScope(tenant_id=_TENANT_A, user_id=_USER)
    pid = procedure_id_for_source_memory(scope, "mem-proc")
    store.upsert_procedure(
        scope,
        _procedure(pid, revision=5, source_memory_id="mem-proc", source_memory_revision=5),
    )
    store.upsert_procedure(
        scope,
        _procedure(pid, revision=6, source_memory_id="mem-proc", source_memory_revision=6),
    )
    stale = store.upsert_procedure(
        scope,
        _procedure(pid, revision=4, source_memory_id="mem-proc", source_memory_revision=4),
    )
    assert stale.source_memory_revision == 6
    base = _procedure(pid, revision=6, source_memory_id="mem-proc", source_memory_revision=6)
    store.upsert_procedure(scope, base)
    conflicting = replace(base, title="different-title")
    with pytest.raises(ProcedureMemoryViolation, match="conflicting procedural projection"):
        store.upsert_procedure(scope, conflicting)


def test_procedure_gated_interleaving_preserves_higher_revision() -> None:
    store = InterleavingProceduralMemoryStore()
    scope = ProceduralMemoryScope(tenant_id=_TENANT_A, user_id=_USER)
    pid = procedure_id_for_source_memory(scope, "mem-gated-proc")
    store.upsert_procedure(
        scope,
        _procedure(pid, revision=5, source_memory_id="mem-gated-proc", source_memory_revision=5),
    )

    def _stale_during_commit() -> None:
        store.upsert_procedure(
            scope,
            _procedure(pid, revision=4, source_memory_id="mem-gated-proc", source_memory_revision=4),
        )

    store.pre_commit = _stale_during_commit
    store.upsert_procedure(
        scope,
        _procedure(pid, revision=6, source_memory_id="mem-gated-proc", source_memory_revision=6),
    )
    final = store.get_procedure(scope, pid)
    assert final is not None
    assert final.source_memory_revision == 6


def test_long_horizon_stale_revision_rejected_higher_wins() -> None:
    store = InMemoryLongHorizonMemoryStore()
    scope = LongHorizonMemoryScope(tenant_id=_TENANT_A, user_id=_USER)
    summary_id = "sum-rev"
    store.upsert_summary(scope, _leaf(summary_id, revision=5))
    store.upsert_summary(scope, _leaf(summary_id, revision=6))
    with pytest.raises(LongHorizonMemoryViolation, match="older than stored"):
        store.upsert_summary(scope, _leaf(summary_id, revision=4))
    final = store.get_summary(scope, summary_id)
    assert final is not None
    assert final.revision == 6


def test_long_horizon_same_revision_conflict_is_deterministic() -> None:
    store = InMemoryLongHorizonMemoryStore()
    scope = LongHorizonMemoryScope(tenant_id=_TENANT_A, user_id=_USER)
    summary_id = "sum-conflict"
    baseline = _leaf(summary_id, revision=2)
    store.upsert_summary(scope, baseline)
    store.upsert_summary(scope, baseline)
    conflicting = replace(baseline, content="different payload")
    with pytest.raises(LongHorizonMemoryViolation, match="conflicting long-horizon"):
        store.upsert_summary(scope, conflicting)


def test_tenant_isolation_under_concurrent_entity_writes() -> None:
    store = InMemoryEntityTemporalMemoryStore()
    scope_a = _entity_scope(_TENANT_A)
    scope_b = _entity_scope(_TENANT_B)
    store.upsert_entity(scope_a, _entity_projection(scope_a, memory_id="mem-a", revision=1, name="A"))
    store.upsert_entity(scope_b, _entity_projection(scope_b, memory_id="mem-b", revision=1, name="B"))
    entity_a = entity_memory_entity_id_for_entry(scope_a, "mem-a")
    entity_b = entity_memory_entity_id_for_entry(scope_b, "mem-b")
    assert store.get_entity(scope_a, entity_a) is not None
    assert store.get_entity(scope_b, entity_b) is not None
    store.delete_by_source_memory(scope_a, "mem-a")
    assert store.get_entity(scope_a, entity_a) is None
    assert store.get_entity(scope_b, entity_b) is not None


def test_procedure_supersession_retry_does_not_self_supersede() -> None:
    store = InMemoryProceduralMemoryStore()
    scope = ProceduralMemoryScope(tenant_id=_TENANT_A, user_id=_USER)
    old_id = "proc-old"
    new_id = "proc-new"
    store.upsert_procedure(scope, _procedure(old_id))
    store.upsert_procedure(scope, _procedure(new_id))
    from intergrax.memory.contracts.procedural_memory import ProcedureSupersessionRequest

    request = ProcedureSupersessionRequest(
        superseded_procedure_id=old_id,
        superseding_record=_procedure(new_id, status=ProcedureStatus.ACTIVE),
    )
    store.apply_supersession(scope, request)
    with pytest.raises(ProcedureMemoryViolation, match="cannot supersede itself"):
        bad = ProcedureSupersessionRequest(
            superseded_procedure_id=new_id,
            superseding_record=_procedure(new_id, status=ProcedureStatus.ACTIVE),
        )
        store.apply_supersession(scope, bad)
