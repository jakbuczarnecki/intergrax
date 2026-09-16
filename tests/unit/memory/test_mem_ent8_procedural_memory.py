# © Artur Czarnecki. All rights reserved.

"""MEM-ENT-8 procedural memory contracts, store, recall, and plugins."""

from __future__ import annotations

import inspect
import json
from dataclasses import asdict
from datetime import datetime, timezone

import pytest

from intergrax.applications._shared.procedural_memory_wiring import (
    resolve_procedural_memory_capability,
    resolve_procedural_memory_store,
)
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    MemoryProfile,
)
from intergrax.memory.contracts.enterprise_memory_record import (
    MemoryProvenance,
    MemoryRecordSourceType,
    MemoryRecordTrust,
    MemoryTrustClass,
)
from intergrax.memory.contracts.procedural_memory import (
    ProcedureActionKind,
    ProcedureApplicability,
    ProceduralMemoryScope,
    ProcedureMemoryViolation,
    ProcedureOutcomeEvidence,
    ProcedureQuery,
    ProcedureRecallContext,
    ProcedureRecord,
    ProcedureStatus,
    ProcedureStep,
    ProcedureSupersessionRequest,
    ProcedureToolReference,
    ProcedureTypeRef,
    order_procedures_deterministic,
    procedure_id_for_source_memory,
)
from intergrax.memory.procedural_memory_indexing import DefaultProceduralMemoryIndexer
from intergrax.memory.procedural_memory_service import ProceduralMemoryService, build_default_procedural_memory_strategies
from intergrax.memory.resolver.discovery import (
    MemoryStorePluginCatalog,
    discover_classified_memory_store_plugins,
)
from intergrax.memory.resolver.errors import MemoryStorePluginResolutionError
from intergrax.memory.resolver.materialization import MemoryStoreMaterializationContext
from intergrax.memory.resolver.resolver import materialize_procedural_memory_store
from intergrax.memory.stores.in_memory_procedural_memory_plugin import (
    DEFAULT_IN_MEMORY_PROCEDURAL_PLUGIN_ID,
    InMemoryProceduralMemoryStorePlugin,
)
from intergrax.memory.stores.in_memory_procedural_memory_store import InMemoryProceduralMemoryStore
from intergrax.memory.user_profile_memory import MemoryKind, UserProfileMemoryEntry

pytestmark = pytest.mark.gate


def _scope(
    tenant: str,
    user: str = "user-a",
    workspace: str | None = None,
) -> ProceduralMemoryScope:
    return ProceduralMemoryScope(tenant_id=tenant, user_id=user, workspace_id=workspace)


def _procedure(
    procedure_id: str,
    *,
    revision: int = 1,
    status: ProcedureStatus = ProcedureStatus.ACTIVE,
    capabilities: tuple[str, ...] = (),
    quality: float | None = None,
    source_memory_id: str | None = None,
    source_memory_revision: int | None = None,
) -> ProcedureRecord:
    return ProcedureRecord(
        procedure_id=procedure_id,
        procedure_type=ProcedureTypeRef("runbook"),
        title=f"title-{procedure_id}",
        revision=revision,
        status=status,
        steps=(
            ProcedureStep(
                step_id="s1",
                position=0,
                action_kind=ProcedureActionKind.TOOL_ACTION,
                instruction="invoke capability",
                tool_reference=ProcedureToolReference(tool_capability_id="pay"),
            ),
        ),
        applicability=ProcedureApplicability(required_capabilities=capabilities),
        outcome_evidence=ProcedureOutcomeEvidence(quality_score=quality),
        provenance=MemoryProvenance(source_type=MemoryRecordSourceType.USER_EXPLICIT),
        trust=MemoryRecordTrust(trust_class=MemoryTrustClass.USER_EXPLICIT),
        created_at="2025-01-01T00:00:00+00:00",
        updated_at="2025-02-01T00:00:00+00:00",
        source_memory_id=source_memory_id,
        source_memory_revision=source_memory_revision,
    )


def _service(store: InMemoryProceduralMemoryStore | None = None) -> ProceduralMemoryService:
    return ProceduralMemoryService(
        _store=store or InMemoryProceduralMemoryStore(),
        _strategies=build_default_procedural_memory_strategies(),
    )


def test_procedure_record_contract_invariants() -> None:
    record = _procedure("proc-1")
    payload = json.dumps(asdict(record), sort_keys=True)
    assert "proc-1" in payload
    with pytest.raises(ProcedureMemoryViolation):
        ProcedureRecord(
            procedure_id="",
            procedure_type=ProcedureTypeRef("x"),
            title="t",
        )


def test_procedure_versioning_same_id() -> None:
    store = InMemoryProceduralMemoryStore()
    scope = _scope("tenant-1")
    store.upsert_procedure(scope, _procedure("proc-P", revision=1))
    updated = store.upsert_procedure(
        scope,
        _procedure("proc-P", revision=2, quality=0.9),
    )
    assert updated.revision == 2
    assert store.get_procedure(scope, "proc-P") is not None
    assert store.get_procedure(scope, "proc-P").revision == 2


def test_tenant_isolation() -> None:
    service = _service()
    service.remember_procedure(_scope("T1"), _procedure("proc-shared"))
    service.remember_procedure(_scope("T2"), _procedure("proc-shared"))
    r1 = service.recall_procedures(
        _scope("T1"),
        ProcedureQuery(limit=5),
        ProcedureRecallContext(),
    )
    r2 = service.recall_procedures(
        _scope("T2"),
        ProcedureQuery(limit=5),
        ProcedureRecallContext(),
    )
    assert len(r1.procedures) == 1
    assert len(r2.procedures) == 1
    assert r1.procedures[0].procedure_id == r2.procedures[0].procedure_id


def test_workspace_scope_isolation() -> None:
    service = _service()
    service.remember_procedure(_scope("T", workspace="W1"), _procedure("proc-w"))
    service.remember_procedure(_scope("T", workspace="W2"), _procedure("proc-w"))
    w1 = service.recall_procedures(
        _scope("T", workspace="W1"),
        ProcedureQuery(limit=5),
        ProcedureRecallContext(),
    )
    w2 = service.recall_procedures(
        _scope("T", workspace="W2"),
        ProcedureQuery(limit=5),
        ProcedureRecallContext(),
    )
    assert len(w1.procedures) == 1
    assert len(w2.procedures) == 1


def test_applicability_required_capability() -> None:
    service = _service()
    scope = _scope("T")
    service.remember_procedure(scope, _procedure("with-cap", capabilities=("billing",)))
    service.remember_procedure(scope, _procedure("no-cap"))
    with_cap = service.recall_procedures(
        scope,
        ProcedureQuery(limit=10),
        ProcedureRecallContext(available_capabilities=("billing",)),
    )
    without = service.recall_procedures(
        scope,
        ProcedureQuery(limit=10),
        ProcedureRecallContext(available_capabilities=()),
    )
    ids_with = {p.procedure_id for p in with_cap.procedures}
    ids_without = {p.procedure_id for p in without.procedures}
    assert "with-cap" in ids_with
    assert "with-cap" not in ids_without
    assert "no-cap" in ids_without


def test_ranking_deterministic_and_tie_break() -> None:
    ordered = order_procedures_deterministic(
        (
            _procedure("proc-b", quality=0.5),
            _procedure("proc-a", quality=0.5),
        )
    )
    assert [p.procedure_id for p in ordered] == ["proc-a", "proc-b"]
    service = _service()
    scope = _scope("T")
    service.remember_procedure(scope, _procedure("proc-b", quality=0.5))
    service.remember_procedure(scope, _procedure("proc-a", quality=0.5))
    ctx = ProcedureRecallContext(available_capabilities=())
    first = service.recall_procedures(scope, ProcedureQuery(limit=10), ctx)
    second = service.recall_procedures(scope, ProcedureQuery(limit=10), ctx)
    assert [p.procedure_id for p in first.procedures] == [p.procedure_id for p in second.procedures]


def test_supersession_recall_excludes_superseded_by_default() -> None:
    service = _service()
    scope = _scope("T")
    service.remember_procedure(scope, _procedure("proc-A"))
    service.supersede_procedure(
        scope,
        ProcedureSupersessionRequest(
            superseded_procedure_id="proc-A",
            superseding_record=_procedure("proc-B"),
        ),
    )
    active = service.recall_procedures(
        scope,
        ProcedureQuery(limit=10),
        ProcedureRecallContext(),
    )
    assert {p.procedure_id for p in active.procedures} == {"proc-B"}
    history = service.recall_procedures(
        scope,
        ProcedureQuery(limit=10, include_history=True, statuses=(ProcedureStatus.SUPERSEDED,)),
        ProcedureRecallContext(),
    )
    assert any(p.procedure_id == "proc-A" for p in history.procedures)
    superseded = service._store.get_procedure(scope, "proc-A")
    assert superseded is not None
    assert superseded.status is ProcedureStatus.SUPERSEDED
    assert superseded.superseded_by_procedure_id == "proc-B"


def test_stale_source_revision_ignored() -> None:
    store = InMemoryProceduralMemoryStore()
    scope = _scope("T")
    store.upsert_procedure(
        scope,
        _procedure(
            procedure_id_for_source_memory(scope, "mem-1"),
            revision=4,
            source_memory_id="mem-1",
            source_memory_revision=4,
        ),
    )
    stale = store.upsert_procedure(
        scope,
        _procedure(
            procedure_id_for_source_memory(scope, "mem-1"),
            revision=3,
            source_memory_id="mem-1",
            source_memory_revision=3,
        ),
    )
    assert stale.source_memory_revision == 4


def test_idempotent_source_projection() -> None:
    store = InMemoryProceduralMemoryStore()
    scope = _scope("T")
    record = _procedure(
        procedure_id_for_source_memory(scope, "mem-1"),
        source_memory_id="mem-1",
        source_memory_revision=2,
    )
    first = store.upsert_procedure(scope, record)
    second = store.upsert_procedure(scope, record)
    assert first == second


def test_delete_by_source_memory() -> None:
    store = InMemoryProceduralMemoryStore()
    indexer = DefaultProceduralMemoryIndexer(store)
    scope = _scope("T")
    entry = UserProfileMemoryEntry(
        entry_id="mem-del",
        kind=MemoryKind.PROCEDURAL,
        content="steps",
        revision=1,
        created_at="2025-01-01T00:00:00+00:00",
    )
    indexer.index_memory_entry(scope, entry)
    pid = procedure_id_for_source_memory(scope, "mem-del")
    assert store.get_procedure(scope, pid) is not None
    removed = indexer.remove_memory_entry(scope, "mem-del")
    assert removed == 1
    assert store.get_procedure(scope, pid) is None


def test_provenance_trust_roundtrip() -> None:
    store = InMemoryProceduralMemoryStore()
    scope = _scope("T")
    record = _procedure("proc-trust")
    stored = store.upsert_procedure(scope, record)
    loaded = store.get_procedure(scope, "proc-trust")
    assert loaded is not None
    assert loaded.provenance == record.provenance
    assert loaded.trust == record.trust


def test_recall_bounded_limit() -> None:
    service = _service()
    scope = _scope("T")
    for idx in range(5):
        service.remember_procedure(scope, _procedure(f"proc-{idx}"))
    result = service.recall_procedures(
        scope,
        ProcedureQuery(limit=2),
        ProcedureRecallContext(),
    )
    assert len(result.procedures) == 2


def test_temporal_applicability_as_of() -> None:
    service = _service()
    scope = _scope("T")
    record = ProcedureRecord(
        procedure_id="proc-temporal",
        procedure_type=ProcedureTypeRef("runbook"),
        title="temporal",
        applicability=ProcedureApplicability(
            valid_from="2025-06-01T00:00:00+00:00",
            valid_until="2025-12-01T00:00:00+00:00",
        ),
        created_at="2025-01-01T00:00:00+00:00",
    )
    service.remember_procedure(scope, record)
    inside = service.recall_procedures(
        scope,
        ProcedureQuery(
            limit=5,
            as_of=datetime(2025, 7, 1, tzinfo=timezone.utc),
        ),
        ProcedureRecallContext(),
    )
    outside = service.recall_procedures(
        scope,
        ProcedureQuery(
            limit=5,
            as_of=datetime(2026, 1, 1, tzinfo=timezone.utc),
        ),
        ProcedureRecallContext(),
    )
    assert len(inside.procedures) == 1
    assert outside.procedures == ()


def test_procedure_record_has_no_runtime_object_fields() -> None:
    from dataclasses import fields

    record = _procedure("proc-1")
    names = {field.name for field in fields(record)}
    for field_name in ("task", "executor", "tool", "context", "coroutine"):
        assert field_name not in names


def test_indexer_entry_parameter_is_not_object() -> None:
    signature = inspect.signature(DefaultProceduralMemoryIndexer.index_memory_entry)
    entry_param = signature.parameters["entry"]
    assert entry_param.annotation is not object


class _RecordingProceduralStore(InMemoryProceduralMemoryStore):
    pass


class _FakeProceduralMemoryStorePlugin:
    @classmethod
    def plugin_id(cls) -> str:
        return "test.fake_procedural"

    @classmethod
    def create_procedural_memory_store(cls, **kwargs: object) -> _RecordingProceduralStore:
        return _RecordingProceduralStore()


def test_plugin_resolution_default_in_memory() -> None:
    env = ApplicationEnvironmentProfile(
        memory_profile=MemoryProfile(enable_procedural_memory=True),
    )
    store = resolve_procedural_memory_store(env)
    assert store is not None
    assert isinstance(store, InMemoryProceduralMemoryStore)


def test_plugin_resolution_external_provider() -> None:
    env = ApplicationEnvironmentProfile(
        memory_profile=MemoryProfile(
            enable_procedural_memory=True,
            procedural_memory_store_plugin_id=_FakeProceduralMemoryStorePlugin.plugin_id(),
        ),
    )
    discovery = discover_classified_memory_store_plugins(
        discover_entry_points=False,
        explicit_plugins=(
            InMemoryProceduralMemoryStorePlugin,
            _FakeProceduralMemoryStorePlugin,
        ),
    )
    catalog = MemoryStorePluginCatalog.from_discovery(discovery)
    ctx = MemoryStoreMaterializationContext(
        env=env,
        tenant_id=None,
        integration_profile=env.integration_profile,
    )
    store = materialize_procedural_memory_store(
        _FakeProceduralMemoryStorePlugin.plugin_id(),
        ctx,
        catalog=catalog,
    )
    assert isinstance(store, _RecordingProceduralStore)


def test_plugin_resolution_invalid_provider_fails() -> None:
    env = ApplicationEnvironmentProfile(
        memory_profile=MemoryProfile(
            enable_procedural_memory=True,
            procedural_memory_store_plugin_id="plugin.does.not.exist",
        ),
    )
    discovery = discover_classified_memory_store_plugins(
        discover_entry_points=False,
        explicit_plugins=(InMemoryProceduralMemoryStorePlugin,),
    )
    catalog = MemoryStorePluginCatalog.from_discovery(discovery)
    ctx = MemoryStoreMaterializationContext(
        env=env,
        tenant_id=None,
        integration_profile=env.integration_profile,
    )
    with pytest.raises(MemoryStorePluginResolutionError):
        materialize_procedural_memory_store("plugin.does.not.exist", ctx, catalog=catalog)


def test_procedural_memory_disabled_returns_none() -> None:
    env = ApplicationEnvironmentProfile(
        memory_profile=MemoryProfile(enable_procedural_memory=False),
    )
    assert resolve_procedural_memory_store(env) is None
    assert resolve_procedural_memory_capability(env) is None


def test_procedural_contracts_no_vendor_imports() -> None:
    import intergrax.memory.contracts.procedural_memory as module

    source_path = inspect.getfile(module)
    text = open(source_path, encoding="utf-8").read()
    assert "boto" not in text
    assert "mongodb" not in text


def test_default_plugin_id_constant() -> None:
    assert DEFAULT_IN_MEMORY_PROCEDURAL_PLUGIN_ID.startswith("intergrax.")
