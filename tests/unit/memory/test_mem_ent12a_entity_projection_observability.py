# © Artur Czarnecki. All rights reserved.

"""MEM-ENT-12A: entity projection diagnostics and shared root observability."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, fields

import pytest

from intergrax.applications._shared.entity_graph_wiring import resolve_entity_memory_indexer
from intergrax.applications._shared.memory_control_wiring import build_default_memory_control_plane
from intergrax.applications._shared.memory_wiring import resolve_memory_platform_wiring
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.contracts.agent_run import RequestIdentity
from intergrax.memory.contracts.entity_temporal_memory import (
    EntityMemoryScope,
    EntityRecord,
    EntityTemporalMemoryStore,
    EntityTypeRef,
    entity_memory_entity_id_for_entry,
)
from intergrax.memory.contracts.memory_control import (
    MemoryControlRememberRequest,
    user_memory_scope,
)
from intergrax.memory.contracts.memory_models import MemoryKind, UserProfileMemoryEntry
from intergrax.memory.contracts.memory_observability import (
    MemoryDiagnosticComponent,
    MemoryDiagnosticEvent,
    MemoryDiagnosticOperation,
    MemoryDiagnosticOutcome,
    MemoryDiagnosticPhase,
    MemoryObservabilitySink,
    RecordingMemoryObservabilitySink,
)
from intergrax.memory.contracts.memory_security_governance import MemoryGovernanceDenied
from intergrax.memory.entity_memory_indexing import DefaultEntityMemoryIndexer
from intergrax.memory.memory_diagnostic_emitter import MemoryDiagnosticEmitter
from intergrax.memory.memory_security_governance_service import (
    MemorySecurityGovernanceService,
    build_default_memory_security_governance_service,
)
from intergrax.memory.stores.in_memory_entity_temporal_memory_store import (
    InMemoryEntityTemporalMemoryStore,
)
from intergrax.memory.stores.in_memory_user_profile_store import InMemoryUserProfileStore
from intergrax.memory.user_profile_manager import UserProfileManager
from tests.unit.memory.test_mem_ent10b_specialized_mutation_governance import (
    _deny_governance,
)

pytestmark = pytest.mark.gate

_TENANT = "tenant-12a"
_USER = "user-12a"


def _identity() -> RequestIdentity:
    return RequestIdentity(tenant_id=_TENANT, user_id=_USER)


def _entity_scope() -> EntityMemoryScope:
    return EntityMemoryScope(tenant_id=_TENANT, user_id=_USER, workspace_id="ws-12a")


def _entry(content: str = "entity fact", revision: int = 1) -> UserProfileMemoryEntry:
    return UserProfileMemoryEntry(
        content=content,
        kind=MemoryKind.USER_FACT,
        revision=revision,
    )


def _projection_terminals(
    events: list[MemoryDiagnosticEvent],
    *,
    operation: MemoryDiagnosticOperation,
) -> list[MemoryDiagnosticEvent]:
    return [
        e
        for e in events
        if e.component is MemoryDiagnosticComponent.ENTITY_TEMPORAL
        and e.phase is MemoryDiagnosticPhase.PROJECTION
        and e.operation is operation
    ]


def _indexer(
    store: EntityTemporalMemoryStore | None = None,
    *,
    governance: MemorySecurityGovernanceService | None = None,
    sink: RecordingMemoryObservabilitySink | None = None,
) -> tuple[DefaultEntityMemoryIndexer, RecordingMemoryObservabilitySink]:
    recording = sink or RecordingMemoryObservabilitySink()
    emitter = MemoryDiagnosticEmitter(_sink=recording)
    indexer = DefaultEntityMemoryIndexer(
        store or InMemoryEntityTemporalMemoryStore(),
        security_governance=governance or build_default_memory_security_governance_service(
            diagnostic_emitter=emitter,
        ),
        diagnostic_emitter=emitter,
    )
    return indexer, recording


def test_entity_projection_success_emits_one_terminal() -> None:
    store = InMemoryEntityTemporalMemoryStore()
    indexer, recording = _indexer(store)
    entry = _entry()
    scope = _entity_scope()
    indexer.index_memory_entry(_identity(), scope, entry)
    terminals = _projection_terminals(
        recording.events, operation=MemoryDiagnosticOperation.PROJECTION_WRITE
    )
    assert len(terminals) == 1
    event = terminals[0]
    assert event.outcome is MemoryDiagnosticOutcome.SUCCESS
    assert event.memory_id == entry.entry_id
    assert event.revision == entry.revision
    assert event.tenant_id == scope.tenant_id
    assert event.user_id == scope.user_id
    assert event.workspace_id == scope.workspace_id
    assert store.get_entity(scope, entity_memory_entity_id_for_entry(scope, entry.entry_id))


def test_entity_projection_deny_emits_denied_without_mutation() -> None:
    store = InMemoryEntityTemporalMemoryStore()
    recording = RecordingMemoryObservabilitySink()
    emitter = MemoryDiagnosticEmitter(_sink=recording)
    indexer = DefaultEntityMemoryIndexer(
        store,
        security_governance=_deny_governance(),
        diagnostic_emitter=emitter,
    )
    entry = _entry()
    scope = _entity_scope()
    with pytest.raises(MemoryGovernanceDenied):
        indexer.index_memory_entry(_identity(), scope, entry)
    terminals = _projection_terminals(
        recording.events, operation=MemoryDiagnosticOperation.PROJECTION_WRITE
    )
    assert len(terminals) == 1
    assert terminals[0].outcome is MemoryDiagnosticOutcome.DENIED
    assert store.get_entity(scope, entity_memory_entity_id_for_entry(scope, entry.entry_id)) is None


def test_entity_projection_store_failure_emits_failed_and_reraises() -> None:
    class _FailingStore(InMemoryEntityTemporalMemoryStore):
        def upsert_entity(self, scope: EntityMemoryScope, record: EntityRecord) -> EntityRecord:
            raise RuntimeError("store down")

    store = _FailingStore()
    indexer, recording = _indexer(store)
    entry = _entry()
    with pytest.raises(RuntimeError, match="store down"):
        indexer.index_memory_entry(_identity(), _entity_scope(), entry)
    terminals = _projection_terminals(
        recording.events, operation=MemoryDiagnosticOperation.PROJECTION_WRITE
    )
    assert len(terminals) == 1
    assert terminals[0].outcome is MemoryDiagnosticOutcome.FAILED


def test_entity_delete_success_emits_terminal() -> None:
    store = InMemoryEntityTemporalMemoryStore()
    indexer, recording = _indexer(store)
    scope = _entity_scope()
    entry = _entry()
    indexer.index_memory_entry(_identity(), scope, entry)
    recording.clear()
    indexer.remove_memory_entry(_identity(), scope, entry.entry_id)
    terminals = _projection_terminals(
        recording.events, operation=MemoryDiagnosticOperation.PROJECTION_DELETE
    )
    assert len(terminals) == 1
    assert terminals[0].outcome is MemoryDiagnosticOutcome.SUCCESS


def test_entity_delete_deny_emits_denied() -> None:
    store = InMemoryEntityTemporalMemoryStore()
    recording = RecordingMemoryObservabilitySink()
    allow_emitter = MemoryDiagnosticEmitter(_sink=RecordingMemoryObservabilitySink())
    indexer = DefaultEntityMemoryIndexer(
        store,
        security_governance=build_default_memory_security_governance_service(
            diagnostic_emitter=allow_emitter,
        ),
        diagnostic_emitter=allow_emitter,
    )
    scope = _entity_scope()
    entry = _entry()
    indexer.index_memory_entry(_identity(), scope, entry)
    deny_emitter = MemoryDiagnosticEmitter(_sink=recording)
    deny_indexer = DefaultEntityMemoryIndexer(
        store,
        security_governance=_deny_governance(),
        diagnostic_emitter=deny_emitter,
    )
    with pytest.raises(MemoryGovernanceDenied):
        deny_indexer.remove_memory_entry(_identity(), scope, entry.entry_id)
    terminals = _projection_terminals(
        recording.events, operation=MemoryDiagnosticOperation.PROJECTION_DELETE
    )
    assert len(terminals) == 1
    assert terminals[0].outcome is MemoryDiagnosticOutcome.DENIED
    assert store.get_entity(scope, entity_memory_entity_id_for_entry(scope, entry.entry_id))


def test_entity_projection_privacy_safe_event() -> None:
    indexer, recording = _indexer()
    indexer.index_memory_entry(_identity(), _entity_scope(), _entry(content="secret payload"))
    event = _projection_terminals(
        recording.events, operation=MemoryDiagnosticOperation.PROJECTION_WRITE
    )[0]
    forbidden = {"content", "canonical_name", "aliases", "body", "text", "summary"}
    for field in fields(event):
        assert field.name not in forbidden
    payload = str(event)
    assert "secret payload" not in payload


def test_sink_failure_isolation_projection_still_succeeds() -> None:
    class _ExplodingSink:
        def record(self, event: MemoryDiagnosticEvent) -> None:
            raise RuntimeError("sink down")

    store = InMemoryEntityTemporalMemoryStore()
    emitter = MemoryDiagnosticEmitter(_sink=_ExplodingSink())
    indexer = DefaultEntityMemoryIndexer(
        store,
        security_governance=build_default_memory_security_governance_service(
            diagnostic_emitter=emitter,
        ),
        diagnostic_emitter=emitter,
    )
    entry = _entry()
    scope = _entity_scope()
    indexer.index_memory_entry(_identity(), scope, entry)
    assert store.get_entity(scope, entity_memory_entity_id_for_entry(scope, entry.entry_id))


def test_standalone_entity_indexer_resolver_without_custom_sink() -> None:
    env = ApplicationEnvironmentProfile.product_defaults()
    indexer = resolve_entity_memory_indexer(env)
    assert indexer is not None
    indexer.index_memory_entry(_identity(), _entity_scope(), _entry())


@dataclass(slots=True)
class _CustomSink:
    events: list[MemoryDiagnosticEvent]

    def record(self, event: MemoryDiagnosticEvent) -> None:
        self.events.append(event)


def test_custom_sink_replaceability() -> None:
    custom = _CustomSink(events=[])
    store = InMemoryEntityTemporalMemoryStore()
    emitter = MemoryDiagnosticEmitter(_sink=custom)
    indexer = DefaultEntityMemoryIndexer(
        store,
        security_governance=build_default_memory_security_governance_service(
            diagnostic_emitter=emitter,
        ),
        diagnostic_emitter=emitter,
    )
    indexer.index_memory_entry(_identity(), _entity_scope(), _entry())
    assert custom.events


def test_shared_root_sink_collects_control_plane_and_entity_projection() -> None:
    env = ApplicationEnvironmentProfile.product_defaults()
    recording = RecordingMemoryObservabilitySink()
    wiring = resolve_memory_platform_wiring(env, memory_observability_sink=recording)
    assert wiring.entity_memory_indexer is not None
    plane = build_default_memory_control_plane(
        user_profile_manager=UserProfileManager(
            store=wiring.user_profile_store,
            tenant_id=_TENANT,
        ),
        memory_observability_sink=recording,
    )

    async def _remember() -> None:
        await plane.remember(
            _identity(),
            user_memory_scope(_identity()),
            MemoryControlRememberRequest(content="cp fact"),
        )

    import asyncio

    asyncio.run(_remember())
    wiring.entity_memory_indexer.index_memory_entry(
        _identity(),
        _entity_scope(),
        _entry(content="projection fact"),
    )
    cp_success = [
        e
        for e in recording.events
        if e.operation is MemoryDiagnosticOperation.REMEMBER
        and e.outcome is MemoryDiagnosticOutcome.SUCCESS
    ]
    entity_success = _projection_terminals(
        recording.events, operation=MemoryDiagnosticOperation.PROJECTION_WRITE
    )
    assert cp_success
    assert entity_success
    assert len(recording.events) >= 2


def test_resolve_memory_platform_wiring_accepts_protocol_sink() -> None:
    signature = inspect.signature(resolve_memory_platform_wiring)
    param = signature.parameters["memory_observability_sink"]
    assert "MemoryObservabilitySink" in str(param.annotation)
