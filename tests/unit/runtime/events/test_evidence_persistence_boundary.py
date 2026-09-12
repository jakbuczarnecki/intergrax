# © Artur Czarnecki. All rights reserved.

"""Enterprise execution evidence persistence boundary (port + adapter)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from intergrax.contracts.execution_evidence.persistence_boundary_errors import (
    EvidencePersistenceBoundaryError,
    EvidencePersistenceIntegrityError,
    MandatoryEvidencePersistenceError,
)
from intergrax.contracts.execution_evidence.persistence_port import EvidencePersistencePort
from intergrax.contracts.execution_identity import EventId, mint_run_id, mint_task_id
from intergrax.runtime.diagnostics.execution_reconstruction import ExecutionReconstructor
from intergrax.runtime.events.evidence_persistence_adapter import (
    RuntimeEventPersistenceEvidenceAdapter,
    as_evidence_persistence_port,
)
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.execution_position import (
    AsOfBoundary,
    ExecutionEventPosition,
    PositionedRuntimeEvent,
)
from intergrax.runtime.events.persistence_contract import (
    RuntimeEventPersistenceIntegrityError,
    TaskRuntimeEventRuns,
)
from intergrax.runtime.events.runtime_event import RuntimeEvent
from intergrax.runtime.events.stores.memory_runtime_event_store import InMemoryRuntimeEventStore
from intergrax.runtime.observability.memory_causal_evidence_persistence import (
    InMemoryCausalEvidencePersistence,
)
from intergrax.runtime.observability.persistence_conformance import sample_runtime_event

pytestmark = [pytest.mark.unit, pytest.mark.gate]


class _DelegateEvidencePersistencePort:
    """
    Test-only ``EvidencePersistencePort`` (provider readiness).

    Not production storage — proves an alternate port can wire without the adapter.
    """

    __slots__ = ("_store",)

    def __init__(self, store: InMemoryRuntimeEventStore) -> None:
        self._store = store

    def append(self, event: RuntimeEvent, *, tenant_id: str) -> PositionedRuntimeEvent:
        return self._store.append(event, tenant_id=tenant_id)

    def list_positioned_for_run(
        self,
        run_id: str,
        *,
        tenant_id: str,
        limit: int = 1000,
        through: ExecutionEventPosition | None = None,
        after: ExecutionEventPosition | None = None,
    ) -> list[PositionedRuntimeEvent]:
        return self._store.list_positioned_for_run(
            run_id,
            tenant_id=tenant_id,
            limit=limit,
            through=through,
            after=after,
        )

    def list_for_task(
        self,
        task_id: str,
        *,
        tenant_id: str,
        limit: int = 1000,
    ) -> list[RuntimeEvent]:
        return self._store.list_for_task(task_id, tenant_id=tenant_id, limit=limit)

    def list_positioned_for_task_grouped_by_run(
        self,
        task_id: str,
        *,
        tenant_id: str,
        limit: int = 1000,
    ) -> TaskRuntimeEventRuns:
        return self._store.list_positioned_for_task_grouped_by_run(
            task_id,
            tenant_id=tenant_id,
            limit=limit,
        )

    def get_by_event_id(
        self,
        *,
        tenant_id: str,
        event_id: EventId,
    ) -> PositionedRuntimeEvent | None:
        return self._store.get_by_event_id(tenant_id=tenant_id, event_id=event_id)

    def list_positioned_through(
        self,
        boundary: AsOfBoundary,
        *,
        tenant_id: str,
        limit: int = 1000,
    ) -> list[PositionedRuntimeEvent]:
        return self._store.list_positioned_through(
            boundary,
            tenant_id=tenant_id,
            limit=limit,
        )

_REPO_ROOT = Path(__file__).resolve().parents[4]
_EVENT_BUS_PATH = _REPO_ROOT / "intergrax/runtime/events/event_bus.py"
_EXECUTION_ROOT = _REPO_ROOT / "intergrax/runtime/execution"
_RECONSTRUCTION_PATH = _REPO_ROOT / "intergrax/runtime/diagnostics/execution_reconstruction.py"
_UNIFIED_RUN_JOURNAL_PATH = _REPO_ROOT / "intergrax/runtime/events/unified_run_journal.py"


def test_runtime_event_bus_uses_evidence_persistence_port_only() -> None:
    source = _EVENT_BUS_PATH.read_text(encoding="utf-8")
    assert "RuntimeEventPersistence" not in source
    assert "EvidencePersistencePort" in source


def test_execution_engine_has_no_runtime_event_persistence_dependency() -> None:
    violations: list[str] = []
    for path in _EXECUTION_ROOT.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        rel = path.relative_to(_REPO_ROOT).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.module != "intergrax.runtime.events.persistence_contract":
                continue
            for alias in node.names:
                if alias.name == "RuntimeEventPersistence":
                    violations.append(f"{rel}:{node.lineno}")
    assert violations == []


def test_adapter_is_evidence_persistence_port() -> None:
    inner = InMemoryRuntimeEventStore()
    adapter = RuntimeEventPersistenceEvidenceAdapter(inner)
    assert isinstance(adapter, EvidencePersistencePort)


def test_adapter_preserves_execution_reconstruction() -> None:
    inner = InMemoryRuntimeEventStore()
    adapter = RuntimeEventPersistenceEvidenceAdapter(inner)
    tenant_id = "tenant-boundary"
    task_id = mint_task_id()
    run_id = mint_run_id()
    event = sample_runtime_event(tenant_id=tenant_id, task_id=task_id, run_id=run_id)
    inner.append(event, tenant_id=tenant_id)

    causal = InMemoryCausalEvidencePersistence()
    direct = ExecutionReconstructor(inner, causal).reconstruct_execution(
        tenant_id,
        task_id,
        run_id,
    )
    via_port = ExecutionReconstructor(adapter, causal).reconstruct_execution(
        tenant_id,
        task_id,
        run_id,
    )
    assert via_port == direct


def test_event_bus_wraps_legacy_runtime_event_persistence() -> None:
    store = InMemoryRuntimeEventStore()
    bus = RuntimeEventBus(persistence=store, record_history=False)
    port = bus.persistence
    assert port is not None
    assert isinstance(port, RuntimeEventPersistenceEvidenceAdapter)
    assert port.inner is store


def test_as_evidence_persistence_port_idempotent() -> None:
    inner = InMemoryRuntimeEventStore()
    first = as_evidence_persistence_port(inner)
    second = as_evidence_persistence_port(first)
    assert first is second


def test_alternate_port_implementation_satisfies_evidence_persistence_port() -> None:
    store = InMemoryRuntimeEventStore()
    port = _DelegateEvidencePersistencePort(store)
    assert isinstance(port, EvidencePersistencePort)


def test_as_evidence_persistence_port_passes_through_port_implementation() -> None:
    store = InMemoryRuntimeEventStore()
    port = _DelegateEvidencePersistencePort(store)
    normalized = as_evidence_persistence_port(port)
    assert normalized is port
    bus = RuntimeEventBus(persistence=port, record_history=False)
    assert bus.persistence is port


def test_event_bus_persists_through_alternate_port_implementation() -> None:
    store = InMemoryRuntimeEventStore()
    port = _DelegateEvidencePersistencePort(store)
    bus = RuntimeEventBus(persistence=port, record_history=False)
    tenant_id = "tenant-alt-port"
    event = sample_runtime_event(tenant_id=tenant_id)
    bus.record(event, tenant_id=tenant_id)
    persisted = port.list_for_task(event.task_id, tenant_id=tenant_id)
    assert len(persisted) == 1
    assert persisted[0].event_id == event.event_id


def test_reconstruction_accepts_alternate_port_implementation() -> None:
    store = InMemoryRuntimeEventStore()
    port = _DelegateEvidencePersistencePort(store)
    tenant_id = "tenant-alt-recon"
    task_id = mint_task_id()
    run_id = mint_run_id()
    event = sample_runtime_event(tenant_id=tenant_id, task_id=task_id, run_id=run_id)
    store.append(event, tenant_id=tenant_id)
    causal = InMemoryCausalEvidencePersistence()
    via_adapter = ExecutionReconstructor(
        RuntimeEventPersistenceEvidenceAdapter(store),
        causal,
    ).reconstruct_execution(tenant_id, task_id, run_id)
    via_port = ExecutionReconstructor(port, causal).reconstruct_execution(
        tenant_id,
        task_id,
        run_id,
    )
    assert via_port == via_adapter


def test_reconstruction_consumers_use_evidence_persistence_port_only() -> None:
    for path in (_RECONSTRUCTION_PATH, _UNIFIED_RUN_JOURNAL_PATH):
        source = path.read_text(encoding="utf-8")
        assert "RuntimeEventPersistence" not in source
        assert "EvidencePersistencePort" in source


class _ProviderSpecificStorageError(OSError):
    """Simulated vendor/storage failure outside Intergrax contracts."""


def test_adapter_translates_storage_integrity_to_port_boundary() -> None:
    inner = InMemoryRuntimeEventStore()
    adapter = RuntimeEventPersistenceEvidenceAdapter(inner)
    tenant_id = "tenant-failure-boundary"
    first = sample_runtime_event(tenant_id=tenant_id)
    inner.append(first, tenant_id=tenant_id)
    conflicting = first.model_copy(update={"payload": {"mutated": True}})
    with pytest.raises(EvidencePersistenceIntegrityError) as exc_info:
        adapter.append(conflicting, tenant_id=tenant_id)
    assert not isinstance(exc_info.value, RuntimeEventPersistenceIntegrityError)
    assert isinstance(exc_info.value.__cause__, RuntimeEventPersistenceIntegrityError)


def test_adapter_translates_provider_exceptions_to_port_boundary() -> None:
    class _BrokenStore(InMemoryRuntimeEventStore):
        def append(self, event, *, tenant_id: str):
            raise _ProviderSpecificStorageError("disk unavailable")

    adapter = RuntimeEventPersistenceEvidenceAdapter(_BrokenStore())
    event = sample_runtime_event(tenant_id="tenant-provider")
    with pytest.raises(EvidencePersistenceBoundaryError) as exc_info:
        adapter.append(event, tenant_id="tenant-provider")
    assert not isinstance(exc_info.value, _ProviderSpecificStorageError)
    assert isinstance(exc_info.value.__cause__, _ProviderSpecificStorageError)


def test_event_bus_mandatory_failure_exposes_only_port_boundary_errors() -> None:
    class _BrokenStore(InMemoryRuntimeEventStore):
        def append(self, event, *, tenant_id: str):
            raise _ProviderSpecificStorageError("backend timeout")

    bus = RuntimeEventBus(persistence=_BrokenStore(), record_history=True)
    event = sample_runtime_event(tenant_id="tenant-bus-boundary")
    with pytest.raises(MandatoryEvidencePersistenceError) as exc_info:
        bus.record(event, tenant_id="tenant-bus-boundary")
    assert not isinstance(exc_info.value.__cause__, _ProviderSpecificStorageError)
    assert isinstance(exc_info.value.__cause__, EvidencePersistenceBoundaryError)


def test_execution_engine_does_not_reference_storage_persistence_errors() -> None:
    violations: list[str] = []
    for path in _EXECUTION_ROOT.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        rel = path.relative_to(_REPO_ROOT).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.module != "intergrax.runtime.events.persistence_contract":
                continue
            for alias in node.names:
                if alias.name == "RuntimeEventPersistenceIntegrityError":
                    violations.append(f"{rel}:{node.lineno}")
    assert violations == []
