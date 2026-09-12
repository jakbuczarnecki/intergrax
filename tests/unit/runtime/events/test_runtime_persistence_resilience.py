# © Artur Czarnecki. All rights reserved.

"""Runtime persistence resilience controls at the EvidencePersistencePort boundary."""

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.contracts.execution_evidence.persistence_boundary_errors import (
    EvidencePersistenceBoundaryError,
    EvidencePersistenceIntegrityError,
    MandatoryEvidencePersistenceError,
)
from intergrax.contracts.execution_evidence.persistence_failure_contract import (
    EvidencePersistenceFailureCategory,
)
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.evidence_durability import EvidencePersistenceRequirement
from intergrax.runtime.events.evidence_persistence_adapter import (
    RuntimeEventPersistenceEvidenceAdapter,
)
from intergrax.runtime.events.persistence_contract import RuntimeEventPersistence
from intergrax.runtime.events.runtime_event import RuntimeEvent
from intergrax.runtime.events.runtime_persistence_resilience import (
    classify_controlled_persistence_failure,
    resolve_runtime_persistence_failure,
)
from intergrax.runtime.events.stores.memory_runtime_event_store import InMemoryRuntimeEventStore
from intergrax.runtime.observability.persistence_conformance import sample_runtime_event

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_RECONSTRUCTION_PATH = _REPO_ROOT / "intergrax/runtime/diagnostics/execution_reconstruction.py"
_UNIFIED_RUN_JOURNAL_PATH = _REPO_ROOT / "intergrax/runtime/events/unified_run_journal.py"


def test_runtime_classifies_port_boundary_failures() -> None:
    integrity = classify_controlled_persistence_failure(
        EvidencePersistenceIntegrityError("idempotency conflict"),
        requirement=EvidencePersistenceRequirement.MANDATORY,
    )
    assert integrity.category is EvidencePersistenceFailureCategory.INTEGRITY
    assert integrity.runtime_may_continue is False

    infra = classify_controlled_persistence_failure(
        EvidencePersistenceBoundaryError("storage unavailable"),
        requirement=EvidencePersistenceRequirement.BEST_EFFORT,
    )
    assert infra.category is EvidencePersistenceFailureCategory.INFRASTRUCTURE
    assert infra.runtime_may_continue is True


def test_resolve_mandatory_failure_raises_contract_error() -> None:
    event = sample_runtime_event(tenant_id="tenant-resilience")
    with pytest.raises(MandatoryEvidencePersistenceError, match="mandatory"):
        resolve_runtime_persistence_failure(
            requirement=EvidencePersistenceRequirement.MANDATORY,
            failure=EvidencePersistenceBoundaryError("backend down"),
            event_type=event.event_type,
        )


def test_resolve_best_effort_failure_allows_runtime_continue() -> None:
    event = sample_runtime_event(tenant_id="tenant-resilience-be")
    resolve_runtime_persistence_failure(
        requirement=EvidencePersistenceRequirement.BEST_EFFORT,
        failure=EvidencePersistenceBoundaryError("backend down"),
        event_type=event.event_type,
    )


class _ProviderStorageFault(OSError):
    pass


def test_provider_exception_does_not_escape_adapter() -> None:
    class _BrokenStore(InMemoryRuntimeEventStore):
        def append(self, event: RuntimeEvent, *, tenant_id: str):
            raise _ProviderStorageFault("disk fault")

    adapter = RuntimeEventPersistenceEvidenceAdapter(_BrokenStore())
    event = sample_runtime_event(tenant_id="tenant-adapter-leak")
    with pytest.raises(EvidencePersistenceBoundaryError) as exc_info:
        adapter.append(event, tenant_id="tenant-adapter-leak")
    assert not isinstance(exc_info.value, _ProviderStorageFault)


def test_normal_persistence_flow_unchanged() -> None:
    store = InMemoryRuntimeEventStore()
    bus = RuntimeEventBus(persistence=store, record_history=True)
    tenant_id = "tenant-normal-flow"
    event = sample_runtime_event(tenant_id=tenant_id)
    bus.record(event, tenant_id=tenant_id)
    assert len(bus.history) == 1
    rows = store.list_for_task(event.task_id, tenant_id=tenant_id)
    assert len(rows) == 1
    assert rows[0].event_id == event.event_id


def test_mandatory_persistence_failure_skips_handler_dispatch() -> None:
    class _Failing(RuntimeEventPersistence):
        def append(self, event, *, tenant_id: str):
            raise RuntimeError("sink down")

        def list_positioned_for_run(
            self,
            run_id,
            *,
            tenant_id: str,
            limit: int = 1000,
            through=None,
            after=None,
        ):
            return []

        def list_for_task(self, task_id, *, tenant_id: str, limit: int = 1000):
            return []

        def list_positioned_for_task_grouped_by_run(
            self,
            task_id,
            *,
            tenant_id: str,
            limit: int = 1000,
        ):
            from intergrax.runtime.events.persistence_contract import TaskRuntimeEventRuns

            return TaskRuntimeEventRuns(runs=())

        def get_by_event_id(self, *, tenant_id: str, event_id):
            return None

    seen: list[str] = []

    bus = RuntimeEventBus(persistence=_Failing(), record_history=True)
    bus.subscribe(lambda _event: seen.append("handled"))
    event = sample_runtime_event(tenant_id="tenant-lifecycle")
    with pytest.raises(MandatoryEvidencePersistenceError):
        bus.record(event, tenant_id="tenant-lifecycle")
    assert seen == []
    assert bus.history == []


def test_evidence_plane_modules_untouched_by_resilience_controls() -> None:
    for path in (_RECONSTRUCTION_PATH, _UNIFIED_RUN_JOURNAL_PATH):
        source = path.read_text(encoding="utf-8")
        assert "runtime_persistence_resilience" not in source
        assert "ControlledEvidencePersistenceFailure" not in source
