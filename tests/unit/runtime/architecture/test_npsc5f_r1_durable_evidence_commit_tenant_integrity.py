# © Artur Czarnecki. All rights reserved.

"""NPSC-5F/R1 — durable evidence commit and tenant routing integrity."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from intergrax.contracts.execution_identity import mint_event_id, mint_run_id
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.runtime.events.evidence_durability import (
    EvidencePersistenceRequirement,
    evidence_persistence_requirement,
)
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.event_catalog import should_persist_event
from intergrax.runtime.events.persistence_contract import (
    EvidenceTenantRoutingMismatchError,
    MandatoryEvidencePersistenceError,
    RuntimeEventPersistence,
    resolve_event_tenant_id,
)
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.events.stores.memory_runtime_event_store import InMemoryRuntimeEventStore
from intergrax.runtime.events.stores.sqlite_runtime_event_store import SQLiteRuntimeEventStore
from intergrax.runtime.observability.persistence_conformance import sample_runtime_event

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_EVIDENCE_ROOTS = (_REPO_ROOT / "intergrax" / "runtime" / "events",)


class _FailingPersistence(RuntimeEventPersistence):
    def append(self, event, *, tenant_id: str):
        raise RuntimeError("sink down")

    def list_positioned_for_run(self, run_id, *, tenant_id: str, limit: int = 1000, through=None):
        return []

    def list_for_task(self, task_id, *, tenant_id: str, limit: int = 1000):
        return []

    def get_by_event_id(self, *, tenant_id: str, event_id):
        return None


def test_r1_mandatory_classification_for_spine_lifecycle() -> None:
    event = sample_runtime_event(tenant_id="t1")
    assert evidence_persistence_requirement(event) is EvidencePersistenceRequirement.MANDATORY


def test_r1_best_effort_classification_for_debug_spine() -> None:
    event = sample_runtime_event(tenant_id="t1").model_copy(
        update={"event_type": RuntimeEventType.TASK_PROGRESS, "phase": ExecutionPhase.STEP_EXECUTION},
    )
    event_id = mint_event_id()
    while not should_persist_event(event.model_copy(update={"event_id": event_id})):
        event_id = mint_event_id()
    sampled = event.model_copy(update={"event_id": event_id})
    assert evidence_persistence_requirement(sampled) is EvidencePersistenceRequirement.BEST_EFFORT


def test_r1_not_persisted_when_sampling_skips() -> None:
    event = sample_runtime_event(tenant_id="t1").model_copy(
        update={"event_type": RuntimeEventType.TASK_PROGRESS, "phase": ExecutionPhase.STEP_EXECUTION},
    )
    event_id = mint_event_id()
    while should_persist_event(event.model_copy(update={"event_id": event_id})):
        event_id = mint_event_id()
    skipped = event.model_copy(update={"event_id": event_id})
    assert evidence_persistence_requirement(skipped) is EvidencePersistenceRequirement.NOT_PERSISTED


def test_r1_mandatory_persistence_failure_blocks_subscribers() -> None:
    store = InMemoryRuntimeEventStore()
    bus = RuntimeEventBus(persistence=_FailingPersistence(), record_history=True)
    notified: list[str] = []
    bus.subscribe(lambda evt: notified.append(evt.event_id))
    event = sample_runtime_event(tenant_id="tenant-sub")
    with pytest.raises(MandatoryEvidencePersistenceError):
        bus.record(event, tenant_id="tenant-sub")
    assert notified == []
    assert store.list_for_run(event.run_id, tenant_id="tenant-sub") == []


def test_r1_successful_mandatory_persist_and_subscriber() -> None:
    store = InMemoryRuntimeEventStore()
    bus = RuntimeEventBus(persistence=store, record_history=True)
    notified: list[str] = []
    bus.subscribe(lambda evt: notified.append(evt.event_id))
    event = sample_runtime_event(tenant_id="tenant-ok")
    bus.record(event, tenant_id="tenant-ok")
    assert notified == [event.event_id]
    assert len(store.list_for_run(event.run_id, tenant_id="tenant-ok")) == 1


def test_r1_tenant_resolution_matrix() -> None:
    event = sample_runtime_event(tenant_id="T1")
    assert resolve_event_tenant_id(event, "T1") == "T1"
    with pytest.raises(EvidenceTenantRoutingMismatchError):
        resolve_event_tenant_id(event, "T2")
    assert resolve_event_tenant_id(event, None) == "T1"
    no_tenant_event = sample_runtime_event(tenant_id="T1").model_copy(update={"tenant_id": None})
    assert resolve_event_tenant_id(no_tenant_event, "T1") == "T1"
    assert resolve_event_tenant_id(no_tenant_event, None) == ""


def test_r1_same_event_id_different_routing_tenant_blocked() -> None:
    store = InMemoryRuntimeEventStore()
    tenant_a = "tenant-a"
    tenant_b = "tenant-b"
    event = sample_runtime_event(tenant_id=tenant_a)
    store.append(event, tenant_id=tenant_a)
    with pytest.raises(EvidenceTenantRoutingMismatchError):
        store.append(event, tenant_id=tenant_b)


def test_r1_sqlite_cross_process_durability(tmp_path: Path) -> None:
    db_path = tmp_path / "r1_cross.db"
    tenant_id = "tenant-xproc-r1"
    run_id = mint_run_id()
    writer = SQLiteRuntimeEventStore(db_path=db_path)
    event = sample_runtime_event(tenant_id=tenant_id, run_id=run_id)
    writer.append(event, tenant_id=tenant_id)
    writer.close()
    reader = SQLiteRuntimeEventStore(db_path=db_path)
    rows = reader.list_for_run(run_id, tenant_id=tenant_id)
    reader.close()
    assert len(rows) == 1


def test_r1_no_reflection_on_evidence_contracts() -> None:
    violations: list[str] = []
    targets = (
        _REPO_ROOT / "intergrax" / "runtime" / "events" / "evidence_durability.py",
        _REPO_ROOT / "intergrax" / "runtime" / "events" / "persistence_contract.py",
        _REPO_ROOT / "intergrax" / "runtime" / "events" / "event_bus.py",
    )
    for path in targets:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in {"getattr", "setattr", "hasattr"}:
                    violations.append(f"{path.name}:{node.lineno}:{node.func.id}")
    assert violations == []


@pytest.mark.gate
def test_npsc5f_r1_qualification_gate() -> None:
    assert evidence_persistence_requirement(sample_runtime_event()) is (
        EvidencePersistenceRequirement.MANDATORY
    )
