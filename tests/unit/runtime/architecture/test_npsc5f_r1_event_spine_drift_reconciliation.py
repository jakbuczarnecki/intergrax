# © Artur Czarnecki. All rights reserved.

"""NPSC-5F/R1 — event spine (event_bus / runtime_event) drift reconciliation."""

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.contracts.event_delivery import EventPriority
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.persistence_contract import (
    MandatoryEvidencePersistenceError,
    RuntimeEventPersistence,
    TaskRuntimeEventRuns,
)
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from intergrax.runtime.events.stores.memory_runtime_event_store import InMemoryRuntimeEventStore
from intergrax.runtime.observability.event_delivery import InMemoryEventSink
from intergrax.runtime.observability.persistence_conformance import sample_runtime_event
from testing_support.npsc5f_r1_event_spine_drift import (
    R1_EVENT_SPINE_PRE_QUALIFICATION_BASELINE_SHA,
    classify_event_spine_r1_protected_change,
    collect_event_spine_r1_protected_paths,
)
from testing_support.npsc5f_r1_protected_drift import (
    R1_POST_R2_QUALIFIED_BASELINE_SHA,
    collect_r1_protected_production_drift,
    git_changed_paths,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_TENANT = "tenant-r1-spine"


def test_npsc5f_r1_event_spine_runtime_event_additive_enum_only() -> None:
    fields = set(RuntimeEvent.model_fields)
    assert {
        "tenant_id",
        "event_id",
        "event_type",
        "phase",
        "severity",
        "timestamp",
        "payload",
    }.issubset(fields)
    assert RuntimeEventType.EXTERNAL_OPERATION_FAILED.value == "external_operation_failed"
    assert RuntimeEventType("external_operation_failed") is RuntimeEventType.EXTERNAL_OPERATION_FAILED
    baseline = sample_runtime_event(tenant_id=_TENANT)
    assert baseline.model_copy(deep=True) == baseline


def test_npsc5f_r1_event_spine_drift_classification_recorded() -> None:
    changed = collect_event_spine_r1_protected_paths(_REPO_ROOT)
    buckets = {path: classify_event_spine_r1_protected_change(path) for path in changed}
    assert buckets.get("intergrax/runtime/events/runtime_event.py") == "A"
    assert buckets.get("intergrax/runtime/events/event_bus.py") == "B"


def test_npsc5f_r1_event_spine_persist_before_optional_sink_on_record() -> None:
    order: list[str] = []

    class _TrackingPersistence(RuntimeEventPersistence):
        def append(self, event, *, tenant_id: str):
            order.append("persist")

        def list_positioned_for_run(self, run_id, *, tenant_id: str, limit: int = 1000, through=None, after=None):
            return []

        def list_for_task(self, task_id, *, tenant_id: str, limit: int = 1000):
            return []

        def list_positioned_for_task_grouped_by_run(self, task_id, *, tenant_id: str, limit: int = 1000):
            return TaskRuntimeEventRuns(runs=())

        def get_by_event_id(self, *, tenant_id: str, event_id):
            return None

    sink = InMemoryEventSink()
    original_publish = sink.publish

    def _tracking_publish(deliverable, *, priority=EventPriority.BEST_EFFORT, source_event=None):
        order.append("sink")
        return original_publish(deliverable, priority=priority)

    sink.publish = _tracking_publish  # type: ignore[method-assign]

    event = sample_runtime_event(tenant_id=_TENANT).model_copy(
        update={"event_type": RuntimeEventType.TASK_COMPLETED, "phase": ExecutionPhase.COMPLETION},
    )
    bus = RuntimeEventBus(
        persistence=_TrackingPersistence(),
        record_history=True,
        event_sink=sink,
    )
    bus.record(event, tenant_id=_TENANT)
    assert order == ["persist", "sink"]
    assert bus.history[-1].event_id == event.event_id


def test_npsc5f_r1_event_spine_mandatory_persistence_fail_closed_without_sink() -> None:
    class _Failing(RuntimeEventPersistence):
        def append(self, event, *, tenant_id: str):
            raise RuntimeError("sink down")

        def list_positioned_for_run(self, run_id, *, tenant_id: str, limit: int = 1000, through=None, after=None):
            return []

        def list_for_task(self, task_id, *, tenant_id: str, limit: int = 1000):
            return []

        def list_positioned_for_task_grouped_by_run(self, task_id, *, tenant_id: str, limit: int = 1000):
            return TaskRuntimeEventRuns(runs=())

        def get_by_event_id(self, *, tenant_id: str, event_id):
            return None

    bus = RuntimeEventBus(persistence=_Failing(), record_history=True, event_sink=InMemoryEventSink())
    event = sample_runtime_event(tenant_id=_TENANT)
    with pytest.raises(MandatoryEvidencePersistenceError):
        bus.record(event, tenant_id=_TENANT)
    assert bus.history == []


def test_npsc5f_r1_event_spine_tenant_isolation_unchanged(tmp_path: Path) -> None:
    store = InMemoryRuntimeEventStore()
    bus = RuntimeEventBus(persistence=store, record_history=False)
    event_a = sample_runtime_event(tenant_id="tenant-a")
    event_b = sample_runtime_event(tenant_id="tenant-b")
    bus.record(event_a, tenant_id="tenant-a")
    bus.record(event_b, tenant_id="tenant-b")
    assert store.list_for_task(str(event_a.task_id), tenant_id="tenant-a")
    assert store.list_for_task(str(event_b.task_id), tenant_id="tenant-b")
    assert not store.list_for_task(str(event_a.task_id), tenant_id="tenant-b")


def test_npsc5f_r1_event_spine_drift_window_captured_before_baseline_advancement() -> None:
    historical = collect_event_spine_r1_protected_paths(
        _REPO_ROOT,
        from_sha=R1_EVENT_SPINE_PRE_QUALIFICATION_BASELINE_SHA,
        to_ref=R1_POST_R2_QUALIFIED_BASELINE_SHA,
    )
    assert "intergrax/runtime/events/event_bus.py" in historical
    assert "intergrax/runtime/events/runtime_event.py" in historical


def test_npsc5f_r1_event_spine_sentinel_clean_since_qualified_baseline() -> None:
    assert collect_r1_protected_production_drift(
        _REPO_ROOT,
        from_sha=R1_POST_R2_QUALIFIED_BASELINE_SHA,
    ) == []


def test_npsc5f_r1_event_spine_pre_qualification_baseline_immutable() -> None:
    assert R1_EVENT_SPINE_PRE_QUALIFICATION_BASELINE_SHA == "40cc8c11e0b57ed4cf0d99ed1b9b297820c6eaa8"
    drift = collect_r1_protected_production_drift(
        _REPO_ROOT,
        from_sha=R1_EVENT_SPINE_PRE_QUALIFICATION_BASELINE_SHA,
        to_ref=R1_EVENT_SPINE_PRE_QUALIFICATION_BASELINE_SHA,
    )
    assert drift == []
    window = git_changed_paths(
        _REPO_ROOT,
        from_sha=R1_EVENT_SPINE_PRE_QUALIFICATION_BASELINE_SHA,
        to_ref="HEAD",
    )
    assert "intergrax/runtime/events/event_bus.py" in window
