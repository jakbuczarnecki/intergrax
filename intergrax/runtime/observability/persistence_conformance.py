# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Conformance harness for observability persistence backends (OBS-BUS-5)."""

from __future__ import annotations

from datetime import datetime, timezone

from intergrax.contracts.event_severity import EventSeverity
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.runtime.events.persistence_contract import RuntimeEventPersistence
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType


def sample_runtime_event(
    *,
    suffix: str,
    tenant_id: str = "tenant-conformance",
    task_id: str = "task-conformance",
    run_id: str = "run-conformance",
) -> RuntimeEvent:
    return RuntimeEvent(
        event_id=f"evt_conformance_{suffix}",
        tenant_id=tenant_id,
        task_id=task_id,
        run_id=run_id,
        event_type=RuntimeEventType.STEP_STARTED,
        phase=ExecutionPhase.STEP_EXECUTION,
        severity=EventSeverity.INFO,
        timestamp=datetime(2026, 6, 8, 12, 0, 0, tzinfo=timezone.utc),
        correlation_id=task_id,
    )


def assert_runtime_event_persistence_conformance(
    store: RuntimeEventPersistence,
    *,
    label: str,
) -> None:
    """
    Shared behavioral contract for every ``RuntimeEventPersistence`` backend.

    Covers tenant scoping, run/task listing, and idempotent append on ``event_id``.
    """
    tenant_a = f"{label}-tenant-a"
    tenant_b = f"{label}-tenant-b"
    run_id = f"{label}-run"
    task_id = f"{label}-task"

    first = sample_runtime_event(
        suffix=f"{label}-1",
        tenant_id=tenant_a,
        task_id=task_id,
        run_id=run_id,
    )
    second = sample_runtime_event(
        suffix=f"{label}-2",
        tenant_id=tenant_a,
        task_id=task_id,
        run_id=run_id,
    )
    foreign = sample_runtime_event(
        suffix=f"{label}-foreign",
        tenant_id=tenant_b,
        task_id="other-task",
        run_id="other-run",
    )

    store.append(first, tenant_id=tenant_a)
    store.append(second, tenant_id=tenant_a)
    store.append(foreign, tenant_id=tenant_b)
    store.append(first, tenant_id=tenant_a)

    by_run = store.list_for_run(run_id, tenant_id=tenant_a)
    by_task = store.list_for_task(task_id, tenant_id=tenant_a)
    assert len(by_run) == 2, f"{label}: expected 2 run events, got {len(by_run)}"
    assert len(by_task) == 2, f"{label}: expected 2 task events, got {len(by_task)}"
    assert {evt.event_id for evt in by_run} == {first.event_id, second.event_id}
    assert all(evt.tenant_id == tenant_a for evt in by_run)
    assert store.list_for_run(run_id, tenant_id=tenant_b) == []
    assert store.list_for_task(task_id, tenant_id=tenant_b) == []
