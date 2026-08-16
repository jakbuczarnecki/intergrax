# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Conformance harness for observability persistence backends (OBS-BUS-5)."""

from __future__ import annotations

from datetime import datetime, timezone

from intergrax.contracts.event_severity import EventSeverity
from intergrax.contracts.execution_identity import (
    AttemptId,
    EventId,
    RunId,
    TaskId,
    mint_attempt_id,
    mint_event_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.runtime.events.persistence_contract import RuntimeEventPersistence
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType


def sample_runtime_event(
    *,
    event_id: EventId | None = None,
    tenant_id: str = "tenant-conformance",
    task_id: TaskId | None = None,
    run_id: RunId | None = None,
    attempt_id: AttemptId | None = None,
) -> RuntimeEvent:
    resolved_task_id = task_id or mint_task_id()
    resolved_run_id = run_id or mint_run_id()
    return RuntimeEvent(
        event_id=event_id or mint_event_id(),
        tenant_id=tenant_id,
        task_id=resolved_task_id,
        run_id=resolved_run_id,
        attempt_id=attempt_id or mint_attempt_id(),
        event_type=RuntimeEventType.STEP_STARTED,
        phase=ExecutionPhase.STEP_EXECUTION,
        severity=EventSeverity.INFO,
        timestamp=datetime(2026, 6, 8, 12, 0, 0, tzinfo=timezone.utc),
        correlation_id=resolved_task_id,
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
    run_id = mint_run_id()
    task_id = mint_task_id()
    attempt_id = mint_attempt_id()

    first = sample_runtime_event(
        tenant_id=tenant_a,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
    )
    second = sample_runtime_event(
        tenant_id=tenant_a,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
    )
    foreign = sample_runtime_event(
        tenant_id=tenant_b,
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
    )

    store.append(first, tenant_id=tenant_a)
    store.append(second, tenant_id=tenant_a)
    store.append(foreign, tenant_id=tenant_b)
    duplicate = store.append(first, tenant_id=tenant_a)

    by_run = store.list_for_run(run_id, tenant_id=tenant_a)
    positioned = store.list_positioned_for_run(run_id, tenant_id=tenant_a)
    by_task = store.list_for_task(task_id, tenant_id=tenant_a)
    assert len(by_run) == 2, f"{label}: expected 2 run events, got {len(by_run)}"
    assert len(positioned) == 2, f"{label}: expected 2 positioned run events, got {len(positioned)}"
    assert len(by_task) == 2, f"{label}: expected 2 task events, got {len(by_task)}"
    assert {evt.event_id for evt in by_run} == {first.event_id, second.event_id}
    assert all(evt.tenant_id == tenant_a for evt in by_run)
    assert positioned[0].position.value < positioned[1].position.value
    assert duplicate.position == positioned[0].position
    assert store.list_for_run(run_id, tenant_id=tenant_b) == []
    assert store.list_for_task(task_id, tenant_id=tenant_b) == []
