# © Artur Czarnecki. All rights reserved.

"""DIAG-FOUNDATION-3 — persistent evidence for terminal diagnostic subsystem failures."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from intergrax.contracts.execution_identity import (
    bind_active_execution_identity,
    mint_attempt_id,
    mint_run_id,
    mint_task_id,
    reset_active_execution_identity,
)
from intergrax.runtime.diagnostics.diagnostic_subsystem_failure_evidence import (
    PLATFORM_DIAGNOSTIC_SUBSYSTEM_FAILURE_KIND,
    diagnostic_subsystem_failure_event_id,
    diagnostic_subsystem_failure_observed_for_run,
    is_diagnostic_subsystem_failure_event,
    record_diagnostic_subsystem_failure,
)
from intergrax.runtime.diagnostics.terminal_execution_diagnostic_bridge import (
    invoke_terminal_execution_diagnostics,
)
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.events.stores.memory_runtime_event_store import InMemoryRuntimeEventStore

pytestmark = pytest.mark.unit

_OBSERVED_AT = datetime(2026, 8, 29, 9, 0, tzinfo=UTC)


def test_record_diagnostic_subsystem_failure_persists_platform_signal() -> None:
    runtime_store = InMemoryRuntimeEventStore()
    event_bus = RuntimeEventBus(persistence=runtime_store)
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    token = bind_active_execution_identity(run_id=run_id, attempt_id=attempt_id)
    try:
        event = record_diagnostic_subsystem_failure(
            event_bus,
            tenant_id="tenant-a",
            task_id=task_id,
            run_id=run_id,
            error_type="RuntimeError",
            observed_at=_OBSERVED_AT,
        )
    finally:
        reset_active_execution_identity(token)

    assert is_diagnostic_subsystem_failure_event(event)
    assert event.event_kind == PLATFORM_DIAGNOSTIC_SUBSYSTEM_FAILURE_KIND
    assert event.tenant_id == "tenant-a"
    assert event.task_id == task_id
    assert event.run_id == run_id
    assert event.attempt_id == attempt_id
    assert event.payload["error_type"] == "RuntimeError"
    assert event.payload["source"] == "terminal_execution_diagnostics"
    assert "traceback" not in event.payload
    assert event.timestamp == _OBSERVED_AT

    persisted = runtime_store.list_for_run(run_id, tenant_id="tenant-a")
    assert len(persisted) == 1
    assert is_diagnostic_subsystem_failure_event(persisted[0])


def test_bridge_records_failure_evidence_without_changing_outcome() -> None:
    runtime_store = InMemoryRuntimeEventStore()
    event_bus = RuntimeEventBus(persistence=runtime_store)
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    failing = MagicMock()
    failing.trigger_for_terminal_execution.side_effect = RuntimeError("persist failed")

    token = bind_active_execution_identity(run_id=run_id, attempt_id=attempt_id)
    try:
        result = invoke_terminal_execution_diagnostics(
            failing,
            tenant_id="tenant-a",
            task_id=task_id,
            run_id=run_id,
            observed_at=_OBSERVED_AT,
            event_bus=event_bus,
        )
    finally:
        reset_active_execution_identity(token)

    assert result is None
    assert diagnostic_subsystem_failure_observed_for_run(
        runtime_store,
        tenant_id="tenant-a",
        run_id=run_id,
    )


def test_bridge_without_event_bus_does_not_record_failure() -> None:
    failing = MagicMock()
    failing.trigger_for_terminal_execution.side_effect = RuntimeError("persist failed")
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    token = bind_active_execution_identity(run_id=run_id, attempt_id=attempt_id)
    try:
        result = invoke_terminal_execution_diagnostics(
            failing,
            tenant_id="tenant-a",
            task_id=task_id,
            run_id=run_id,
            observed_at=_OBSERVED_AT,
        )
    finally:
        reset_active_execution_identity(token)

    assert result is None


def test_successful_diagnostics_do_not_emit_failure_event() -> None:
    runtime_store = InMemoryRuntimeEventStore()
    event_bus = RuntimeEventBus(persistence=runtime_store)
    trigger = MagicMock()
    trigger.trigger_for_terminal_execution.return_value = MagicMock(
        lifecycle_result=MagicMock(created=[], updated=[], unchanged=[]),
    )
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    token = bind_active_execution_identity(run_id=run_id, attempt_id=attempt_id)
    try:
        invoke_terminal_execution_diagnostics(
            trigger,
            tenant_id="tenant-a",
            task_id=task_id,
            run_id=run_id,
            observed_at=_OBSERVED_AT,
            event_bus=event_bus,
        )
    finally:
        reset_active_execution_identity(token)

    assert runtime_store.list_for_run(run_id, tenant_id="tenant-a") == []


def test_repeated_failure_record_is_idempotent_per_execution() -> None:
    runtime_store = InMemoryRuntimeEventStore()
    event_bus = RuntimeEventBus(persistence=runtime_store)
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    token = bind_active_execution_identity(run_id=run_id, attempt_id=attempt_id)
    try:
        for _ in range(2):
            record_diagnostic_subsystem_failure(
                event_bus,
                tenant_id="tenant-a",
                task_id=task_id,
                run_id=run_id,
                error_type="RuntimeError",
                observed_at=_OBSERVED_AT,
            )
    finally:
        reset_active_execution_identity(token)

    persisted = runtime_store.list_for_run(run_id, tenant_id="tenant-a")
    assert len(persisted) == 1
    assert persisted[0].event_id == diagnostic_subsystem_failure_event_id(
        run_id=run_id,
        attempt_id=attempt_id,
    )


def test_failure_event_preserves_execution_identity() -> None:
    runtime_store = InMemoryRuntimeEventStore()
    event_bus = RuntimeEventBus(persistence=runtime_store)
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    token = bind_active_execution_identity(run_id=run_id, attempt_id=attempt_id)
    try:
        record_diagnostic_subsystem_failure(
            event_bus,
            tenant_id="tenant-a",
            task_id=task_id,
            run_id=run_id,
            error_type="ValueError",
            observed_at=_OBSERVED_AT,
        )
    finally:
        reset_active_execution_identity(token)

    event = runtime_store.list_for_task(task_id, tenant_id="tenant-a")[0]
    assert event.task_id == task_id
    assert event.run_id == run_id
    assert event.attempt_id == attempt_id
    assert event.event_type is RuntimeEventType.DOMAIN_SIGNAL
