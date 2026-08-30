# © Artur Czarnecki. All rights reserved.

"""DIAG-FOUNDATION-3 — persistent evidence for terminal diagnostic subsystem failures."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Iterator
from unittest.mock import MagicMock

import pytest

from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    TaskId,
    bind_active_execution_identity,
    mint_attempt_id,
    mint_execution_id,
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
    TerminalDiagnosticIdentityMismatchError,
    invoke_terminal_execution_diagnostics,
)
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.events.stores.memory_runtime_event_store import InMemoryRuntimeEventStore
from intergrax.runtime.execution.boundary import ExecutionIdentityBinding

pytestmark = pytest.mark.unit

_OBSERVED_AT = datetime(2026, 8, 29, 9, 0, tzinfo=UTC)


@contextmanager
def _terminal_execution_identity_scope(
    *,
    run_id: RunId | None = None,
    attempt_id: AttemptId | None = None,
    execution_id: ExecutionId | None = None,
) -> Iterator[ExecutionIdentityBinding]:
    """Bind canonical root execution identity matching NexusLoop.handle_task."""
    resolved_run_id = run_id or mint_run_id()
    resolved_attempt_id = attempt_id or mint_attempt_id()
    resolved_execution_id = execution_id or mint_execution_id()
    token = bind_active_execution_identity(
        run_id=resolved_run_id,
        attempt_id=resolved_attempt_id,
        execution_id=resolved_execution_id,
    )
    identity = ExecutionIdentityBinding(
        run_id=resolved_run_id,
        attempt_id=resolved_attempt_id,
        execution_id=resolved_execution_id,
    )
    try:
        yield identity
    finally:
        reset_active_execution_identity(token)


def test_record_diagnostic_subsystem_failure_persists_platform_signal() -> None:
    runtime_store = InMemoryRuntimeEventStore()
    event_bus = RuntimeEventBus(persistence=runtime_store)
    task_id = mint_task_id()
    with _terminal_execution_identity_scope() as identity:
        event = record_diagnostic_subsystem_failure(
            event_bus,
            tenant_id="tenant-a",
            task_id=task_id,
            run_id=identity.run_id,
            error_type="RuntimeError",
            observed_at=_OBSERVED_AT,
        )

    assert is_diagnostic_subsystem_failure_event(event)
    assert event.event_kind == PLATFORM_DIAGNOSTIC_SUBSYSTEM_FAILURE_KIND
    assert event.tenant_id == "tenant-a"
    assert event.task_id == task_id
    assert event.run_id == identity.run_id
    assert event.attempt_id == identity.attempt_id
    assert event.execution_id == identity.execution_id
    assert event.payload["error_type"] == "RuntimeError"
    assert event.payload["source"] == "terminal_execution_diagnostics"
    assert "traceback" not in event.payload
    assert event.timestamp == _OBSERVED_AT

    persisted = runtime_store.list_for_run(identity.run_id, tenant_id="tenant-a")
    assert len(persisted) == 1
    assert is_diagnostic_subsystem_failure_event(persisted[0])


def test_bridge_records_failure_evidence_without_changing_outcome() -> None:
    runtime_store = InMemoryRuntimeEventStore()
    event_bus = RuntimeEventBus(persistence=runtime_store)
    task_id = mint_task_id()
    failing = MagicMock()
    failing.trigger_for_terminal_execution.side_effect = RuntimeError("persist failed")

    with _terminal_execution_identity_scope() as identity:
        result = invoke_terminal_execution_diagnostics(
            failing,
            tenant_id="tenant-a",
            task_id=task_id,
            run_id=identity.run_id,
            observed_at=_OBSERVED_AT,
            event_bus=event_bus,
            execution_identity=identity,
        )

    assert result is None
    assert diagnostic_subsystem_failure_observed_for_run(
        runtime_store,
        tenant_id="tenant-a",
        run_id=identity.run_id,
    )
    persisted = runtime_store.list_for_run(identity.run_id, tenant_id="tenant-a")[0]
    assert persisted.execution_id == identity.execution_id


def test_bridge_without_event_bus_does_not_record_failure() -> None:
    failing = MagicMock()
    failing.trigger_for_terminal_execution.side_effect = RuntimeError("persist failed")
    task_id = mint_task_id()
    with _terminal_execution_identity_scope() as identity:
        result = invoke_terminal_execution_diagnostics(
            failing,
            tenant_id="tenant-a",
            task_id=task_id,
            run_id=identity.run_id,
            observed_at=_OBSERVED_AT,
            execution_identity=identity,
        )

    assert result is None


def test_bridge_swallows_evidence_recording_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_store = InMemoryRuntimeEventStore()
    event_bus = RuntimeEventBus(persistence=runtime_store)
    task_id = mint_task_id()
    failing = MagicMock()
    failing.trigger_for_terminal_execution.side_effect = RuntimeError("persist failed")

    def _raise_evidence(*_args: object, **_kwargs: object) -> None:
        raise OSError("evidence journal unavailable")

    monkeypatch.setattr(
        "intergrax.runtime.diagnostics.terminal_execution_diagnostic_bridge._persist_diagnostic_subsystem_failure",
        _raise_evidence,
    )

    with _terminal_execution_identity_scope() as identity:
        result = invoke_terminal_execution_diagnostics(
            failing,
            tenant_id="tenant-a",
            task_id=task_id,
            run_id=identity.run_id,
            observed_at=_OBSERVED_AT,
            event_bus=event_bus,
            execution_identity=identity,
        )

    assert result is None
    assert not diagnostic_subsystem_failure_observed_for_run(
        runtime_store,
        tenant_id="tenant-a",
        run_id=identity.run_id,
    )


def test_successful_diagnostics_do_not_emit_failure_event() -> None:
    runtime_store = InMemoryRuntimeEventStore()
    event_bus = RuntimeEventBus(persistence=runtime_store)
    trigger = MagicMock()
    trigger.trigger_for_terminal_execution.return_value = MagicMock(
        lifecycle_result=MagicMock(created=[], updated=[], unchanged=[]),
    )
    task_id = mint_task_id()
    with _terminal_execution_identity_scope() as identity:
        invoke_terminal_execution_diagnostics(
            trigger,
            tenant_id="tenant-a",
            task_id=task_id,
            run_id=identity.run_id,
            observed_at=_OBSERVED_AT,
            event_bus=event_bus,
            execution_identity=identity,
        )

    assert runtime_store.list_for_run(identity.run_id, tenant_id="tenant-a") == []


def test_repeated_failure_record_is_idempotent_per_execution() -> None:
    runtime_store = InMemoryRuntimeEventStore()
    event_bus = RuntimeEventBus(persistence=runtime_store)
    task_id = mint_task_id()
    with _terminal_execution_identity_scope() as identity:
        for _ in range(2):
            record_diagnostic_subsystem_failure(
                event_bus,
                tenant_id="tenant-a",
                task_id=task_id,
                run_id=identity.run_id,
                error_type="RuntimeError",
                observed_at=_OBSERVED_AT,
            )

    persisted = runtime_store.list_for_run(identity.run_id, tenant_id="tenant-a")
    assert len(persisted) == 1
    assert persisted[0].event_id == diagnostic_subsystem_failure_event_id(
        run_id=identity.run_id,
        attempt_id=identity.attempt_id,
    )


def test_failure_event_preserves_execution_identity() -> None:
    runtime_store = InMemoryRuntimeEventStore()
    event_bus = RuntimeEventBus(persistence=runtime_store)
    task_id = mint_task_id()
    with _terminal_execution_identity_scope() as identity:
        record_diagnostic_subsystem_failure(
            event_bus,
            tenant_id="tenant-a",
            task_id=task_id,
            run_id=identity.run_id,
            error_type="ValueError",
            observed_at=_OBSERVED_AT,
        )

    event = runtime_store.list_for_task(task_id, tenant_id="tenant-a")[0]
    assert event.task_id == task_id
    assert event.run_id == identity.run_id
    assert event.attempt_id == identity.attempt_id
    assert event.execution_id == identity.execution_id
    assert event.event_type is RuntimeEventType.DOMAIN_SIGNAL


def test_different_attempts_do_not_collide_on_failure_event_id() -> None:
    run_id = mint_run_id()
    attempt_a = mint_attempt_id()
    attempt_b = mint_attempt_id()
    event_id_a = diagnostic_subsystem_failure_event_id(run_id=run_id, attempt_id=attempt_a)
    event_id_b = diagnostic_subsystem_failure_event_id(run_id=run_id, attempt_id=attempt_b)
    assert event_id_a != event_id_b


def test_record_without_active_execution_id_fails_closed() -> None:
    runtime_store = InMemoryRuntimeEventStore()
    event_bus = RuntimeEventBus(persistence=runtime_store)
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    token = bind_active_execution_identity(run_id=run_id, attempt_id=attempt_id)
    try:
        with pytest.raises(RuntimeError, match="active ExecutionId required"):
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


def test_bridge_without_execution_identity_outside_context_swallows_evidence_failure() -> None:
    runtime_store = InMemoryRuntimeEventStore()
    event_bus = RuntimeEventBus(persistence=runtime_store)
    task_id = mint_task_id()
    run_id = mint_run_id()
    failing = MagicMock()
    failing.trigger_for_terminal_execution.side_effect = RuntimeError("persist failed")

    result = invoke_terminal_execution_diagnostics(
        failing,
        tenant_id="tenant-a",
        task_id=task_id,
        run_id=run_id,
        observed_at=_OBSERVED_AT,
        event_bus=event_bus,
    )

    assert result is None
    assert not diagnostic_subsystem_failure_observed_for_run(
        runtime_store,
        tenant_id="tenant-a",
        run_id=run_id,
    )


def test_bridge_rejects_run_id_mismatch_with_execution_identity() -> None:
    runtime_store = InMemoryRuntimeEventStore()
    event_bus = RuntimeEventBus(persistence=runtime_store)
    task_id = mint_task_id()
    failing = MagicMock()
    failing.trigger_for_terminal_execution.side_effect = RuntimeError("persist failed")

    with _terminal_execution_identity_scope() as identity:
        with pytest.raises(TerminalDiagnosticIdentityMismatchError, match="run_id conflicts with terminal execution identity"):
            invoke_terminal_execution_diagnostics(
                failing,
                tenant_id="tenant-a",
                task_id=task_id,
                run_id=mint_run_id(),
                observed_at=_OBSERVED_AT,
                event_bus=event_bus,
                execution_identity=identity,
            )


def test_bridge_unrelated_runtime_error_from_persistence_is_not_identity_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_store = InMemoryRuntimeEventStore()
    event_bus = RuntimeEventBus(persistence=runtime_store)
    task_id = mint_task_id()
    failing = MagicMock()
    failing.trigger_for_terminal_execution.side_effect = RuntimeError("persist failed")

    def _raise_unrelated_runtime_error(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("conflicts with something unrelated")

    monkeypatch.setattr(
        "intergrax.runtime.diagnostics.terminal_execution_diagnostic_bridge._persist_diagnostic_subsystem_failure",
        _raise_unrelated_runtime_error,
    )

    with _terminal_execution_identity_scope() as identity:
        result = invoke_terminal_execution_diagnostics(
            failing,
            tenant_id="tenant-a",
            task_id=task_id,
            run_id=identity.run_id,
            observed_at=_OBSERVED_AT,
            event_bus=event_bus,
            execution_identity=identity,
        )

    assert result is None
    assert not diagnostic_subsystem_failure_observed_for_run(
        runtime_store,
        tenant_id="tenant-a",
        run_id=identity.run_id,
    )


def test_record_rejects_run_id_mismatch_with_active_context() -> None:
    runtime_store = InMemoryRuntimeEventStore()
    event_bus = RuntimeEventBus(persistence=runtime_store)
    task_id = mint_task_id()
    with _terminal_execution_identity_scope() as identity:
        with pytest.raises(RuntimeError, match="run_id conflicts with active execution identity"):
            record_diagnostic_subsystem_failure(
                event_bus,
                tenant_id="tenant-a",
                task_id=task_id,
                run_id=mint_run_id(),
                error_type="RuntimeError",
                observed_at=_OBSERVED_AT,
            )
