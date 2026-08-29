# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Persistent evidence when terminal diagnostics fail (DIAG-FOUNDATION-3)."""

from __future__ import annotations

import hashlib
from datetime import datetime

from intergrax.contracts.event_severity import EventSeverity
from intergrax.contracts.execution_identity import (
    AttemptId,
    EventId,
    RunId,
    TaskId,
    validate_event_id,
    validate_run_id,
)
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.persistence_contract import RuntimeEventPersistence
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from intergrax.runtime.events.spine_consolidation import build_platform_signal_event

PLATFORM_DIAGNOSTIC_SUBSYSTEM_FAILURE_KIND = "platform.diagnostic.subsystem_failure"
_DIAGNOSTIC_SUBSYSTEM_FAILURE_SOURCE = "terminal_execution_diagnostics"


def diagnostic_subsystem_failure_event_id(*, run_id: RunId, attempt_id: AttemptId) -> EventId:
    """Deterministic id so one terminal execution emits at most one failure record."""
    digest = hashlib.sha256(
        f"{PLATFORM_DIAGNOSTIC_SUBSYSTEM_FAILURE_KIND}:{run_id}:{attempt_id}".encode("utf-8")
    ).hexdigest()[:32]
    return validate_event_id(f"evt_{digest}")


def is_diagnostic_subsystem_failure_event(event: RuntimeEvent) -> bool:
    return (
        event.event_type is RuntimeEventType.DOMAIN_SIGNAL
        and event.event_kind == PLATFORM_DIAGNOSTIC_SUBSYSTEM_FAILURE_KIND
    )


def diagnostic_subsystem_failure_observed_for_run(
    store: RuntimeEventPersistence,
    *,
    tenant_id: str,
    run_id: RunId | str,
) -> bool:
    """Return whether terminal diagnostics failure evidence exists for a run."""
    resolved_run_id = validate_run_id(run_id)
    return any(
        is_diagnostic_subsystem_failure_event(event)
        for event in store.list_for_run(resolved_run_id, tenant_id=tenant_id)
    )


def record_diagnostic_subsystem_failure(
    event_bus: RuntimeEventBus,
    *,
    tenant_id: str,
    task_id: TaskId,
    run_id: RunId,
    error_type: str,
    observed_at: datetime,
) -> RuntimeEvent:
    """
    Persist terminal diagnostic subsystem failure on the RuntimeEvent journal.

    Must not route through DiagnosticOrchestrator or other diagnostic write paths.
    """
    safe_error_type = (error_type or "Exception").strip() or "Exception"
    event = build_platform_signal_event(
        kind=PLATFORM_DIAGNOSTIC_SUBSYSTEM_FAILURE_KIND,
        task_id=str(task_id),
        run_id=str(run_id),
        tenant_id=tenant_id,
        severity=EventSeverity.ERROR,
        correlation_id=str(task_id),
        payload={
            "error_type": safe_error_type,
            "source": _DIAGNOSTIC_SUBSYSTEM_FAILURE_SOURCE,
        },
    )
    event = event.model_copy(
        update={
            "event_id": diagnostic_subsystem_failure_event_id(
                run_id=run_id,
                attempt_id=event.attempt_id,
            ),
            "timestamp": observed_at,
        }
    )
    event_bus.record(event, tenant_id=tenant_id)
    return event


__all__ = [
    "PLATFORM_DIAGNOSTIC_SUBSYSTEM_FAILURE_KIND",
    "diagnostic_subsystem_failure_event_id",
    "diagnostic_subsystem_failure_observed_for_run",
    "is_diagnostic_subsystem_failure_event",
    "record_diagnostic_subsystem_failure",
]
