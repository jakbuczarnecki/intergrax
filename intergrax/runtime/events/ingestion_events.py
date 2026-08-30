# © Artur Czarnecki. All rights reserved.

"""Synchronous ingestion failure events (Phase Q+-O.4)."""

from __future__ import annotations

from typing import Optional

from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.contracts.event_severity import EventSeverity
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from intergrax.runtime.events.runtime_event_identity import runtime_event_identity_kwargs


def record_ingestion_failed(
    bus: Optional[RuntimeEventBus],
    *,
    attachment_id: str,
    session_id: str,
    user_id: str,
    error: Exception,
    tenant_id: Optional[str] = None,
    run_id: Optional[str] = None,
) -> None:
    if bus is None:
        return
    candidate_task_id = session_id
    if run_id is not None and str(run_id).startswith("task_"):
        candidate_task_id = run_id
    bus.record(
        RuntimeEvent(
            tenant_id=tenant_id,
            event_type=RuntimeEventType.INGESTION_FAILED,
            phase=ExecutionPhase.CONTEXT_BUILDING,
            severity=EventSeverity.ERROR,
            correlation_id=session_id,
            payload={
                "attachment_id": attachment_id,
                "session_id": session_id,
                "user_id": user_id,
                "error_type": type(error).__name__,
                "error_message": str(error),
            },
            **runtime_event_identity_kwargs(task_id=candidate_task_id, run_id=run_id),
        )
    )
