# © Artur Czarnecki. All rights reserved.

"""Synchronous planner failure events (Phase Q+-N.5)."""

from __future__ import annotations

from typing import Optional

from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from intergrax.runtime.events.runtime_event_identity import runtime_event_identity_kwargs
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState


def record_plan_failed(
    bus: Optional[RuntimeEventBus],
    *,
    config: RuntimeConfig,
    state: RuntimeState,
    error: Exception,
    failure_kind: str,
    raw_hash: Optional[str] = None,
) -> None:
    """Emit ``PLAN_FAILED`` when planner parse or PlanSource fails (narrow, sync ``record``)."""
    resolved = bus if bus is not None else config.runtime_event_bus
    if resolved is None:
        return

    run_id = state.run_id or ""
    meta = state.request.metadata or {}
    task_id = meta.get("task_id") if isinstance(meta.get("task_id"), str) else run_id

    resolved.record(
        RuntimeEvent(
            tenant_id=state.request.tenant_id,
            event_type=RuntimeEventType.PLAN_FAILED,
            phase=ExecutionPhase.PLANNING,
            correlation_id=run_id or task_id or "planner",
            payload={
                "failure_kind": failure_kind,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "raw_hash": raw_hash,
            },
            **runtime_event_identity_kwargs(task_id=task_id or run_id or "planner", run_id=run_id or None),
        )
    )
