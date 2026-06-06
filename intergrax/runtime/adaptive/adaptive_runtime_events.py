# © Artur Czarnecki. All rights reserved.

"""Adaptive harness runtime event helpers (Phase W-ADAPT-4.7)."""

from __future__ import annotations

from intergrax.contracts.event_severity import EventSeverity
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType


def build_adaptive_signal_event(
    *,
    task_id: str,
    run_id: str,
    tenant_id: str,
    signal_id: str,
) -> RuntimeEvent:
    return RuntimeEvent(
        tenant_id=tenant_id,
        task_id=task_id,
        run_id=run_id,
        event_type=RuntimeEventType.ADAPTIVE_SIGNAL_RECORDED,
        phase=ExecutionPhase.TRACE_PERSISTENCE,
        severity=EventSeverity.INFO,
        payload={"signal_id": signal_id},
    )


def build_adaptive_proposal_event(
    *,
    task_id: str,
    run_id: str,
    tenant_id: str,
    proposal_id: str,
    loop_id: str,
) -> RuntimeEvent:
    return RuntimeEvent(
        tenant_id=tenant_id,
        task_id=task_id,
        run_id=run_id,
        event_type=RuntimeEventType.ADAPTIVE_PROPOSAL_SUBMITTED,
        phase=ExecutionPhase.FINALIZATION,
        severity=EventSeverity.INFO,
        payload={"proposal_id": proposal_id, "loop_id": loop_id},
    )


def build_adaptive_apply_event(
    *,
    task_id: str,
    run_id: str,
    tenant_id: str,
    version_id: str,
    artifact_type: str,
) -> RuntimeEvent:
    return RuntimeEvent(
        tenant_id=tenant_id,
        task_id=task_id,
        run_id=run_id,
        event_type=RuntimeEventType.ADAPTIVE_PROFILE_APPLIED,
        phase=ExecutionPhase.FINALIZATION,
        severity=EventSeverity.INFO,
        payload={"version_id": version_id, "artifact_type": artifact_type},
    )


def build_adaptive_rollback_event(
    *,
    task_id: str,
    run_id: str,
    tenant_id: str,
    restored_version_id: str,
    artifact_type: str,
) -> RuntimeEvent:
    return RuntimeEvent(
        tenant_id=tenant_id,
        task_id=task_id,
        run_id=run_id,
        event_type=RuntimeEventType.ADAPTIVE_PROFILE_ROLLBACK,
        phase=ExecutionPhase.FINALIZATION,
        severity=EventSeverity.WARNING,
        payload={"restored_version_id": restored_version_id, "artifact_type": artifact_type},
    )
