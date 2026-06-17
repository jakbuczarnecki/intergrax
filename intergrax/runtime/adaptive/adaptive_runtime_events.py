# © Artur Czarnecki. All rights reserved.

"""Adaptive harness runtime event helpers (Phase W-ADAPT-4.7)."""

from __future__ import annotations

from intergrax.contracts.event_severity import EventSeverity
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.runtime.events.runtime_event import RuntimeEvent
from intergrax.runtime.events.spine_consolidation import build_platform_signal_event


def build_adaptive_signal_event(
    *,
    task_id: str,
    run_id: str,
    tenant_id: str,
    signal_id: str,
) -> RuntimeEvent:
    return build_platform_signal_event(
        kind="platform.adaptive.adaptive_signal_recorded",
        tenant_id=tenant_id,
        task_id=task_id,
        run_id=run_id,
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
    return build_platform_signal_event(
        kind="platform.adaptive.adaptive_proposal_submitted",
        tenant_id=tenant_id,
        task_id=task_id,
        run_id=run_id,
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
    return build_platform_signal_event(
        kind="platform.adaptive.adaptive_profile_applied",
        tenant_id=tenant_id,
        task_id=task_id,
        run_id=run_id,
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
    return build_platform_signal_event(
        kind="platform.adaptive.adaptive_profile_rollback",
        tenant_id=tenant_id,
        task_id=task_id,
        run_id=run_id,
        phase=ExecutionPhase.FINALIZATION,
        severity=EventSeverity.WARNING,
        payload={"restored_version_id": restored_version_id, "artifact_type": artifact_type},
    )


def build_adaptive_verification_failed_event(
    *,
    task_id: str,
    run_id: str,
    tenant_id: str,
    candidate_version_id: str,
    failure_reasons: list[str],
) -> RuntimeEvent:
    return build_platform_signal_event(
        kind="platform.adaptive.adaptive_verification_failed",
        tenant_id=tenant_id,
        task_id=task_id,
        run_id=run_id,
        phase=ExecutionPhase.FINALIZATION,
        severity=EventSeverity.WARNING,
        payload={
            "candidate_version_id": candidate_version_id,
            "failure_reasons": failure_reasons,
        },
    )


def build_adaptive_loop_blocked_event(
    *,
    task_id: str,
    run_id: str,
    tenant_id: str,
    loop_kind: str,
    reason: str,
) -> RuntimeEvent:
    return build_platform_signal_event(
        kind="platform.adaptive.adaptive_loop_blocked",
        tenant_id=tenant_id,
        task_id=task_id,
        run_id=run_id,
        phase=ExecutionPhase.FINALIZATION,
        severity=EventSeverity.WARNING,
        payload={"loop_kind": loop_kind, "reason": reason},
    )
