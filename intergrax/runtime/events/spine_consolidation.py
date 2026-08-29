# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Pre-release spine consolidation — platform kinds on DOMAIN_SIGNAL (OBS-EVOL-9.7)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Final

from intergrax.contracts.event_severity import EventSeverity
from intergrax.contracts.execution_identity import (
    require_active_execution_id,
    require_active_execution_identity,
    validate_run_id,
    validate_task_id,
)
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.runtime.events.event_taxonomy import EventCategory, RetentionClass
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType

_PUBLICATION_SPINE_TARGET_MAX = 56


@dataclass(frozen=True, slots=True)
class PlatformKindEntry:
    kind: str
    phase: ExecutionPhase
    ops_hint: str
    sample_rate: float = 1.0
    retention_class: RetentionClass = RetentionClass.OPERATIONAL


PLATFORM_KIND_CATALOG: Final[dict[str, PlatformKindEntry]] = {
    "platform.adaptive.adaptive_signal_recorded": PlatformKindEntry(
        kind="platform.adaptive.adaptive_signal_recorded",
        phase=ExecutionPhase.TRACE_PERSISTENCE,
        ops_hint="ops:adaptive",
    ),
    "platform.adaptive.adaptive_proposal_submitted": PlatformKindEntry(
        kind="platform.adaptive.adaptive_proposal_submitted",
        phase=ExecutionPhase.FINALIZATION,
        ops_hint="ops:adaptive",
    ),
    "platform.adaptive.adaptive_profile_applied": PlatformKindEntry(
        kind="platform.adaptive.adaptive_profile_applied",
        phase=ExecutionPhase.FINALIZATION,
        ops_hint="ops:adaptive",
    ),
    "platform.adaptive.adaptive_profile_rollback": PlatformKindEntry(
        kind="platform.adaptive.adaptive_profile_rollback",
        phase=ExecutionPhase.FINALIZATION,
        ops_hint="ops:alert",
        retention_class=RetentionClass.AUDIT,
    ),
    "platform.adaptive.adaptive_verification_failed": PlatformKindEntry(
        kind="platform.adaptive.adaptive_verification_failed",
        phase=ExecutionPhase.FINALIZATION,
        ops_hint="ops:alert",
    ),
    "platform.adaptive.adaptive_loop_blocked": PlatformKindEntry(
        kind="platform.adaptive.adaptive_loop_blocked",
        phase=ExecutionPhase.FINALIZATION,
        ops_hint="ops:alert",
    ),
    "platform.capacity.capacity_signal_collected": PlatformKindEntry(
        kind="platform.capacity.capacity_signal_collected",
        phase=ExecutionPhase.STEP_EXECUTION,
        ops_hint="ops:capacity",
        sample_rate=0.5,
    ),
    "platform.capacity.scale_evaluated": PlatformKindEntry(
        kind="platform.capacity.scale_evaluated",
        phase=ExecutionPhase.STEP_EXECUTION,
        ops_hint="ops:capacity",
    ),
    "platform.capacity.scale_requested": PlatformKindEntry(
        kind="platform.capacity.scale_requested",
        phase=ExecutionPhase.HUMAN_APPROVAL,
        ops_hint="ops:capacity",
    ),
    "platform.capacity.scale_approved": PlatformKindEntry(
        kind="platform.capacity.scale_approved",
        phase=ExecutionPhase.HUMAN_APPROVAL,
        ops_hint="ops:hitl",
    ),
    "platform.capacity.scale_denied": PlatformKindEntry(
        kind="platform.capacity.scale_denied",
        phase=ExecutionPhase.HUMAN_APPROVAL,
        ops_hint="ops:hitl",
    ),
    "platform.capacity.scale_applied": PlatformKindEntry(
        kind="platform.capacity.scale_applied",
        phase=ExecutionPhase.STEP_EXECUTION,
        ops_hint="ops:capacity",
    ),
    "platform.capacity.scale_failed": PlatformKindEntry(
        kind="platform.capacity.scale_failed",
        phase=ExecutionPhase.STEP_EXECUTION,
        ops_hint="ops:alert",
    ),
    "platform.capacity.autonomy_level_set": PlatformKindEntry(
        kind="platform.capacity.autonomy_level_set",
        phase=ExecutionPhase.INTAKE,
        ops_hint="ops:governance",
    ),
    "platform.capacity.autonomy_level_changed": PlatformKindEntry(
        kind="platform.capacity.autonomy_level_changed",
        phase=ExecutionPhase.STEP_EXECUTION,
        ops_hint="ops:governance",
    ),
    "platform.recovery.reboot": PlatformKindEntry(
        kind="platform.recovery.reboot",
        phase=ExecutionPhase.RETRY_HANDLING,
        ops_hint="ops:retry",
    ),
    "platform.hook.hook_blocked": PlatformKindEntry(
        kind="platform.hook.hook_blocked",
        phase=ExecutionPhase.STEP_EXECUTION,
        ops_hint="ops:alert",
    ),
    "platform.hook.hook_error": PlatformKindEntry(
        kind="platform.hook.hook_error",
        phase=ExecutionPhase.STEP_EXECUTION,
        ops_hint="ops:alert",
    ),
    "platform.hook.hook_timeout": PlatformKindEntry(
        kind="platform.hook.hook_timeout",
        phase=ExecutionPhase.STEP_EXECUTION,
        ops_hint="ops:alert",
    ),
    "platform.security.defense_blocked": PlatformKindEntry(
        kind="platform.security.defense_blocked",
        phase=ExecutionPhase.STEP_EXECUTION,
        ops_hint="ops:alert",
        retention_class=RetentionClass.AUDIT,
    ),
    "platform.security.encryption_denied": PlatformKindEntry(
        kind="platform.security.encryption_denied",
        phase=ExecutionPhase.STEP_EXECUTION,
        ops_hint="ops:alert",
        retention_class=RetentionClass.AUDIT,
    ),
    "platform.diagnostic.subsystem_failure": PlatformKindEntry(
        kind="platform.diagnostic.subsystem_failure",
        phase=ExecutionPhase.STEP_EXECUTION,
        ops_hint="ops:alert",
        retention_class=RetentionClass.AUDIT,
    ),
}


def get_platform_kind_entry(kind: str) -> PlatformKindEntry | None:
    return PLATFORM_KIND_CATALOG.get(kind)


def should_persist_platform_kind(kind: str, event_id: str) -> bool:
    entry = get_platform_kind_entry(kind)
    if entry is None or entry.sample_rate >= 1.0:
        return True
    digest = int(hashlib.sha256(event_id.encode("utf-8")).hexdigest()[:8], 16)
    return (digest % 10_000) < int(entry.sample_rate * 10_000)


def build_platform_signal_event(
    *,
    kind: str,
    task_id: str,
    run_id: str,
    phase: ExecutionPhase | None = None,
    payload: dict | None = None,
    tenant_id: str | None = None,
    severity: EventSeverity = EventSeverity.INFO,
    node_id: str | None = None,
    agent_id: str | None = None,
    step_id: str | None = None,
    correlation_id: str = "",
    parent_event_id: str | None = None,
) -> RuntimeEvent:
    """Emit a consolidated platform domain signal on the unified bus."""
    entry = get_platform_kind_entry(kind)
    if entry is None:
        raise ValueError(f"unknown platform event_kind: {kind!r}")
    body = dict(payload or {})
    active_run_id, attempt_id = require_active_execution_identity()
    execution_id = require_active_execution_id()
    resolved_run_id = validate_run_id(run_id)
    if resolved_run_id != active_run_id:
        raise RuntimeError("run_id conflicts with active execution identity")
    return RuntimeEvent(
        tenant_id=tenant_id,
        task_id=validate_task_id(task_id),
        run_id=resolved_run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
        node_id=node_id,
        agent_id=agent_id,
        step_id=step_id,
        event_type=RuntimeEventType.DOMAIN_SIGNAL,
        event_kind=kind,
        event_category=EventCategory.PLATFORM,
        ops_hint=entry.ops_hint,
        phase=phase or entry.phase,
        severity=severity,
        correlation_id=correlation_id or task_id,
        parent_event_id=parent_event_id,
        payload=body,
    )


def publication_spine_type_count() -> int:
    return len(RuntimeEventType)


def assert_publication_spine_budget() -> None:
    count = publication_spine_type_count()
    if count > _PUBLICATION_SPINE_TARGET_MAX:
        raise AssertionError(
            f"RuntimeEventType has {count} members; publication budget is {_PUBLICATION_SPINE_TARGET_MAX}"
        )
