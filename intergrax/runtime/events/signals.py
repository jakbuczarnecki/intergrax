# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Public HOS emit APIs for platform and domain signals (OBS-EVOL-9.3)."""

from __future__ import annotations

import re

from intergrax.contracts.event_severity import EventSeverity
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.runtime.events.emit_context import EmitContext
from intergrax.runtime.events.event_catalog import category_for_event_kind, get_catalog_entry
from intergrax.runtime.events.payload_registry import runtime_event_with_payload
from intergrax.runtime.events.payloads import RuntimeEventPayload
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType

_EVENT_KIND_RE = re.compile(
    r"^(agents|applications|platform|intergrax)\.[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+$"
)


class DomainSignalError(ValueError):
    """Raised when a domain signal kind or payload is invalid."""


def validate_event_kind(kind: str) -> None:
    if not _EVENT_KIND_RE.match(kind):
        raise DomainSignalError(
            "event_kind must be a namespaced lowercase id "
            "(e.g. agents.legal.clause_flagged)"
        )


def emit_platform_event(
    ctx: EmitContext,
    *,
    event_type: RuntimeEventType,
    payload: RuntimeEventPayload,
    phase: ExecutionPhase | None = None,
    severity: EventSeverity = EventSeverity.INFO,
    agent_id: str | None = None,
    step_id: str | None = None,
    node_id: str | None = None,
) -> RuntimeEvent:
    """Emit a platform lifecycle spine event (Tier-0/1 only)."""
    entry = get_catalog_entry(event_type)
    if entry is None:
        raise DomainSignalError(f"unknown spine event_type: {event_type.value}")
    event = RuntimeEvent(
        tenant_id=ctx.tenant_id,
        task_id=ctx.task_id,
        run_id=ctx.run_id,
        node_id=node_id,
        agent_id=agent_id,
        step_id=step_id,
        event_type=event_type,
        phase=phase or entry.phase,
        severity=severity,
        correlation_id=ctx.effective_correlation_id,
        parent_event_id=ctx.parent_event_id,
    )
    event = runtime_event_with_payload(event, payload)
    if ctx.bus is not None:
        ctx.bus.record(event, tenant_id=ctx.tenant_id)
    return event


def emit_domain_signal(
    ctx: EmitContext,
    *,
    kind: str,
    payload: RuntimeEventPayload,
    severity: EventSeverity = EventSeverity.INFO,
    agent_id: str | None = None,
    step_id: str | None = None,
    node_id: str | None = None,
) -> RuntimeEvent:
    """Emit a Tier-2/3 domain signal on the unified bus."""
    validate_event_kind(kind)
    safe_payload = payload.redact() if ctx.production_mode else payload
    event = RuntimeEvent(
        tenant_id=ctx.tenant_id,
        task_id=ctx.task_id,
        run_id=ctx.run_id,
        node_id=node_id,
        agent_id=agent_id,
        step_id=step_id,
        event_type=RuntimeEventType.DOMAIN_SIGNAL,
        event_kind=kind,
        event_category=category_for_event_kind(kind),
        ops_hint="ops:domain_signal",
        phase=ExecutionPhase.STEP_EXECUTION,
        severity=severity,
        correlation_id=ctx.effective_correlation_id,
        parent_event_id=ctx.parent_event_id,
    )
    event = runtime_event_with_payload(event, safe_payload)
    if ctx.bus is not None:
        ctx.bus.record(event, tenant_id=ctx.tenant_id)
    return event
