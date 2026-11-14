# © Artur Czarnecki. All rights reserved.

"""Append-only security audit trail (IDEAL-23.4)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol
from uuid import uuid4

from intergrax.contracts.event_severity import EventSeverity
from intergrax.contracts.execution_evidence.persistence_port import EvidencePersistencePort
from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    TaskId,
    validate_attempt_id,
    validate_event_id,
    validate_execution_id,
    validate_run_id,
    validate_task_id,
)
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.contracts.runtime_event import RuntimeEvent
from intergrax.contracts.runtime_event_type import RuntimeEventType
from intergrax.core.security.secret_safety import (
    SecretSafeValidationPolicy,
    validate_secret_safe_value,
)
from intergrax.runtime.events.payload_registry import register_payload_schema
from intergrax.runtime.events.payloads.base import RuntimeEventPayload


@dataclass(frozen=True, slots=True)
class SecurityAuditEntry:
    entry_id: str
    tenant_id: str
    action: str
    actor_id: str
    resource: str
    recorded_at: datetime
    metadata: dict[str, Any]


@dataclass
class SecurityAuditTrail:
    """In-memory append-only trail (tests and lab hosts); product hosts use the evidence-plane trail."""

    _entries: list[SecurityAuditEntry] = field(default_factory=list)

    def append(
        self,
        *,
        tenant_id: str,
        action: str,
        actor_id: str,
        resource: str,
        metadata: dict[str, Any] | None = None,
    ) -> SecurityAuditEntry:
        entry = SecurityAuditEntry(
            entry_id=f"audit_{uuid4().hex}",
            tenant_id=tenant_id,
            action=action,
            actor_id=actor_id,
            resource=resource,
            recorded_at=datetime.now(UTC),
            metadata=dict(metadata or {}),
        )
        self._entries.append(entry)
        return entry

    def list_entries(self, *, tenant_id: str | None = None) -> tuple[SecurityAuditEntry, ...]:
        if tenant_id is None:
            return tuple(self._entries)
        return tuple(entry for entry in self._entries if entry.tenant_id == tenant_id)


# ---------------------------------------------------------------------------
# PDC-D6-R020 - durable trail over the canonical evidence persistence plane
# ---------------------------------------------------------------------------

SECURITY_AUDIT_EVENT_KIND = "security.audit_entry"

_AUDIT_METADATA_POLICY = SecretSafeValidationPolicy(
    forbidden_key_names=frozenset({"token", "authorization", "password", "secret"}),
    forbidden_key_fragments=frozenset(
        {"secret", "password", "passwd", "api_key", "apikey", "private_key", "signing_key", "credential"}
    ),
    split_key_segments=True,
)


class SecurityAuditPayloadV1(RuntimeEventPayload):
    """Typed ``RuntimeEvent`` body carrying one security audit entry."""

    schema_id = "security.audit_entry.v1"

    entry_id: str
    action: str
    actor_id: str
    resource: str
    metadata: dict[str, Any] = {}


register_payload_schema(SecurityAuditPayloadV1, extension=True)


class SecurityAuditSink(Protocol):
    """Append-only trail surface shared by the in-memory and evidence-plane trails."""

    def append(
        self,
        *,
        tenant_id: str,
        action: str,
        actor_id: str,
        resource: str,
        metadata: dict[str, Any] | None = None,
    ) -> SecurityAuditEntry: ...

    def list_entries(self, *, tenant_id: str | None = None) -> tuple[SecurityAuditEntry, ...]: ...


def _deterministic_suffix(label: str, tenant_id: str) -> str:
    return hashlib.sha256(f"security-audit:{label}:{tenant_id}".encode()).hexdigest()[:32]


@dataclass(frozen=True, slots=True)
class _TenantStream:
    task_id: TaskId
    run_id: RunId
    attempt_id: AttemptId
    execution_id: ExecutionId


def _tenant_stream(tenant_id: str) -> _TenantStream:
    """Stable per-tenant execution identity so a tenant's entries share one evidence stream."""
    return _TenantStream(
        task_id=validate_task_id(f"task_{_deterministic_suffix('task', tenant_id)}"),
        run_id=validate_run_id(f"run_{_deterministic_suffix('run', tenant_id)}"),
        attempt_id=validate_attempt_id(f"attempt_{_deterministic_suffix('attempt', tenant_id)}"),
        execution_id=validate_execution_id(f"exec_{_deterministic_suffix('exec', tenant_id)}"),
    )


class EvidencePlaneSecurityAuditTrail:
    """Append-only, tenant-scoped security audit trail persisted via ``EvidencePersistencePort``.

    No security-owned store: each entry is a canonical ``RuntimeEvent`` appended through the
    platform evidence plane. The port exposes no update or delete, so the trail is append-only
    by construction; reads are tenant-scoped by the port.
    """

    def __init__(self, evidence_persistence: EvidencePersistencePort) -> None:
        self._evidence = evidence_persistence

    def append(
        self,
        *,
        tenant_id: str,
        action: str,
        actor_id: str,
        resource: str,
        metadata: dict[str, Any] | None = None,
    ) -> SecurityAuditEntry:
        if not isinstance(tenant_id, str) or not tenant_id.strip():
            raise ValueError("security audit entry requires a non-empty tenant_id")
        tenant = tenant_id.strip()
        safe_metadata = json.loads(json.dumps(dict(metadata or {}), default=str))
        validate_secret_safe_value(
            safe_metadata,
            policy=_AUDIT_METADATA_POLICY,
            context_label="security audit metadata",
        )
        suffix = uuid4().hex
        recorded_at = datetime.now(UTC)
        entry = SecurityAuditEntry(
            entry_id=f"audit_{suffix}",
            tenant_id=tenant,
            action=action,
            actor_id=actor_id,
            resource=resource,
            recorded_at=recorded_at,
            metadata=safe_metadata,
        )
        payload = SecurityAuditPayloadV1(
            entry_id=entry.entry_id,
            action=action,
            actor_id=actor_id,
            resource=resource,
            metadata=safe_metadata,
        )
        stream = _tenant_stream(tenant)
        event = RuntimeEvent(
            event_id=validate_event_id(f"evt_{suffix}"),
            tenant_id=tenant,
            task_id=stream.task_id,
            run_id=stream.run_id,
            attempt_id=stream.attempt_id,
            execution_id=stream.execution_id,
            event_type=RuntimeEventType.DOMAIN_SIGNAL,
            event_kind=SECURITY_AUDIT_EVENT_KIND,
            phase=ExecutionPhase.STEP_EXECUTION,
            severity=EventSeverity.INFO,
            timestamp=recorded_at,
            correlation_id=entry.entry_id,
            payload=payload.to_envelope(),
        )
        self._evidence.append(event, tenant_id=tenant)
        return entry

    def list_entries(self, *, tenant_id: str | None = None) -> tuple[SecurityAuditEntry, ...]:
        if tenant_id is None or not tenant_id.strip():
            raise ValueError("evidence-plane security audit trail is tenant-scoped: tenant_id required")
        tenant = tenant_id.strip()
        events = self._evidence.list_for_task(
            _tenant_stream(tenant).task_id,
            tenant_id=tenant,
            limit=1000,
        )
        entries: list[SecurityAuditEntry] = []
        for event in events:
            if event.event_kind != SECURITY_AUDIT_EVENT_KIND or event.tenant_id != tenant:
                continue
            body = SecurityAuditPayloadV1.from_envelope(event.payload)
            entries.append(
                SecurityAuditEntry(
                    entry_id=body.entry_id,
                    tenant_id=tenant,
                    action=body.action,
                    actor_id=body.actor_id,
                    resource=body.resource,
                    recorded_at=event.timestamp,
                    metadata=dict(body.metadata),
                )
            )
        return tuple(entries)
