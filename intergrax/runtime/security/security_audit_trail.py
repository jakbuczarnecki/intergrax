# © Artur Czarnecki. All rights reserved.

"""Append-only security audit trail (IDEAL-23.4)."""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol
from uuid import uuid4

from intergrax.contracts.event_severity import EventSeverity
from intergrax.contracts.execution_event_position import ExecutionEventPosition
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
from intergrax.contracts.positioned_runtime_event import PositionedRuntimeEvent
from intergrax.contracts.runtime_event import RuntimeEvent
from intergrax.contracts.runtime_event_type import RuntimeEventType
from intergrax.core.security import secret_safety
from intergrax.runtime.events.evidence_persistence_adapter import as_evidence_persistence_port
from intergrax.runtime.events.payload_registry import register_payload_schema
from intergrax.runtime.events.payloads.base import RuntimeEventPayload

#: ``prev_digest`` of the first entry of a tenant's chain (EX-D6-01, F-D6-02).
SECURITY_AUDIT_CHAIN_GENESIS = "sha256:" + "0" * 64


@dataclass(frozen=True, slots=True)
class SecurityAuditEntry:
    entry_id: str
    tenant_id: str
    action: str
    actor_id: str
    resource: str
    recorded_at: datetime
    metadata: dict[str, Any]
    #: Digest of the previous entry of the same tenant (``SECURITY_AUDIT_CHAIN_GENESIS`` for the first).
    prev_digest: str = ""
    #: sha256 over the canonical JSON of the entry fields and ``prev_digest``; empty for unchained entries.
    entry_digest: str = ""


@dataclass(frozen=True, slots=True)
class SecurityAuditChainReport:
    """Result of :meth:`verify_chain` for one tenant; a clean chain has ``first_broken_index is None``.

    Honest limit: a hash chain detects a modified or a removed interior entry. It does not detect a
    truncated tail or a rewrite of the whole chain by a writer of the store, because there is no
    external anchor of the head; ``head_digest`` is exposed so that a caller can anchor it elsewhere.
    """

    verified_count: int
    head_digest: str
    first_broken_index: int | None = None
    reason: str | None = None
    #: Entries written before the chain existed (no digests); they are reported, not verified.
    unchained_count: int = 0

    @property
    def intact(self) -> bool:
        return self.first_broken_index is None


class AuditTrailListingTruncated(RuntimeError):
    """The evidence port returned exactly its limit: the listing may be truncated (F-D6-03).

    Raised instead of silently returning the oldest entries only; the port has no cursor for
    ``list_for_task``.
    """

    def __init__(self, tenant_id: str, limit: int) -> None:
        super().__init__(f"security audit listing for tenant {tenant_id!r} reached the port limit of {limit} entries")
        self.tenant_id = tenant_id
        self.limit = limit


def _canonical_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    redacted = secret_safety.redact_secret_shaped_mapping(json.loads(json.dumps(dict(metadata or {}), default=str)))
    return json.loads(json.dumps(redacted, default=str))


def compute_security_audit_entry_digest(
    *,
    entry_id: str,
    tenant_id: str,
    action: str,
    actor_id: str,
    resource: str,
    recorded_at: datetime,
    metadata: dict[str, Any],
    prev_digest: str,
) -> str:
    """Chain digest of one entry: sha256 over its canonical JSON including ``prev_digest``."""
    body = {
        "entry_id": entry_id,
        "tenant_id": tenant_id,
        "action": action,
        "actor_id": actor_id,
        "resource": resource,
        "recorded_at": recorded_at.astimezone(UTC).isoformat(),
        "metadata": metadata,
        "prev_digest": prev_digest,
    }
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _digest_of(entry: SecurityAuditEntry) -> str:
    return compute_security_audit_entry_digest(
        entry_id=entry.entry_id,
        tenant_id=entry.tenant_id,
        action=entry.action,
        actor_id=entry.actor_id,
        resource=entry.resource,
        recorded_at=entry.recorded_at,
        metadata=entry.metadata,
        prev_digest=entry.prev_digest,
    )


def _verify_entries(entries: list[SecurityAuditEntry]) -> SecurityAuditChainReport:
    head = SECURITY_AUDIT_CHAIN_GENESIS
    verified = 0
    unchained = 0
    for index, entry in enumerate(entries):
        if not entry.entry_digest and not entry.prev_digest:
            if verified:
                return SecurityAuditChainReport(verified, head, index, "unchained_entry_after_chain_start")
            unchained += 1
            continue
        if entry.prev_digest != head:
            return SecurityAuditChainReport(verified, head, index, "prev_digest_mismatch", unchained)
        if _digest_of(entry) != entry.entry_digest:
            return SecurityAuditChainReport(verified, head, index, "entry_digest_mismatch", unchained)
        head = entry.entry_digest
        verified += 1
    return SecurityAuditChainReport(verified, head, None, None, unchained)


@dataclass
class SecurityAuditTrail:
    """In-memory append-only trail (tests and lab hosts); product hosts use the evidence-plane trail.

    Entries of a tenant are hash-chained like the evidence-plane trail and secret-shaped metadata
    values are redacted before storage.
    """

    _entries: list[SecurityAuditEntry] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def append(
        self,
        *,
        tenant_id: str,
        action: str,
        actor_id: str,
        resource: str,
        metadata: dict[str, Any] | None = None,
    ) -> SecurityAuditEntry:
        safe_metadata = secret_safety.redact_secret_shaped_mapping(dict(metadata or {}))
        with self._lock:
            prev = self.head_digest(tenant_id)
            recorded_at = datetime.now(UTC)
            entry_id = f"audit_{uuid4().hex}"
            entry = SecurityAuditEntry(
                entry_id=entry_id,
                tenant_id=tenant_id,
                action=action,
                actor_id=actor_id,
                resource=resource,
                recorded_at=recorded_at,
                metadata=safe_metadata,
                prev_digest=prev,
                entry_digest=compute_security_audit_entry_digest(
                    entry_id=entry_id,
                    tenant_id=tenant_id,
                    action=action,
                    actor_id=actor_id,
                    resource=resource,
                    recorded_at=recorded_at,
                    metadata=safe_metadata,
                    prev_digest=prev,
                ),
            )
            self._entries.append(entry)
        return entry

    def list_entries(self, *, tenant_id: str | None = None) -> tuple[SecurityAuditEntry, ...]:
        if tenant_id is None:
            return tuple(self._entries)
        return tuple(entry for entry in self._entries if entry.tenant_id == tenant_id)

    def head_digest(self, tenant_id: str) -> str:
        """Digest of the tenant's last chained entry (``SECURITY_AUDIT_CHAIN_GENESIS`` when empty)."""
        for entry in reversed(self._entries):
            if entry.tenant_id == tenant_id and entry.entry_digest:
                return entry.entry_digest
        return SECURITY_AUDIT_CHAIN_GENESIS

    def verify_chain(self, tenant_id: str) -> SecurityAuditChainReport:
        """Recompute the tenant's chain; report the first modified or missing interior entry."""
        return _verify_entries([entry for entry in self._entries if entry.tenant_id == tenant_id])


# ---------------------------------------------------------------------------
# PDC-D6-R020 - durable trail over the canonical evidence persistence plane
# ---------------------------------------------------------------------------

SECURITY_AUDIT_EVENT_KIND = "security.audit_entry"

_AUDIT_METADATA_POLICY = secret_safety.SecretSafeValidationPolicy(
    forbidden_key_names=frozenset({"token", "authorization", "password", "secret"}),
    forbidden_key_fragments=frozenset(
        {"secret", "password", "passwd", "api_key", "apikey", "private_key", "signing_key", "credential"}
    ),
    split_key_segments=True,
)

_PAGE_LIMIT = 1000


class SecurityAuditPayloadV1(RuntimeEventPayload):
    """Typed ``RuntimeEvent`` body carrying one security audit entry."""

    schema_id = "security.audit_entry.v1"

    entry_id: str
    action: str
    actor_id: str
    resource: str
    metadata: dict[str, Any] = {}
    # Chain fields (EX-D6-01); empty for events stored before the chain existed.
    prev_digest: str = ""
    entry_digest: str = ""


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


def _entry_from_event(event: RuntimeEvent, tenant: str) -> SecurityAuditEntry:
    body = SecurityAuditPayloadV1.from_envelope(event.payload)
    return SecurityAuditEntry(
        entry_id=body.entry_id,
        tenant_id=tenant,
        action=body.action,
        actor_id=body.actor_id,
        resource=body.resource,
        recorded_at=event.timestamp,
        metadata=dict(body.metadata),
        prev_digest=body.prev_digest,
        entry_digest=body.entry_digest,
    )


@dataclass(slots=True)
class _ChainHead:
    digest: str
    position: ExecutionEventPosition | None


class EvidencePlaneSecurityAuditTrail:
    """Append-only, tenant-scoped security audit trail persisted via ``EvidencePersistencePort``.

    No security-owned store: each entry is a canonical ``RuntimeEvent`` appended through the
    platform evidence plane. The port exposes no update or delete, so the trail is append-only
    by construction; reads are tenant-scoped by the port.

    The injected port is normalized with ``as_evidence_persistence_port``: every write goes through a
    ``CanonicalRuntimeEventWriteValidatedPort`` (canonical prepare and schema validation), whatever
    port the host passes. This is what the COMPAT-X classification of ``append`` as a canonical
    validating path relies on.

    Tamper evidence (EX-D6-01, F-D6-02): a tenant's entries form a hash chain (``prev_digest`` /
    ``entry_digest``); :meth:`verify_chain` reads the stored entries back and reports the first
    modified or removed interior entry. A per-trail lock serializes the read-head-then-append step;
    writers in other processes are not serialized by it. Tail truncation and a full rewrite of the
    chain are not detected without an external anchor of :meth:`head_digest`.
    """

    def __init__(self, evidence_persistence: EvidencePersistencePort) -> None:
        port = as_evidence_persistence_port(evidence_persistence)
        if port is None:
            raise TypeError("EvidencePlaneSecurityAuditTrail requires an evidence persistence port")
        self._evidence = port
        self._lock = threading.Lock()
        self._heads: dict[str, _ChainHead] = {}

    def _iter_stream(self, tenant: str, after: ExecutionEventPosition | None) -> list[PositionedRuntimeEvent]:
        """Every stored event of the tenant's stream after ``after``, oldest first (paged)."""
        run_id = _tenant_stream(tenant).run_id
        collected: list[PositionedRuntimeEvent] = []
        cursor = after
        while True:
            page = self._evidence.list_positioned_for_run(run_id, tenant_id=tenant, limit=_PAGE_LIMIT, after=cursor)
            collected.extend(page)
            if len(page) < _PAGE_LIMIT:
                return collected
            cursor = page[-1].position

    def _refresh_head(self, tenant: str) -> _ChainHead:
        head = self._heads.get(tenant) or _ChainHead(SECURITY_AUDIT_CHAIN_GENESIS, None)
        for positioned in self._iter_stream(tenant, head.position):
            head.position = positioned.position
            event = positioned.event
            if event.event_kind != SECURITY_AUDIT_EVENT_KIND or event.tenant_id != tenant:
                continue
            digest = SecurityAuditPayloadV1.from_envelope(event.payload).entry_digest
            if digest:
                head.digest = digest
        self._heads[tenant] = head
        return head

    def head_digest(self, tenant_id: str) -> str:
        """Digest of the tenant's last chained entry as stored (``SECURITY_AUDIT_CHAIN_GENESIS`` when empty)."""
        with self._lock:
            return self._refresh_head(tenant_id.strip()).digest

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
        safe_metadata = _canonical_metadata(metadata)
        secret_safety.validate_secret_safe_value(
            safe_metadata,
            policy=_AUDIT_METADATA_POLICY,
            context_label="security audit metadata",
        )
        with self._lock:
            prev = self._refresh_head(tenant).digest
            suffix = uuid4().hex
            recorded_at = datetime.now(UTC)
            entry_id = f"audit_{suffix}"
            digest = compute_security_audit_entry_digest(
                entry_id=entry_id,
                tenant_id=tenant,
                action=action,
                actor_id=actor_id,
                resource=resource,
                recorded_at=recorded_at,
                metadata=safe_metadata,
                prev_digest=prev,
            )
            entry = SecurityAuditEntry(
                entry_id=entry_id,
                tenant_id=tenant,
                action=action,
                actor_id=actor_id,
                resource=resource,
                recorded_at=recorded_at,
                metadata=safe_metadata,
                prev_digest=prev,
                entry_digest=digest,
            )
            payload = SecurityAuditPayloadV1(
                entry_id=entry.entry_id,
                action=action,
                actor_id=actor_id,
                resource=resource,
                metadata=safe_metadata,
                prev_digest=prev,
                entry_digest=digest,
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
            self._refresh_head(tenant)
        return entry

    def list_entries(self, *, tenant_id: str | None = None) -> tuple[SecurityAuditEntry, ...]:
        if tenant_id is None or not tenant_id.strip():
            raise ValueError("evidence-plane security audit trail is tenant-scoped: tenant_id required")
        tenant = tenant_id.strip()
        events = self._evidence.list_for_task(
            _tenant_stream(tenant).task_id,
            tenant_id=tenant,
            limit=_PAGE_LIMIT,
        )
        if len(events) >= _PAGE_LIMIT:
            raise AuditTrailListingTruncated(tenant, _PAGE_LIMIT)
        entries: list[SecurityAuditEntry] = []
        for event in events:
            if event.event_kind != SECURITY_AUDIT_EVENT_KIND or event.tenant_id != tenant:
                continue
            entries.append(_entry_from_event(event, tenant))
        return tuple(entries)

    def verify_chain(self, tenant_id: str) -> SecurityAuditChainReport:
        """Read every stored entry of the tenant back and recompute the chain."""
        if not tenant_id.strip():
            raise ValueError("evidence-plane security audit trail is tenant-scoped: tenant_id required")
        tenant = tenant_id.strip()
        entries = [
            _entry_from_event(positioned.event, tenant)
            for positioned in self._iter_stream(tenant, None)
            if positioned.event.event_kind == SECURITY_AUDIT_EVENT_KIND and positioned.event.tenant_id == tenant
        ]
        return _verify_entries(entries)
