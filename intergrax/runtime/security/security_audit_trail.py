# © Artur Czarnecki. All rights reserved.

"""Append-only security audit trail (IDEAL-23.4)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


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
    """In-memory append-only trail; production hosts may persist to WORM store."""

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
            recorded_at=datetime.now(timezone.utc),
            metadata=dict(metadata or {}),
        )
        self._entries.append(entry)
        return entry

    def list_entries(self, *, tenant_id: str | None = None) -> tuple[SecurityAuditEntry, ...]:
        if tenant_id is None:
            return tuple(self._entries)
        return tuple(entry for entry in self._entries if entry.tenant_id == tenant_id)
