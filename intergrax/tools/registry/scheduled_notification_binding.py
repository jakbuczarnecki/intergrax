# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""In-memory scheduled notification store for ``notify.schedule`` (Tier-3 wiring default)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List
from uuid import uuid4

from intergrax.tools.registry.runtime_bindings import ScheduledNotificationBinding


@dataclass(frozen=True)
class ScheduledNotificationRecord:
    schedule_id: str
    tenant_id: str
    channel: str
    subject: str
    body: str
    deliver_at_utc: str
    status: str = "pending"


@dataclass
class InMemoryScheduledNotificationStore:
    """Process-local schedule index for catalog tools and tests."""

    _records: dict[str, ScheduledNotificationRecord] = field(default_factory=dict)

    def schedule(
        self,
        *,
        tenant_id: str,
        channel: str,
        subject: str,
        body: str,
        deliver_at_utc: str,
    ) -> str:
        schedule_id = f"sched_{uuid4().hex[:12]}"
        self._records[schedule_id] = ScheduledNotificationRecord(
            schedule_id=schedule_id,
            tenant_id=tenant_id.strip(),
            channel=channel.strip(),
            subject=subject.strip(),
            body=body.strip(),
            deliver_at_utc=deliver_at_utc.strip(),
        )
        return schedule_id

    def list_scheduled(
        self,
        tenant_id: str,
        *,
        limit: int = 50,
        status: str = "pending",
    ) -> List[dict[str, str]]:
        rows: list[dict[str, str]] = []
        for item in self._records.values():
            if item.tenant_id != tenant_id.strip():
                continue
            if status and item.status != status:
                continue
            rows.append(
                {
                    "schedule_id": item.schedule_id,
                    "tenant_id": item.tenant_id,
                    "channel": item.channel,
                    "subject": item.subject,
                    "deliver_at_utc": item.deliver_at_utc,
                    "status": item.status,
                }
            )
        rows.sort(key=lambda row: row["deliver_at_utc"])
        return rows[:limit]


def scheduled_notification_binding(
    store: ScheduledNotificationBinding | InMemoryScheduledNotificationStore | None,
) -> ScheduledNotificationBinding | None:
    if store is None:
        return None
    if isinstance(store, InMemoryScheduledNotificationStore):
        return store
    return store
