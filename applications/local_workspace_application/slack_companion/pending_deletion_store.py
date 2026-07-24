# © Artur Czarnecki. All rights reserved.

"""Process-local pending workspace deletion (no DocumentStore / no MongoDB)."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

_DEFAULT_TTL = timedelta(minutes=5)


@dataclass(frozen=True, slots=True)
class SlackPendingWorkspaceDeletion:
    """Actor-scoped pending delete confirmation payload."""

    workspace_id: str
    workspace_name: str
    requested_at: datetime
    expires_at: datetime


def slack_pending_deletion_actor_key(*, team_id: str, user_id: str) -> str:
    return f"{(team_id or '').strip()}:{(user_id or '').strip()}"


class InMemorySlackPendingDeletionStore:
    """Thread-safe process-local pending deletion; single-use via consume_valid."""

    def __init__(self, *, ttl: timedelta = _DEFAULT_TTL) -> None:
        self._lock = threading.Lock()
        self._by_actor: dict[str, SlackPendingWorkspaceDeletion] = {}
        self._ttl = ttl

    def get(self, actor_key: str) -> SlackPendingWorkspaceDeletion | None:
        key = (actor_key or "").strip()
        if not key:
            return None
        now = datetime.now(UTC)
        with self._lock:
            pending = self._by_actor.get(key)
            if pending is None:
                return None
            if pending.expires_at <= now:
                self._by_actor.pop(key, None)
                return None
            return pending

    def set(
        self,
        actor_key: str,
        *,
        workspace_id: str,
        workspace_name: str,
    ) -> SlackPendingWorkspaceDeletion | None:
        key = (actor_key or "").strip()
        workspace_id = (workspace_id or "").strip()
        workspace_name = (workspace_name or "").strip()
        if not key or not workspace_id or not workspace_name:
            return None
        now = datetime.now(UTC)
        pending = SlackPendingWorkspaceDeletion(
            workspace_id=workspace_id,
            workspace_name=workspace_name,
            requested_at=now,
            expires_at=now + self._ttl,
        )
        with self._lock:
            self._by_actor[key] = pending
        return pending

    def clear(self, actor_key: str) -> None:
        key = (actor_key or "").strip()
        if not key:
            return
        with self._lock:
            self._by_actor.pop(key, None)

    def consume_valid(self, actor_key: str) -> SlackPendingWorkspaceDeletion | None:
        """Return and clear a non-expired pending deletion, else None."""
        key = (actor_key or "").strip()
        if not key:
            return None
        now = datetime.now(UTC)
        with self._lock:
            pending = self._by_actor.pop(key, None)
            if pending is None:
                return None
            if pending.expires_at <= now:
                return None
            return pending
