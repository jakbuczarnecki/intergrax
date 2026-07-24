# © Artur Czarnecki. All rights reserved.

"""In-process Slack workspace selection (no DocumentStore / no TTL)."""

from __future__ import annotations

import threading
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SlackWorkspaceSelection:
    """Selected LKW workspace for one Slack actor (safe name for rendering)."""

    workspace_id: str
    workspace_name: str


def slack_selection_actor_key(*, team_id: str, user_id: str) -> str:
    """Canonical in-memory key: Slack team_id + approved user_id."""
    return f"{(team_id or '').strip()}:{(user_id or '').strip()}"


class InMemorySlackWorkspaceSelectionStore:
    """Thread-safe process-local selection; cleared on restart."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_actor: dict[str, SlackWorkspaceSelection] = {}
        self._configured_suppressed: set[str] = set()

    def get(self, actor_key: str) -> SlackWorkspaceSelection | None:
        key = (actor_key or "").strip()
        if not key:
            return None
        with self._lock:
            return self._by_actor.get(key)

    def set(self, actor_key: str, selection: SlackWorkspaceSelection) -> None:
        key = (actor_key or "").strip()
        if not key:
            return
        workspace_id = (selection.workspace_id or "").strip()
        workspace_name = (selection.workspace_name or "").strip()
        if not workspace_id or not workspace_name:
            return
        with self._lock:
            self._by_actor[key] = SlackWorkspaceSelection(
                workspace_id=workspace_id,
                workspace_name=workspace_name,
            )
            self._configured_suppressed.discard(key)

    def clear(self, actor_key: str) -> None:
        key = (actor_key or "").strip()
        if not key:
            return
        with self._lock:
            self._by_actor.pop(key, None)

    def suppress_configured(self, actor_key: str) -> None:
        """Mark configured fallback unavailable after it was deleted in-process."""
        key = (actor_key or "").strip()
        if not key:
            return
        with self._lock:
            self._configured_suppressed.add(key)

    def clear_configured_suppression(self, actor_key: str) -> None:
        key = (actor_key or "").strip()
        if not key:
            return
        with self._lock:
            self._configured_suppressed.discard(key)

    def is_configured_suppressed(self, actor_key: str) -> bool:
        key = (actor_key or "").strip()
        if not key:
            return False
        with self._lock:
            return key in self._configured_suppressed
