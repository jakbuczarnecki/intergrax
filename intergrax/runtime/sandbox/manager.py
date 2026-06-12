# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Lifecycle manager for active sandbox sessions."""

from __future__ import annotations

import os
from pathlib import Path

from intergrax.runtime.sandbox.session import SandboxSession

ENV_SANDBOX_ROOT = "INTERGRAX_SANDBOX_ROOT"
DEFAULT_SANDBOX_ROOT = Path("build") / "sandbox_sessions"


def resolve_sandbox_root(explicit: Path | None = None) -> Path:
    if explicit is not None:
        return explicit
    env = os.environ.get(ENV_SANDBOX_ROOT, "").strip()
    if env:
        return Path(env)
    return DEFAULT_SANDBOX_ROOT


class SandboxSessionManager:
    """Creates, tracks, and disposes sandbox sessions per tenant/task."""

    def __init__(self, *, root: Path | None = None) -> None:
        self._root = resolve_sandbox_root(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._active: dict[str, SandboxSession] = {}

    @property
    def root(self) -> Path:
        return self._root

    def open_or_create(self, *, tenant_id: str, task_id: str) -> SandboxSession:
        key = f"{tenant_id}:{task_id}"
        existing = self._active.get(key)
        if existing is not None and existing.exists_on_disk():
            return existing

        session = SandboxSession.create(
            self._root,
            tenant_id=tenant_id,
            task_id=task_id,
        )
        self._active[key] = session
        self._active[session.session_id] = session
        return session

    def get(self, session_id: str) -> SandboxSession | None:
        return self._active.get(session_id)

    def cleanup(self, session_id: str) -> bool:
        session = self._active.pop(session_id, None)
        if session is None:
            return False
        key = f"{session.tenant_id}:{session.task_id}"
        self._active.pop(key, None)
        session.cleanup()
        return True

    def cleanup_for_task(self, *, tenant_id: str, task_id: str) -> bool:
        key = f"{tenant_id}:{task_id}"
        session = self._active.pop(key, None)
        if session is None:
            return False
        self._active.pop(session.session_id, None)
        session.cleanup()
        return True

    @property
    def active_count(self) -> int:
        """Number of distinct active sessions (dedupes tenant:task and id keys)."""
        return len({id(session) for session in self._active.values()})

    def dispose_all_active(self) -> int:
        """Dispose every tracked session — host shutdown / lifespan teardown."""
        seen: set[int] = set()
        disposed = 0
        for session in list(self._active.values()):
            token = id(session)
            if token in seen:
                continue
            seen.add(token)
            session.cleanup()
            disposed += 1
        self._active.clear()
        return disposed
