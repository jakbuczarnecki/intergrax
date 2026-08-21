# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""CodeCraftSessionManager — craft_id lifecycle per tenant/task (ECC-2)."""

from __future__ import annotations

from threading import RLock
from uuid import uuid4

from intergrax.codecraft.contracts import CodeCraftSession
from intergrax.runtime.codecraft.ownership import (
    CodeCraftOwnershipError,
    CodeCraftSessionOwnership,
    matches_session_ownership,
)
from intergrax.tools.registry.wiring import ToolWiringContext


class CodeCraftSessionManager:
    """Task-scoped in-memory craft session store."""

    def __init__(self) -> None:
        self._sessions: dict[str, CodeCraftSession] = {}
        self._lock = RLock()

    def open(
        self,
        *,
        goal: str,
        ownership: CodeCraftSessionOwnership,
        mode: str,
        language: str = "python",
        max_iterations: int = 8,
        craft_id: str | None = None,
    ) -> CodeCraftSession:
        sid = craft_id or f"craft_{uuid4().hex[:12]}"
        with self._lock:
            existing = self._sessions.get(sid)
            if existing is not None and not existing.disposed:
                if matches_session_ownership(
                    existing.tenant_id,
                    existing.task_id,
                    existing.run_id,
                    ownership,
                ):
                    raise CodeCraftOwnershipError("craft_session_already_open")
                raise CodeCraftOwnershipError("craft_session_ownership_conflict")
            session = CodeCraftSession(
                craft_id=sid,
                task_id=ownership.task_id,
                tenant_id=ownership.tenant_id,
                run_id=ownership.run_id,
                goal=goal,
                mode=mode,
                language=language,
                max_iterations=max_iterations,
            )
            self._sessions[sid] = session
        return session

    def get_owned(self, craft_id: str, ownership: CodeCraftSessionOwnership) -> CodeCraftSession | None:
        with self._lock:
            session = self._sessions.get(craft_id)
        if session is None or session.disposed:
            return None
        if not matches_session_ownership(
            session.tenant_id,
            session.task_id,
            session.run_id,
            ownership,
        ):
            raise CodeCraftOwnershipError("craft_session_ownership_mismatch")
        return session

    def save_owned(self, session: CodeCraftSession, ownership: CodeCraftSessionOwnership) -> None:
        current = self.get_owned(session.craft_id, ownership)
        if current is None:
            raise CodeCraftOwnershipError("craft_session_not_found")
        with self._lock:
            self._sessions[session.craft_id] = session

    def dispose_owned(
        self,
        craft_id: str,
        ownership: CodeCraftSessionOwnership,
    ) -> CodeCraftSession | None:
        session = self.get_owned(craft_id, ownership)
        if session is None:
            return None
        with self._lock:
            self._sessions.pop(craft_id, None)
        disposed = session.model_copy(update={"disposed": True, "status": "disposed"})
        return disposed

    def list_for_task(self, task_id: str, *, tenant_id: str) -> list[CodeCraftSession]:
        with self._lock:
            return [
                item
                for item in self._sessions.values()
                if item.task_id == task_id
                and item.tenant_id == tenant_id
                and not item.disposed
            ]


_default_manager: CodeCraftSessionManager | None = None


def get_session_manager(ctx: ToolWiringContext) -> CodeCraftSessionManager:
    raw = ctx.extras.get("codecraft_session_manager")
    if isinstance(raw, CodeCraftSessionManager):
        return raw
    global _default_manager
    if _default_manager is None:
        _default_manager = CodeCraftSessionManager()
    return _default_manager
