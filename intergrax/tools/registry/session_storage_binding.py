# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Sync adapter from async ``SessionStorage`` to ``SessionStorageBinding``."""

from __future__ import annotations

import asyncio
import inspect
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

from intergrax.runtime.nexus.session.session_storage import SessionStorage
from intergrax.tools.registry.runtime_bindings import SessionStorageBinding


def _run_async(awaitable: object) -> object:
    if not asyncio.iscoroutine(awaitable):
        return awaitable
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(awaitable)
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, awaitable).result()


class SessionStorageToolBinding:
    """Adapts ``SessionStorage`` for sync interaction catalog tools."""

    def __init__(self, storage: SessionStorage) -> None:
        self._storage = storage

    def list_sessions(self, tenant_id: str, user_id: str, *, limit: int = 20) -> List[dict[str, str]]:
        sessions = _run_async(self._list_sessions_async(tenant_id, user_id, limit=limit))
        rows: list[dict[str, str]] = []
        for session in list(sessions or []):
            updated_at = getattr(session, "updated_at", None)
            updated_at_utc = updated_at.isoformat() if hasattr(updated_at, "isoformat") else str(updated_at or "")
            rows.append(
                {
                    "session_id": str(getattr(session, "id", "")),
                    "user_id": str(getattr(session, "user_id", None) or user_id),
                    "tenant_id": str(getattr(session, "tenant_id", None) or tenant_id),
                    "updated_at_utc": updated_at_utc,
                }
            )
        return [row for row in rows if row["session_id"]][:limit]

    async def _list_sessions_async(self, tenant_id: str, user_id: str, *, limit: int) -> list[object]:
        method = self._storage.list_sessions_for_user
        params = inspect.signature(method).parameters
        if "tenant_id" in params:
            result = await method(tenant_id=tenant_id, user_id=user_id, limit=limit)
        else:
            result = await method(user_id, limit=limit)
            result = [
                item
                for item in list(result or [])
                if str(getattr(item, "tenant_id", tenant_id) or tenant_id) == tenant_id
            ]
        return list(result or [])

    def get_last_user_input(self, tenant_id: str, session_id: str) -> Optional[str]:
        history = _run_async(
            self._storage.get_history(tenant_id=tenant_id, session_id=session_id),
        )
        for message in reversed(list(history or [])):
            if getattr(message, "role", "") == "user":
                content = str(getattr(message, "content", "") or "").strip()
                if content:
                    return content
        return None

    def get_session_history(
        self,
        tenant_id: str,
        session_id: str,
        *,
        limit: int = 50,
    ) -> List[dict[str, str]]:
        history = _run_async(
            self._storage.get_history(tenant_id=tenant_id, session_id=session_id),
        )
        rows: list[dict[str, str]] = []
        for message in list(history or [])[-limit:]:
            content = str(getattr(message, "content", "") or "").strip()
            if not content:
                continue
            rows.append(
                {
                    "role": str(getattr(message, "role", "") or ""),
                    "content": content,
                    "session_id": session_id.strip(),
                    "tenant_id": tenant_id.strip(),
                }
            )
        return rows


def session_storage_tool_binding(storage: SessionStorage | SessionStorageBinding | None) -> SessionStorageBinding | None:
    if storage is None:
        return None
    if isinstance(storage, SessionStorageBinding):
        return storage
    return SessionStorageToolBinding(storage)
