# © Artur Czarnecki. All rights reserved.

"""Background memory consolidation scheduler (Phase MEM-DEPTH-3.1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

from intergrax.llm.messages import ChatMessage
from intergrax.runtime.nexus.session.session_manager import SessionManager
from intergrax.runtime.user_profile.session_memory_consolidation_service import (
    SessionMemoryConsolidationService,
)


@dataclass(frozen=True, slots=True)
class MemoryConsolidationJobResult:
    session_id: str
    user_id: str
    entries_stored: int
    skipped_reason: Optional[str] = None


class MemoryConsolidationJob:
    """
    Scheduler hook for automatic session → LTM consolidation.

    Wired when ``RuntimeConfig.memory_consolidation_mode`` is ``scheduled`` or ``auto``.
    """

    def __init__(
        self,
        *,
        session_manager: SessionManager,
        consolidation_service: SessionMemoryConsolidationService,
        consolidation_mode: str = "manual",
    ) -> None:
        self._session_manager = session_manager
        self._service = consolidation_service
        self._mode = consolidation_mode

    @property
    def enabled(self) -> bool:
        return self._mode in {"scheduled", "auto"}

    async def run_for_session(
        self,
        *,
        tenant_id: str,
        session_id: str,
        user_id: str,
        messages: Sequence[ChatMessage],
        run_id: Optional[str] = None,
    ) -> MemoryConsolidationJobResult:
        if not self.enabled:
            return MemoryConsolidationJobResult(
                session_id=session_id,
                user_id=user_id,
                entries_stored=0,
                skipped_reason=f"mode={self._mode}",
            )
        stored = await self._service.consolidate_session(
            user_id=user_id,
            session_id=session_id,
            messages=list(messages),
            run_id=run_id,
        )
        return MemoryConsolidationJobResult(
            session_id=session_id,
            user_id=user_id,
            entries_stored=len(stored),
        )

    async def run_close_session(
        self,
        *,
        tenant_id: str,
        session_id: str,
        user_id: str,
        run_id: Optional[str] = None,
    ) -> MemoryConsolidationJobResult:
        if not self.enabled:
            return MemoryConsolidationJobResult(
                session_id=session_id,
                user_id=user_id,
                entries_stored=0,
                skipped_reason=f"mode={self._mode}",
            )
        history = await self._session_manager.get_history(
            tenant_id=tenant_id,
            session_id=session_id,
        )
        return await self.run_for_session(
            tenant_id=tenant_id,
            session_id=session_id,
            user_id=user_id,
            messages=history,
            run_id=run_id,
        )
