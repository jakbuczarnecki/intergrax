# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from datetime import timezone
from enum import Enum
from typing import Optional, Sequence, TYPE_CHECKING

from intergrax.llm.messages import ChatMessage
from intergrax.runtime.nexus.session.chat_session import ChatSession
from intergrax.runtime.nexus.tracing.session.session_consolidation_diag import SessionConsolidationDiagV1
from intergrax.runtime.nexus.tracing.trace_models import TraceComponent, TraceLevel
from intergrax.runtime.nexus.session.session_profile_instructions import build_consolidation_diag
from intergrax.runtime.user_profile.session_memory_consolidation_service import (
    SessionMemoryConsolidationService,
)
from intergrax.utils.time_provider import SystemTimeProvider

if TYPE_CHECKING:
    from intergrax.runtime.nexus.engine.runtime_state import RuntimeState


class SessionConsolidationReason(str, Enum):
    """Triggers for session memory consolidation (JSON-serializable values)."""

    MID_SESSION = "mid_session"
    CLOSE_SESSION = "close_session"


class SessionMemoryConsolidationCoordinator:
    """
    Coordinates mid-session and close-session memory consolidation.

    Keeps consolidation policy (interval, cooldown) and persistence side-effects
    out of SessionManager orchestration code.
    """

    def __init__(
        self,
        *,
        service: Optional[SessionMemoryConsolidationService],
        user_turns_interval: int,
        cooldown_seconds: int,
        consolidation_mode: str = "manual",
    ) -> None:
        self._service = service
        self._user_turns_interval = user_turns_interval
        self._cooldown_seconds = cooldown_seconds
        self._consolidation_mode = consolidation_mode

    @property
    def enabled(self) -> bool:
        return self._service is not None

    def should_consolidate_on_close(self, session: ChatSession) -> bool:
        if self._consolidation_mode == "manual":
            return False
        return self._service is not None and bool(session.user_id)

    def should_consolidate_mid_session(
        self,
        session: ChatSession,
        *,
        user_turns: int,
    ) -> bool:
        if self._consolidation_mode not in {"scheduled", "auto"}:
            return False
        if self._service is None or not session.user_id:
            return False
        interval = self._user_turns_interval
        if interval <= 0:
            return False
        if (user_turns % interval) != 0:
            return False
        return self._is_mid_session_allowed(session)

    async def consolidate(
        self,
        *,
        user_id: str,
        session_id: str,
        messages: Sequence[ChatMessage],
        run_id: Optional[str] = None,
    ) -> SessionConsolidationDiagV1:
        if self._service is None:
            return build_consolidation_diag(())
        stored = await self._service.consolidate_session(
            user_id=user_id,
            session_id=session_id,
            messages=list(messages),
            run_id=run_id,
        )
        return build_consolidation_diag(stored)

    def apply_consolidation_metadata(
        self,
        session: ChatSession,
        *,
        reason: SessionConsolidationReason,
        turn: Optional[int] = None,
    ) -> None:
        session.last_consolidated_at = SystemTimeProvider.utc_now()
        session.last_consolidated_reason = reason.value
        session.needs_user_instructions_refresh = True
        if turn is not None:
            session.last_consolidated_turn = int(turn)

    def trace_close_consolidation(
        self,
        trace_state: RuntimeState,
        diag: SessionConsolidationDiagV1,
    ) -> None:
        trace_state.trace_event(
            component=TraceComponent.ENGINE,
            step="SessionManager.close_session",
            message="Session consolidated",
            level=TraceLevel.DEBUG,
            payload=diag,
        )

    def _is_mid_session_allowed(self, session: ChatSession) -> bool:
        cooldown = self._cooldown_seconds
        if cooldown <= 0:
            return True
        last_dt = session.last_consolidated_at
        if last_dt is None:
            return True
        if last_dt.tzinfo is None:
            last_dt = last_dt.replace(tzinfo=timezone.utc)
        elapsed = (SystemTimeProvider.utc_now() - last_dt).total_seconds()
        return elapsed >= cooldown
