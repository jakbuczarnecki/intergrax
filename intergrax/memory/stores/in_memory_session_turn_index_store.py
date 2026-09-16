# © Artur Czarnecki. All rights reserved.

"""Reference in-process session turn index for qualification (MEM-ENT-13B)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.llm.messages import ChatMessage
from intergrax.memory.contracts.session_turn_index import (
    SessionTurnIndexHit,
    SessionTurnIndexStore,
)


@dataclass(slots=True)
class _IndexedTurn:
    tenant_id: str
    session_id: str
    user_id: str | None
    message: ChatMessage


class InMemorySessionTurnIndexStore(SessionTurnIndexStore):
    """Vendor-neutral in-memory ``SessionTurnIndexStore`` for behavioral qualification."""

    def __init__(self) -> None:
        self._rows: dict[str, _IndexedTurn] = {}
        self._tombstoned: set[str] = set()

    async def upsert_turn(
        self,
        *,
        tenant_id: str,
        session_id: str,
        user_id: str | None,
        message: ChatMessage,
    ) -> None:
        if message.deleted:
            await self.tombstone_turn(message.entry_id)
            return
        self._tombstoned.discard(message.entry_id)
        self._rows[message.entry_id] = _IndexedTurn(
            tenant_id=tenant_id,
            session_id=session_id,
            user_id=user_id,
            message=message,
        )

    async def tombstone_turn(self, entry_id: str) -> None:
        self._tombstoned.add(entry_id)
        self._rows.pop(entry_id, None)

    async def search_turns(
        self,
        *,
        query: str,
        tenant_id: str,
        session_id: str | None = None,
        user_id: str | None = None,
        top_k: int = 8,
        score_threshold: float | None = None,
        include_cross_session: bool = False,
    ) -> list[SessionTurnIndexHit]:
        needle = (query or "").strip().lower()
        matched: list[SessionTurnIndexHit] = []
        for entry_id, row in self._rows.items():
            if entry_id in self._tombstoned:
                continue
            if row.tenant_id != tenant_id:
                continue
            if session_id is not None and row.session_id != session_id:
                if not include_cross_session:
                    continue
            if user_id is not None and row.user_id != user_id:
                continue
            content = (row.message.content or "").strip()
            if needle and needle not in content.lower():
                continue
            matched.append(
                SessionTurnIndexHit(
                    entry_id=entry_id,
                    tenant_id=row.tenant_id,
                    session_id=row.session_id,
                    user_id=row.user_id,
                    message=row.message,
                    score=1.0,
                )
            )
        if score_threshold is not None:
            matched = [item for item in matched if item.score >= score_threshold]
        return matched[:top_k]
