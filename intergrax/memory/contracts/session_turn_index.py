# © Artur Czarnecki. All rights reserved.

"""Session turn vector index contracts (Phase MEM-VEC-2.1)."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from intergrax.llm.messages import ChatMessage


@runtime_checkable
class SessionTurnIndexStore(Protocol):
    """Episodic vector index over session turns — index over ``SessionStorage``, not a replacement."""

    async def upsert_turn(
        self,
        *,
        tenant_id: str,
        session_id: str,
        user_id: str | None,
        message: ChatMessage,
    ) -> None: ...

    async def tombstone_turn(self, entry_id: str) -> None: ...

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
    ) -> list[dict[str, Any]]: ...


@runtime_checkable
class SessionTurnIndexStorePlugin(Protocol):
    """Entry-point plugin for custom episodic index backends (MEM-VEC-3.1)."""

    @classmethod
    def plugin_id(cls) -> str: ...

    @classmethod
    def create_session_turn_index(cls, **kwargs: Any) -> SessionTurnIndexStore: ...
