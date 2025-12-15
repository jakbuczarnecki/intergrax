# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Dict, List, Optional

from intergrax.llm.messages import ChatMessage
from intergrax.memory.conversational_memory import ConversationalMemory
from intergrax.memory.conversational_store import ConversationalMemoryStore


class InMemoryConversationalMemoryStore(ConversationalMemoryStore):
    """
    In-memory implementation of ConversationalMemoryStore.

    Primary use-cases:
    - local development,
    - prototyping,
    - unit and integration testing,
    - environments where persistence is NOT required.

    This implementation is NOT suitable for:
    - distributed runtime deployments,
    - multi-instance scaling,
    - long-lived production storage.

    Data remains isolated per Python interpreter and will NOT survive restart.
    """

    def __init__(self) -> None:
        # Maps session_id -> ordered list of ChatMessage
        self._sessions: Dict[str, List[ChatMessage]] = {}

    async def load_memory(
        self,
        session_id: str,
        *,
        max_messages: Optional[int] = None,
    ) -> ConversationalMemory:
        """
        Load conversation history into an IntergraxConversationalMemory instance.
        """
        messages = self._sessions.get(session_id, [])

        memory = ConversationalMemory(
            session_id=session_id,
            max_messages=max_messages,
        )

        if messages:
            memory.extend(messages)

        return memory

    async def save_memory(
        self,
        memory: ConversationalMemory,
    ) -> None:
        """
        Persist the full conversation history using defensive copying.
        """
        self._sessions[memory.session_id] = list(memory.get_all())

    async def append_message(
        self,
        memory: ConversationalMemory,
        message: ChatMessage,
    ) -> None:
        """
        Append message in memory and persist the updated state.
        """
        # First apply runtime logic (includes trimming & locking)
        memory.add(message.role, message.content)

        # Then persist the new state
        if memory.session_id not in self._sessions:
            self._sessions[memory.session_id] = []

        self._sessions[memory.session_id].append(message)

    async def delete_session(
        self,
        session_id: str,
    ) -> None:
        """
        Remove persistent session data (no-error semantics).
        """
        self._sessions.pop(session_id, None)

    # Optional helper for diagnostics and testing
    def list_sessions(self) -> List[str]:
        """Return list of active persisted session IDs."""
        return list(self._sessions.keys())
