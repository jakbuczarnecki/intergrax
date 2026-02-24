# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

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
        self._sessions: Dict[Tuple[str, str], List[ChatMessage]] = {}

    async def load_memory(
        self,
        *,
        tenant_id: str,
        session_id: str,
        max_messages: Optional[int] = None,
    ) -> ConversationalMemory:
        """
        Load conversation history into an IntergraxConversationalMemory instance.
        """
        key = (tenant_id, session_id)
        messages = self._sessions.get(key, [])

        memory = ConversationalMemory(
            session_id=session_id,
            max_messages=max_messages,
        )

        if messages:
            memory.extend(messages)

        return memory

    async def save_memory(
        self,
        *,
        tenant_id: str,
        memory: ConversationalMemory,
    ) -> None:
        """
        Persist the full conversation history using defensive copying.
        """
        key = (tenant_id, memory.session_id)
        self._sessions[key] = list(memory.get_all())

    async def append_message(
        self,
        *,
        tenant_id: str,
        memory: ConversationalMemory,
        message: ChatMessage,
    ) -> None:
        # First apply runtime logic (includes trimming & locking)
        memory.add(message.role, message.content)

        key = (tenant_id, memory.session_id)

        # Then persist the new state
        if key not in self._sessions:
            self._sessions[key] = []

        self._sessions[key].append(message)

    async def delete_session(
        self,
        *,
        tenant_id: str,
        session_id: str,
    ) -> None:
        key = (tenant_id, session_id)
        self._sessions.pop(key, None)

    # Optional helper for diagnostics and testing
    def list_sessions(self, *, tenant_id: str) -> List[str]:
        return [
            session_id
            for (t_id, session_id) in self._sessions.keys()
            if t_id == tenant_id
        ]
