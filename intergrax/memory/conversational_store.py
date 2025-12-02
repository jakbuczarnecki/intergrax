# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Protocol, Optional

from intergrax.llm.messages import ChatMessage
from intergrax.memory.conversational_memory import ConversationalMemory


class ConversationalMemoryStore(Protocol):
    """
    Abstract persistent storage interface for conversational memory.

    This interface operates directly on the conversational memory aggregate
    (`IntergraxConversationalMemory`), instead of raw collections of messages.

    Motivation:
    - The runtime always interacts with `IntergraxConversationalMemory`.
    - The application may swap persistence backends (JSON, SQLite, Redis, Cloud DB)
      without changing runtime logic.
    - The storage backend should NOT implement business logic, trimming policies,
      token heuristics, or model-format conversions — that belongs to the aggregate.

    Implementations MUST guarantee:
    - deterministic persistence,
    - idempotent write operations,
    - no modification of semantic meaning of messages,
    - safe interaction in async environments.
    """

    async def load_memory(
        self,
        session_id: str,
        *,
        max_messages: Optional[int] = None,
    ) -> ConversationalMemory:
        """
        Load the full conversational history for a given session
        and return a fully initialized `IntergraxConversationalMemory` instance.

        Parameters:
            session_id: Unique identifier representing a logical conversation.
            max_messages: Optional soft limit for the number of messages to preload.

        NOTE:
            Implementations may choose to:
            - preload full history (recommended),
            - respect the limit pre-load,
            - or ignore the parameter entirely if unsupported.

        Returns:
            A ready-to-use conversational memory object.
        """
        ...

    async def save_memory(
        self,
        memory: ConversationalMemory,
    ) -> None:
        """
        Persist the entire state of the conversational memory for a session.

        Typical usage:
        - after summarization / compression steps,
        - when closing or archiving a session,
        - when switching storage backends or running maintenance.

        MUST overwrite existing storage for the given session_id.
        """
        ...

    async def append_message(
        self,
        memory: ConversationalMemory,
        message: ChatMessage,
    ) -> None:
        """
        Append a single message to persistent storage AND update the in-memory
        instance accordingly.

        Implementations that do not support incremental persistence may fallback to:
        - load + append + save_memory(memory)

        This method SHOULD respect trimming rules applied by the aggregate.
        """
        ...

    async def delete_session(
        self,
        session_id: str,
    ) -> None:
        """
        Permanently remove stored history for a given session.

        Required for:
        - GDPR compliance,
        - explicit "clear conversation"
        - session reset flows,
        - user privacy controls.

        MUST silently ignore unknown session IDs (no-error semantics).
        """
        ...
