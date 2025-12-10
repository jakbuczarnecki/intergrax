# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Protocol, Optional, List
from intergrax.llm.messages import ChatMessage
from intergrax.runtime.drop_in_knowledge_mode.session.chat_session import ChatSession


class SessionStorage(Protocol):
    """
    Low-level storage interface for chat sessions and their conversation
    history.

    Responsibilities:
      - Persist and load ChatSession objects.
      - Persist and load conversation history (ChatMessage sequences)
        for a given session.

    This interface is intentionally minimal and does not contain any
    higher-level domain logic (no profile instructions, no memory
    synthesis, no counters). Those responsibilities belong to the
    SessionManager layer built on top of this storage.
    """

    # ------------------------------------------------------------------
    # Session metadata CRUD
    # ------------------------------------------------------------------

    async def get_session(self, session_id: str) -> Optional[ChatSession]:
        """
        Load a session by its identifier.

        Returns:
            ChatSession if found, otherwise None.
        """
        ...

    async def create_session(
        self,
        session_id: Optional[str] = None,
        *,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> ChatSession:
        """
        Create and persist a new ChatSession.

        If session_id is None, the storage is allowed to generate its own
        identifier (e.g. UUID4).
        """
        ...
        

    async def save_session(self, session: ChatSession) -> None:
        """
        Persist changes to an existing ChatSession.

        This should be called by the SessionManager whenever session
        metadata (including timestamps) is updated.
        """
        ...

    async def list_sessions_for_user(
        self,
        user_id: str,
        *,
        limit: Optional[int] = None,
    ) -> List[ChatSession]:
        """
        List recent sessions for a given user, ordered by recency
        (e.g. updated_at descending).

        Parameters:
            user_id:
                Identifier of the user.
            limit:
                Maximum number of sessions to return.

        Returns:
            A list of ChatSession objects (may be empty).
        """
        ...

    # ------------------------------------------------------------------
    # Conversation history operations
    # ------------------------------------------------------------------

    async def append_message(
        self,
        session_id: str,
        message: ChatMessage,
    ) -> ChatMessage:
        """
        Append a single message to the conversation history of a session.

        Implementations may apply trimming or other retention policies,
        but the general contract is:
          - message is persisted,
          - the stored message (possibly enriched with additional data)
            is returned.
        """
        ...

    async def get_history(
        self,
        session_id: str,
        *,
        native_tools: bool = False,
    ) -> List[ChatMessage]:
        """
        Return the ordered conversation history for a given session id.

        Parameters:
            session_id:
                Identifier of the session.
            native_tools:
                If True, return messages with native tool-calls preserved
                (depending on the underlying history implementation).

        Returns:
            A list of ChatMessage objects in chronological order
            (may be empty if no history exists for the session).
        """
        ...
