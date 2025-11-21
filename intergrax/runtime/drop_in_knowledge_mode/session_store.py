# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""
Session storage abstractions for the Drop-In Knowledge Mode runtime.

This module defines:
  - Data classes representing chat sessions and messages.
  - A SessionStore protocol that can be implemented for different backends
    (in-memory, SQLite, PostgreSQL, Supabase, etc.).
  - A simple in-memory implementation for quick experiments and notebooks.

The goal is:
  * Each runtime instance always works inside a session.
  * The session can be loaded from or persisted to any storage backend.
  * Messages can carry attachment references and arbitrary metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import (
    Any, 
    Dict, 
    List, 
    Mapping, 
    Optional, 
    Protocol, 
    Literal, 
    runtime_checkable)
import uuid
from intergrax.llm.conversational_memory import (
    AttachmentRef,
    MessageRole
)


@dataclass
class SessionMessage:
    """
    A single message in a chat session.

    This is intentionally storage-agnostic and independent from any specific
    LLM schema. The runtime will later map it to the concrete adapter format.
    """

    id: str
    role: MessageRole
    content: str
    created_at: datetime
    attachments: List[AttachmentRef] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChatSession:
    """
    Top-level representation of a chat session.

    A session groups messages, attachments and metadata under a stable ID.
    It also carries tenant / workspace scope for multi-tenant environments.
    """

    id: str
    user_id: Optional[str] = None
    tenant_id: Optional[str] = None
    workspace_id: Optional[str] = None

    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    updated_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    messages: List[SessionMessage] = field(default_factory=list)
    # Optional global session-level attachments (not tied to a single message)
    attachments: List[AttachmentRef] = field(default_factory=list)

    metadata: Dict[str, Any] = field(default_factory=dict)

    def touch(self) -> None:
        """Update the `updated_at` timestamp to 'now'."""
        self.updated_at = datetime.now(timezone.utc)


@runtime_checkable
class SessionStore(Protocol):
    """
    Abstract interface for session persistence.

    Implementations can use:
      - pure in-memory storage (for tests / notebooks),
      - SQLite / PostgreSQL,
      - document stores,
      - Supabase, etc.

    The Drop-In Knowledge Runtime will depend only on this protocol.
    """

    async def get_session(self, session_id: str) -> Optional[ChatSession]:
        """Load a session by its ID, or return None if it does not exist."""
        ...

    async def create_session(
        self,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> ChatSession:
        """
        Create a new session with a generated ID and return it.

        Implementations should persist the session immediately.
        """
        ...

    async def save_session(self, session: ChatSession) -> None:
        """
        Persist the entire session object.

        Implementations may choose to upsert only changed fields, but the
        contract is: after this call, the given session is durable.
        """
        ...

    async def append_message(
        self,
        session_id: str,
        message: SessionMessage,
    ) -> None:
        """
        Append a new message to an existing session and persist the change.

        Implementations may fail if the session does not exist.
        """
        ...

    async def list_sessions_for_user(
        self,
        user_id: str,
        limit: int = 50,
    ) -> List[ChatSession]:
        """
        Return recent sessions for a given user, ordered by updated_at desc.

        Implementations may choose to omit full messages for performance and
        only return lightweight session headers; in that case they should
        document the behavior.
        """
        ...


class InMemorySessionStore(SessionStore):
    """
    Trivial in-memory SessionStore implementation.

    This is useful for:
      - notebooks and examples,
      - quick prototypes,
      - unit tests.

    It is NOT suitable for production use, because data is kept in RAM only.
    """

    def __init__(self) -> None:
        self._sessions: Dict[str, ChatSession] = {}

    async def get_session(self, session_id: str) -> Optional[ChatSession]:
        return self._sessions.get(session_id)

    async def create_session(
        self,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> ChatSession:
        
        session_id = session_id or str(uuid.uuid4())

        session = ChatSession(
            id=session_id,
            user_id=user_id,
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            metadata=dict(metadata) if metadata is not None else {},
        )
        self._sessions[session_id] = session
        return session

    async def save_session(self, session: ChatSession) -> None:
        session.touch()
        self._sessions[session.id] = session

    async def append_message(
        self,
        session_id: str,
        message: SessionMessage,
    ) -> None:
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(f"Session '{session_id}' does not exist")

        session.messages.append(message)
        session.touch()
        self._sessions[session_id] = session

    async def list_sessions_for_user(
        self,
        user_id: str,
        limit: int = 50,
    ) -> List[ChatSession]:
        sessions = [
            s for s in self._sessions.values() if s.user_id == user_id
        ]
        sessions.sort(key=lambda s: s.updated_at, reverse=True)
        return sessions[:limit]
