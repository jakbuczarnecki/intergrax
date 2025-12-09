# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Any, Dict, List, Optional

from intergrax.llm.messages import ChatMessage
from intergrax.memory.user_profile_manager import UserProfileManager
from intergrax.runtime.drop_in_knowledge_mode.session.chat_session import ChatSession
from intergrax.runtime.drop_in_knowledge_mode.session.session_storage import SessionStorage
from intergrax.runtime.organization.organization_profile_manager import (
    OrganizationProfileManager,
)


class SessionManager:
    """
    High-level manager for chat sessions.

    Responsibilities:
      - Orchestrate session lifecycle on top of a SessionStorage backend.
      - Provide a stable API for the runtime engine (DropInKnowledgeRuntime).
      - Integrate with user/organization profile managers to expose
        prompt-ready system instructions per session.

    This class should be the *only* component that the runtime engine
    talks to when it comes to sessions and their metadata/history.
    """

    def __init__(
        self,
        storage: SessionStorage,
        *,
        user_profile_manager: Optional[UserProfileManager] = None,
        organization_profile_manager: Optional[OrganizationProfileManager] = None,
    ) -> None:
        # Low-level storage backend (in-memory, DB, Redis, etc.).
        self._storage = storage

        # High-level managers for profile-based instructions (optional).
        self._user_profile_manager = user_profile_manager
        self._organization_profile_manager = organization_profile_manager

    # ------------------------------------------------------------------
    # Session lifecycle (metadata)
    # ------------------------------------------------------------------

    async def get_session(self, session_id: str) -> Optional[ChatSession]:
        """
        Return ChatSession metadata if it exists, else None.
        """
        return await self._storage.get_session(session_id)

    async def create_session(
        self,
        session_id: Optional[str] = None,
        *,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ChatSession:
        """
        Create and persist a new ChatSession via the underlying storage.

        Notes:
          - If session_id is None, the storage may generate a new identifier.
          - This method only encapsulates construction + basic defaults;
            all persistence is delegated to SessionStorage.
        """
        return await self._storage.create_session(
            session_id=session_id,
            user_id=user_id,
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            metadata=metadata,
        )

    async def save_session(self, session: ChatSession) -> None:
        """
        Persist changes to an existing ChatSession.

        The manager refreshes the modification timestamp and delegates
        the actual persistence to the storage backend.
        """
        session.touch()
        await self._storage.save_session(session)

    
    async def close_session(
        self,
        session_id: str,
        *,
        reason: Optional[str] = None,
    ) -> None:
        """
        Mark a session as closed at the domain level.

        This does NOT delete messages or remove the session. It only updates
        metadata so that higher-level components (APIs, UI, runtime) can
        treat the session as no longer active.
        """
        session = await self._storage.get_session(session_id)
        if session is None:
            return

        # Use domain logic encapsulated in ChatSession.
        session.mark_closed(reason=reason)
        await self._storage.save_session(session)


    async def list_sessions_for_user(
        self,
        user_id: str,
        *,
        limit: int = 50,
    ) -> List[ChatSession]:
        """
        List recent sessions for a given user, ordered by recency.
        """
        return await self._storage.list_sessions_for_user(user_id, limit=limit)

    # ------------------------------------------------------------------
    # Conversation history
    # ------------------------------------------------------------------

    async def append_message(
        self,
        session_id: str,
        message: ChatMessage,
    ) -> ChatMessage:
        """
        Append a single message to the conversation history of a session.

        Trimming / retention policies are implemented by the storage
        (e.g. via ConversationalMemory in an in-memory backend).
        """
        return await self._storage.append_message(session_id, message)

    async def get_history_for_session(
        self,
        session_id: str,
        *,
        native_tools: bool = False,
    ) -> List[ChatMessage]:
        """
        Convenience helper: return conversation history by session id.

        This is the primary method to use from higher-level components
        (e.g. memory consolidation services) when they need the full
        conversation for a given session.
        """
        return await self._storage.get_history(
            session_id=session_id,
            native_tools=native_tools,
        )

    # ------------------------------------------------------------------
    # User profile memory – prompt-level instructions (per session)
    # ------------------------------------------------------------------

    async def get_user_profile_instructions_for_session(
        self,
        session: ChatSession,
    ) -> Optional[str]:
        """
        Return a prompt-ready user profile instruction string for this session.

        Behavior:
          - If a cached value is present in session.metadata["user_profile_instructions"],
            it is returned (after stripping whitespace).
          - Otherwise this method delegates to UserProfileManager, calling
            `get_system_instructions_for_user(user_id)` which returns the
            effective user-level system instructions (already including any
            internal fallbacks), caches the resulting string in metadata,
            and saves the updated session.

        Semantics:
          - Instructions are effectively *snapshotted per session*.
            If the underlying user profile changes, existing sessions
            keep their cached version until explicitly refreshed or
            the session is recreated.
        """
        # No associated user or no profile manager → no instructions.
        if not session.user_id:
            return None
        if self._user_profile_manager is None:
            return None

        # 1) Try cached instructions from session metadata.
        cached = session.metadata.get("user_profile_instructions")
        if isinstance(cached, str):
            stripped = cached.strip()
            if stripped:
                return stripped

        # 2) Fallback: resolve from the user profile manager.
        # The manager encapsulates all logic of:
        #   - using profile.system_instructions if set,
        #   - or falling back to a deterministic summary if not.
        instructions = await self._user_profile_manager.get_system_instructions_for_user(
            session.user_id
        )
        if not isinstance(instructions, str):
            return None

        stripped = instructions.strip()
        if not stripped:
            return None

        # 3) Cache in session metadata and persist the session.
        session.metadata["user_profile_instructions"] = stripped
        await self.save_session(session)

        return stripped

    # ------------------------------------------------------------------
    # Organization profile memory – prompt-level instructions (per session)
    # ------------------------------------------------------------------

    async def get_org_profile_instructions_for_session(
        self,
        session: ChatSession,
    ) -> Optional[str]:
        """
        Return a prompt-ready organization profile instruction string
        for this session.

        Behavior:
          - If a cached value is present in session.metadata["org_profile_instructions"],
            it is returned (after stripping whitespace).
          - Otherwise this method delegates to OrganizationProfileManager, calling
            `get_system_instructions_for_organization(organization_id, ...)`,
            caches the resulting string in metadata, and saves the updated session.

        Note:
          - This method no longer uses prompt bundles; it works purely on
            the final system-instructions string exposed by the manager.
          - The organization identifier is derived from session.tenant_id.
        """
        # No associated tenant or no organization profile manager → no instructions.
        if not session.tenant_id:
            return None
        if self._organization_profile_manager is None:
            return None

        # 1) Try cached instructions from session metadata.
        cached = session.metadata.get("org_profile_instructions")
        if isinstance(cached, str):
            stripped = cached.strip()
            if stripped:
                return stripped

        # 2) Fallback: resolve from the organization profile manager.
        instructions = (
            await self._organization_profile_manager.get_system_instructions_for_organization(
                organization_id=session.tenant_id
            )
        )
        if not isinstance(instructions, str):
            return None

        stripped = instructions.strip()
        if not stripped:
            return None

        # 3) Cache in session metadata and persist the session.
        session.metadata["org_profile_instructions"] = stripped
        await self.save_session(session)

        return stripped
