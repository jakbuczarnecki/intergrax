from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

from intergrax.globals.settings import GLOBAL_SETTINGS
from intergrax.llm.messages import ChatMessage
from intergrax.memory.user_profile_manager import UserProfileManager
from intergrax.runtime.nexus.session.session_message_append_result import SessionMessageAppendResult
from intergrax.runtime.nexus.tracing.session.session_consolidation_diag import SessionConsolidationDiagV1
if TYPE_CHECKING:
    from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.session.chat_session import (
    ChatSession,
    SessionCloseReason,
)
from intergrax.runtime.nexus.session.session_storage import (
    SessionStorage,
)
from intergrax.runtime.organization.organization_profile_manager import (
    OrganizationProfileManager,
)
from intergrax.runtime.nexus.session.session_consolidation import (
    SessionConsolidationReason,
    SessionMemoryConsolidationCoordinator,
)
from intergrax.runtime.nexus.session.session_lifecycle import SessionLifecycleCoordinator
from intergrax.runtime.nexus.session.session_profile_instructions import (
    SessionProfileInstructionResolver,
)
from intergrax.runtime.user_profile.session_memory_consolidation_service import (
    SessionMemoryConsolidationService,
)


class SessionManager:
    """
    High-level manager for chat sessions.

    Responsibilities:
      - Orchestrate session lifecycle on top of a SessionStorage backend.
      - Provide a stable API for agent runtime sessions (AgentEngine / UAEP).
      - Integrate with user/organization profile managers to expose
        prompt-ready system instructions per session.
      - Optionally trigger long-term user memory consolidation for a session.

    This class should be the *only* component that the runtime engine
    talks to when it comes to sessions and their metadata/history.
    """

    def __init__(
        self,
        storage: SessionStorage,
        *,
        user_profile_manager: Optional[UserProfileManager] = None,
        organization_profile_manager: Optional[OrganizationProfileManager] = None,
        session_memory_consolidation_service: Optional[SessionMemoryConsolidationService] = None,
        user_turns_consolidation_interval: Optional[int] = GLOBAL_SETTINGS.default_user_turns_consolidation_interval,
        consolidation_cooldown_seconds: Optional[int] = GLOBAL_SETTINGS.default_consolidation_cooldown_seconds,
    ) -> None:
        """
        Initialize a new SessionManager instance.

        Args:
            storage:
                Low-level session + history storage backend (in-memory, DB, etc.).
            user_profile_manager:
                Optional manager used to resolve user-level system instructions
                and to write long-term user profile memory.
            organization_profile_manager:
                Optional manager used to resolve organization-level
                system instructions (per tenant / org).
            session_memory_consolidation_service:
                Optional service responsible for consolidating a single session
                into long-term user profile memory entries and refreshing
                user-level system instructions.
            user_turns_consolidation_interval:
                Interval (in user turns) for mid-session consolidation.
                If None or non-positive, mid-session consolidation is disabled.
            consolidation_cooldown_seconds:
                Cooldown (in seconds) between mid-session consolidations for a
                single session. If None or non-positive, no cooldown is applied.
        """
        # Low-level storage backend (in-memory, DB, Redis, etc.).
        self._storage = storage

        # High-level managers for profile-based instructions (optional).
        self._user_profile_manager = user_profile_manager
        self._organization_profile_manager = organization_profile_manager

        # Optional service that can consolidate a single session into
        # long-term user profile memory entries and refresh user-level
        # system instructions.
        self._session_memory_consolidation_service = (
            session_memory_consolidation_service
        )

        # Resolve the effective interval for mid-session consolidation.
        # The value is interpreted as:
        #   - > 0  → consolidate every N-th user message,
        #   - <= 0 → mid-session consolidation disabled.
        if (
            user_turns_consolidation_interval is not None
            and user_turns_consolidation_interval > 0
        ):
            effective_interval = user_turns_consolidation_interval
        else:
            effective_interval = 0

        self._user_turns_consolidation_interval: int = effective_interval

        # Effective cooldown in seconds between mid-session consolidations.
        # The value is interpreted as:
        #   - > 0  → enforce cooldown,
        #   - <= 0 → no cooldown (only the interval is applied).
        if (
            consolidation_cooldown_seconds is not None
            and consolidation_cooldown_seconds > 0
        ):
            effective_cooldown = consolidation_cooldown_seconds
        else:
            effective_cooldown = 0

        self._consolidation_cooldown_seconds: int = effective_cooldown
        self._consolidation = SessionMemoryConsolidationCoordinator(
            service=session_memory_consolidation_service,
            user_turns_interval=effective_interval,
            cooldown_seconds=effective_cooldown,
        )
        self._lifecycle = SessionLifecycleCoordinator(storage)
        self._profile_instructions = SessionProfileInstructionResolver(
            user_profile_manager=user_profile_manager,
            organization_profile_manager=organization_profile_manager,
        )


    async def get_history(
        self,
        *,
        tenant_id: str,
        session_id: str,
    ) -> List[ChatMessage]:
        return await self._storage.get_history(
            tenant_id=tenant_id,
            session_id=session_id,
        )

    # ------------------------------------------------------------------
    # Session lifecycle (metadata)
    # ------------------------------------------------------------------

    async def get_session(
        self,
        *,
        tenant_id: str,
        session_id: str,
    ) -> Optional[ChatSession]:
        """
        Return ChatSession metadata if it exists for a given tenant.
        """
        return await self._lifecycle.get_session(
            tenant_id=tenant_id,
            session_id=session_id,
        )

    async def create_session(
        self,
        *,
        tenant_id: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
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
        return await self._lifecycle.create_session(
            tenant_id=tenant_id,
            session_id=session_id,
            user_id=user_id,
            workspace_id=workspace_id,
            metadata=metadata,
        )

    async def get_or_create_session(
        self,
        *,
        user_id: str,
        session_id: str,
        tenant_id: str,
        workspace_id: Optional[str] = None,
    ) -> ChatSession:
        return await self._lifecycle.get_or_create_session(
            user_id=user_id,
            session_id=session_id,
            tenant_id=tenant_id,
            workspace_id=workspace_id,
        )

    async def save_session(self, session: ChatSession) -> None:
        """Persist changes to an existing ChatSession."""
        await self._lifecycle.save_session(session)

    async def close_session(
        self,
        session_id: str,
        *,
        reason: Optional[SessionCloseReason] = None,
        run_id: Optional[str] = None,
        trace_state: Optional[RuntimeState] = None,
    ) -> None:
        """
        Mark a session as closed at the domain level and, if configured,
        trigger long-term memory consolidation for this session.

        Behavior:
          - Mark the ChatSession as closed and persist it.
          - If a SessionMemoryConsolidationService is available and the
            session has an associated user_id:
              * load the conversation history for this session,
              * call consolidate_session(user_id, session_id, messages)
                to extract long-term memory entries and update the
                user's system_instructions (side-effect).

        Args:
            session_id:
                Identifier of the session to close.
            reason:
                Optional domain-level reason. If None, a default
                SessionCloseReason.EXPLICIT is used.
        """
        session = await self._storage.get_session(session_id)
        if session is None:
            return

        # Decide which close reason to apply. If caller did not provide one,
        # we use EXPLICIT as the default semantic.
        effective_reason = reason or SessionCloseReason.EXPLICIT

        # 1) Domain-level close (no deletion of messages).
        #    ChatSession is responsible for updating its own status and
        #    closed_reason according to the enum value.
        session.mark_closed(reason=effective_reason)
        await self._storage.save_session(session)

        # 2) Optional: consolidate this session into long-term user memory.
        #    We only do this if:
        #      - the service is configured, and
        #      - the session is associated with a user_id.
        if self._consolidation.should_consolidate_on_close(session):
            messages = await self.get_history_for_session(session_id)
            if messages:
                diag = await self._consolidation.consolidate(
                    user_id=session.user_id,
                    session_id=session_id,
                    messages=messages,
                    run_id=run_id,
                )
                if trace_state is not None:
                    self._consolidation.trace_close_consolidation(trace_state, diag)
                self._consolidation.apply_consolidation_metadata(
                    session,
                    reason=SessionConsolidationReason.CLOSE_SESSION,
                    turn=session.user_turns,
                )
                await self.save_session(session)

    async def list_sessions_for_user(
        self,
        user_id: str,
        *,
        limit: Optional[int] = None,
    ) -> List[ChatSession]:
        """
        List recent sessions for a given user, ordered by recency.
        """
        return await self._lifecycle.list_sessions_for_user(user_id, limit=limit)

    # ------------------------------------------------------------------
    # Conversation history
    # ------------------------------------------------------------------

    async def append_message(
        self,
        *,
        tenant_id: str,
        session_id: str,
        message: ChatMessage,
    ) -> SessionMessageAppendResult:
        """
        Append a single message to the conversation history of a session.

        Domain rules:
          - For user messages, increment the per-session "user_turns" counter
            stored in ChatSession.user_turns.
          - Optionally, every N-th user message, trigger mid-session
            long-term memory consolidation for this session.
          - Trimming / retention policies for history are implemented by the
            underlying storage (e.g. ConversationalMemory).

        Note:
          - This method intentionally keeps domain logic (user_turns and
            consolidation hooks) at the manager level, while the storage
            remains responsible only for persisting sessions and their history.
        """

        consolidation_diag: Optional[SessionConsolidationDiagV1] = None

        # Try to load the session so we can apply domain-level updates
        # (user_turns counter, timestamps, etc.).
        session = await self._lifecycle.get_session(
            tenant_id=tenant_id,
            session_id=session_id,
        )

        # Increment user_turns only for user messages and only if the
        # session exists. If the session is missing, we delegate error
        # handling to the storage.append_message call below.
        if session is not None and message.role == "user":
            # This updates in-memory state and timestamps; persistence is
            # delegated to save_session().
            user_turns = session.increment_user_turns()
            await self.save_session(session)

            if self._consolidation.should_consolidate_mid_session(
                session,
                user_turns=user_turns,
            ):
                messages = await self.get_history_for_session(session_id)
                if messages:
                    consolidation_diag = await self._consolidation.consolidate(
                        user_id=session.user_id,
                        session_id=session_id,
                        messages=messages,
                    )
                    self._consolidation.apply_consolidation_metadata(
                        session,
                        reason=SessionConsolidationReason.MID_SESSION,
                        turn=user_turns,
                    )
                    await self.save_session(session)

        # Delegate message persistence to the storage backend. The storage
        # may apply its own retention/trimming logic (FIFO, max_messages, etc.).
        
        stored_message = await self._storage.append_message(
            tenant_id=tenant_id,
            session_id=session_id,
            message=message,
        )

        return SessionMessageAppendResult(
            message=stored_message,
            consolidation_diag=consolidation_diag,
        )

    async def get_history_for_session(
        self,
        *,
        tenant_id: str,
        session_id: str,
        native_tools: bool = False,
    ) -> List[ChatMessage]:
        """
        Return conversation history scoped to a tenant.
        """
        return await self._storage.get_history(
            tenant_id=tenant_id,
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
        """Return prompt-ready user profile instructions (cached per session)."""
        instructions = await self._profile_instructions.user_instructions_for_session(session)
        if instructions is None:
            return None
        await self.save_session(session)
        return instructions


    async def search_user_longterm_memory(
        self,
        user_id: str,
        query: str,
        *,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Delegate long-term memory retrieval to UserProfileManager (if available).
        Engine should not know how the profile manager implements retrieval.
        """
        if self._user_profile_manager is None:
            return None

        return await self._user_profile_manager.search_longterm_memory(
            user_id=user_id,
            query=query,
            top_k=top_k,
            score_threshold=score_threshold,
        )


    # ------------------------------------------------------------------
    # Organization profile memory – prompt-level instructions (per session)
    # ------------------------------------------------------------------

    async def get_org_profile_instructions_for_session(
        self,
        session: ChatSession,
    ) -> Optional[str]:
        """Return prompt-ready organization profile instructions (cached per session)."""
        instructions = await self._profile_instructions.org_instructions_for_session(session)
        if instructions is None:
            return None
        await self.save_session(session)
        return instructions
