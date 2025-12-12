from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Sequence
from datetime import datetime, timezone

from intergrax.globals.settings import GLOBAL_SETTINGS
from intergrax.llm.messages import ChatMessage
from intergrax.memory.user_profile_manager import UserProfileManager
from intergrax.runtime.drop_in_knowledge_mode.session.chat_session import (
    ChatSession,
    SessionCloseReason,
)
from intergrax.runtime.drop_in_knowledge_mode.session.session_storage import (
    SessionStorage,
)
from intergrax.runtime.organization.organization_profile_manager import (
    OrganizationProfileManager,
)
from intergrax.runtime.user_profile.session_memory_consolidation_service import (
    SessionMemoryConsolidationService,
)


class SessionConsolidationReason(str, Enum):
    """
    Enumeration of session consolidation triggers.
    Keeping this as `str` + `Enum` ensures that the value
    is safe to serialize into metadata and logs.
    """

    MID_SESSION = "mid_session"
    CLOSE_SESSION = "close_session"


class SessionManager:
    """
    High-level manager for chat sessions.

    Responsibilities:
      - Orchestrate session lifecycle on top of a SessionStorage backend.
      - Provide a stable API for the runtime engine (DropInKnowledgeRuntime).
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
        session_memory_consolidation_service: Optional[
            SessionMemoryConsolidationService
        ] = None,
        user_turns_consolidation_interval: Optional[
            int
        ] = GLOBAL_SETTINGS.default_user_turns_consolidation_interval,
        consolidation_cooldown_seconds: Optional[
            int
        ] = GLOBAL_SETTINGS.default_consolidation_cooldown_seconds,
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


    async def get_history(self, session_id: str) -> List[ChatMessage]:
        return self._storage.get_history(session_id=session_id)

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
        reason: Optional[SessionCloseReason] = None,
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
        if (
            self._session_memory_consolidation_service is not None
            and session.user_id
        ):
            # Fetch full conversation history for this session. This allows the
            # consolidation service to decide how much to trim and which parts
            # to keep, based on its own config (max messages, char budget, etc.).
            messages = await self.get_history_for_session(session_id)

            # If there's no history, there is nothing to consolidate.
            if messages:
                stored_entries = (
                    await self._session_memory_consolidation_service.consolidate_session(
                        user_id=session.user_id,
                        session_id=session_id,
                        messages=messages,
                    )
                )

                debug_info = self._build_consolidation_debug_info(stored_entries)

                # Mark that this session has been consolidated as part of close_session.
                await self._mark_session_consolidated(
                    session,
                    reason=SessionConsolidationReason.CLOSE_SESSION,
                    # Store the final user_turns value to indicate at which
                    # point the last consolidation happened.
                    turn=session.user_turns,
                    debug_info=debug_info,
                )

    async def list_sessions_for_user(
        self,
        user_id: str,
        *,
        limit: Optional[int] = None,
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
        # Try to load the session so we can apply domain-level updates
        # (user_turns counter, timestamps, etc.).
        session = await self._storage.get_session(session_id)

        # Increment user_turns only for user messages and only if the
        # session exists. If the session is missing, we delegate error
        # handling to the storage.append_message call below.
        if session is not None and message.role == "user":
            # This updates in-memory state and timestamps; persistence is
            # delegated to save_session().
            user_turns = session.increment_user_turns()
            await self.save_session(session)

            # Decide whether to trigger mid-session consolidation.
            if (
                self._session_memory_consolidation_service is not None
                and session.user_id
            ):
                interval = self._user_turns_consolidation_interval
                # Only trigger if:
                #   - the interval is positive,
                #   - we reached an exact multiple (e.g. 8, 16, 24...),
                #   - and the cooldown since the last consolidation has passed.
                if (
                    interval > 0
                    and (user_turns % interval) == 0
                    and self._is_mid_session_consolidation_allowed(session)
                ):
                    # Fetch the current conversation history for this session.
                    # The consolidation service is responsible for trimming
                    # or summarizing as needed based on its own config.
                    messages = await self.get_history_for_session(session_id)

                    if messages:
                        stored_entries = (
                            await self._session_memory_consolidation_service.consolidate_session(
                                user_id=session.user_id,
                                session_id=session_id,
                                messages=messages,
                            )
                        )

                        # Build a small debug payload based on the stored entries.
                        debug_info = self._build_consolidation_debug_info(
                            stored_entries
                        )

                        # Record consolidation metadata for debugging and future heuristics.
                        await self._mark_session_consolidated(
                            session,
                            reason=SessionConsolidationReason.MID_SESSION,
                            turn=user_turns,
                            debug_info=debug_info,
                        )

        # Delegate message persistence to the storage backend. The storage
        # may apply its own retention/trimming logic (FIFO, max_messages, etc.).
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
          - If a cached value is present in session.user_profile_instructions
            and the session is not marked as requiring a refresh, it is
            returned (after stripping whitespace).
          - Otherwise this method delegates to UserProfileManager, calling
            `get_system_instructions_for_user(user_id)` which returns the
            effective user-level system instructions (already including any
            internal fallbacks), caches the resulting string on the session,
            and saves the updated session.

        Semantics:
          - Instructions are effectively *snapshotted per session*.
            If the underlying user profile changes (e.g. after consolidation),
            the session can be marked as requiring a refresh and will then
            re-resolve instructions on the next call.
        """
        # No associated user or no profile manager → no instructions.
        if not session.user_id:
            return None
        if self._user_profile_manager is None:
            return None

        needs_refresh = session.needs_user_instructions_refresh

        # 1) Try cached instructions from the session.
        cached = session.user_profile_instructions
        if not needs_refresh and isinstance(cached, str):
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

        # 3) Cache on the session and persist.
        session.user_profile_instructions = stripped
        session.needs_user_instructions_refresh = False

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
          - If a cached value is present in session.org_profile_instructions,
            it is returned (after stripping whitespace).
          - Otherwise this method delegates to OrganizationProfileManager, calling
            `get_system_instructions_for_organization(organization_id, ...)`,
            caches the resulting string on the session, and saves the updated
            session.

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

        # 1) Try cached instructions from the session.
        cached = session.org_profile_instructions
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

        # 3) Cache on the session and persist.
        session.org_profile_instructions = stripped
        await self.save_session(session)

        return stripped

    # ------------------------------------------------------------------
    # Consolidation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_consolidation_debug_info(
        entries: Sequence[Any],
    ) -> Dict[str, Any]:
        """
        Build a lightweight debug payload describing the outcome of a
        consolidation run.

        The structure is intentionally simple and JSON-serializable so it can
        be safely stored in session.last_consolidation_debug.
        """
        total = len(entries)

        # Try to infer entry types in a defensive way. We deliberately avoid
        # using getattr(...) here. Instead:
        #   - if the object exposes an 'entry_type' attribute, we use it,
        #   - otherwise we fallback to the literal "unknown".
        type_counts: Dict[str, int] = {}
        for e in entries:
            if hasattr(e, "entry_type"):
                # We ignore type-checker complaints here because not all
                # objects in the list are guaranteed to have this attribute.
                entry_type = e.entry_type  # type: ignore[attr-defined]
            else:
                entry_type = "unknown"

            key = str(entry_type)
            type_counts[key] = type_counts.get(key, 0) + 1

        return {
            "entries_count": total,
            "entry_types": type_counts,
        }

    async def _mark_session_consolidated(
        self,
        session: ChatSession,
        *,
        reason: SessionConsolidationReason,
        turn: Optional[int] = None,
        debug_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Mark the given session as having been consolidated into long-term
        user memory.

        Side effects:
          - Updates typed consolidation fields on the ChatSession:
              last_consolidated_at
              last_consolidated_reason
              last_consolidated_turn
              last_consolidation_debug
              needs_user_instructions_refresh
          - Persists the updated session via save_session().

        The debug payload is intentionally small and JSON-serializable so it
        can be logged or inspected by tooling without additional parsing.
        """
        # When using typed fields we keep the timestamp as a proper datetime
        # object in UTC. If string serialization is needed (e.g. for DB),
        # the storage backend is responsible for that conversion.
        now_utc = datetime.now(timezone.utc)

        session.last_consolidated_at = now_utc
        session.last_consolidated_reason = reason.value

        # Mark that the underlying user profile may have changed
        # (new memory entries, regenerated system_instructions).
        # Existing sessions should refresh their cached instructions
        # on the next call to get_user_profile_instructions_for_session().
        session.needs_user_instructions_refresh = True

        if turn is not None:
            session.last_consolidated_turn = int(turn)

        if debug_info is not None:
            session.last_consolidation_debug = debug_info

        # Persist the updated consolidation metadata (and refresh modification
        # timestamp via save_session()).
        await self.save_session(session)

    def _is_mid_session_consolidation_allowed(self, session: ChatSession) -> bool:
        """
        Check whether we are allowed to run a mid-session consolidation
        for the given session based on a simple cooldown.

        Logic:
          - If cooldown <= 0 → always allowed.
          - If there is no last_consolidated_at on the session → allowed.
          - Otherwise, only allowed if at least `cooldown` seconds have
            passed since the last consolidation.
        """
        cooldown = self._consolidation_cooldown_seconds

        if cooldown <= 0:
            return True

        last_dt = session.last_consolidated_at
        if last_dt is None:
            return True

        # Ensure we are working with an aware UTC datetime.
        if last_dt.tzinfo is None:
            last_dt = last_dt.replace(tzinfo=timezone.utc)

        now = datetime.now(timezone.utc)
        elapsed_seconds = (now - last_dt).total_seconds()

        return elapsed_seconds >= cooldown
