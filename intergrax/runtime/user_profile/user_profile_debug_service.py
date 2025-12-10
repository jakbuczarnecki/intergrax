# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import asdict
from typing import List, Optional
from datetime import datetime

from intergrax.memory.user_profile_manager import UserProfileManager
from intergrax.memory.user_profile_memory import UserProfile, UserProfileMemoryEntry
from intergrax.runtime.drop_in_knowledge_mode.session.session_manager import SessionManager
from intergrax.runtime.user_profile.user_profile_debug_snapshot import (
    UserProfileDebugSnapshot,
    SessionDebugView,
    MemoryEntryDebugView,
    UNKNOWN,
)


class UserProfileDebugService:
    """
    High-level service responsible for building UserProfileDebugSnapshot
    objects for a given user.

    This service is read-only and does not introduce any new behavior in
    the runtime. It aggregates data from:

      - UserProfileManager (identity, preferences, memory, system_instructions),
      - SessionManager (recent ChatSession metadata).

    Typical usage:
      - exposing a "debug user profile" API endpoint,
      - feeding an admin / developer UI panel,
      - ad-hoc diagnostics during development.
    """

    def __init__(
        self,
        user_profile_manager: UserProfileManager,
        session_manager: SessionManager,
        *,
        max_sessions: int = 10,
        max_memory_entries: int = 50,
    ) -> None:
        self._user_profile_manager = user_profile_manager
        self._session_manager = session_manager
        self._max_sessions = max_sessions
        self._max_memory_entries = max_memory_entries

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_debug_snapshot(self, user_id: str) -> UserProfileDebugSnapshot:
        """
        Build and return a UserProfileDebugSnapshot for the given user_id.

        This method performs only read operations on the underlying managers.
        It does not modify any profile, memory entries or sessions.
        """
        profile: UserProfile = await self._user_profile_manager.get_profile(user_id)
        identity_dict = self._build_identity_dict(profile)
        preferences_dict = self._build_preferences_dict(profile)

        # System instructions as currently stored on the profile.
        system_instructions: Optional[str] = profile.system_instructions

        # Long-term memory entries as seen from the profile.
        all_entries: List[UserProfileMemoryEntry] = list(profile.memory_entries)
        memory_entries_total = len(all_entries)
        memory_entries_by_kind = UserProfileDebugSnapshot.build_memory_kind_counters(
            all_entries
        )

        # Select most recent memory entries for detailed debug view.
        recent_memory_entries = self._build_recent_memory_entries(all_entries)

        # Recent sessions for this user.
        recent_sessions = await self._build_recent_sessions(user_id)

        snapshot = UserProfileDebugSnapshot(
            user_id=user_id,
            identity=identity_dict,
            preferences=preferences_dict,
            system_instructions=system_instructions,
            memory_entries_total=memory_entries_total,
            memory_entries_by_kind=memory_entries_by_kind,
            recent_memory_entries=recent_memory_entries,
            recent_sessions=recent_sessions,
        )

        return snapshot

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_identity_dict(self, profile: UserProfile) -> dict:
        """
        Convert the profile.identity object into a plain dict.

        We intentionally only expose a well-defined subset of fields so the
        debug surface remains stable even if the UserProfile model evolves.
        """
        identity = profile.identity

        # If identity is a dataclass, asdict() will work. Otherwise we
        # explicitly map the known fields.
        try:
            identity_dict = asdict(identity)  # type: ignore[arg-type]
        except TypeError:
            identity_dict = {
                "user_id": identity.user_id,
                "display_name": identity.display_name,
                "role": identity.role,
                "domain_expertise": identity.domain_expertise,
                "language": identity.language,
                "locale": identity.locale,
                "timezone": identity.timezone,
            }

        return identity_dict

    def _build_preferences_dict(self, profile: UserProfile) -> dict:
        """
        Convert the profile.preferences object into a plain dict.

        Same rationale as for identity: we expose only the subset of fields
        that is relevant for debug and instructions generation.
        """
        prefs = profile.preferences

        try:
            prefs_dict = asdict(prefs)  # type: ignore[arg-type]
        except TypeError:
            prefs_dict = {
                "preferred_language": prefs.preferred_language,
                "answer_length": prefs.answer_length,
                "tone": prefs.tone,
                "no_emojis_in_code": prefs.no_emojis_in_code,
                "no_emojis_in_docs": prefs.no_emojis_in_docs,
                "prefer_markdown": prefs.prefer_markdown,
                "prefer_code_blocks": prefs.prefer_code_blocks,
                "default_project_context": prefs.default_project_context,
            }

        return prefs_dict

    def _build_recent_memory_entries(
        self,
        all_entries: List[UserProfileMemoryEntry],
    ) -> List[MemoryEntryDebugView]:
        """
        Select a limited number of most recent memory entries and convert
        them into debug views.
        """
        if not all_entries:
            return []

        # Sort entries by created_at descending if available.
        sorted_entries = sorted(
            all_entries,
            key=lambda e: e.created_at if e.created_at is not None else datetime.min,  # type: ignore[name-defined]
            reverse=True,
        )

        limited_entries = sorted_entries[: self._max_memory_entries]

        result: List[MemoryEntryDebugView] = []
        for entry in limited_entries:
            result.append(UserProfileDebugSnapshot.from_memory_entry(entry))

        return result

    async def _build_recent_sessions(
        self,
        user_id: str,
    ) -> List[SessionDebugView]:
        """
        Fetch recent sessions for the given user and convert them into
        debug views.
        """
        sessions = await self._session_manager.list_sessions_for_user(
            user_id=user_id,
            limit=self._max_sessions,
        )

        result: List[SessionDebugView] = []
        for session in sessions:
            result.append(UserProfileDebugSnapshot.from_domain_session(session))

        return result
