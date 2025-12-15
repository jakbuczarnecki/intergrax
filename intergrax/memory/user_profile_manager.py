# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Optional, Dict, Any, List, Union

from intergrax.memory.user_profile_memory import (
    UserProfile,
    UserProfileMemoryEntry,
)
from intergrax.memory.user_profile_store import UserProfileStore


class UserProfileManager:
    """
    High-level facade for working with user profiles.

    Responsibilities:
      - provide convenient methods to:
          * load or create a UserProfile for a given user_id,
          * persist profile changes,
          * manage long-term user memory entries,
          * manage system-level instructions derived from the profile;
      - hide direct interaction with the underlying UserProfileStore.

    It intentionally does NOT:
      - call LLMs directly,
      - perform RAG over long-term user memory,
      - decide *when* the profile should be updated (this is a policy concern
        for higher-level components such as the runtime or application logic).
    """

    def __init__(self, store: UserProfileStore) -> None:
        self._store = store

    # ---------------------------------------------------------------------
    # Core profile APIs
    # ---------------------------------------------------------------------

    async def get_profile(self, user_id: str) -> UserProfile:
        """
        Load the user profile for the given user_id.

        Implementations of UserProfileStore are expected to return an
        initialized profile even if no data exists yet for that user.
        """
        return await self._store.get_profile(user_id)

    async def save_profile(self, profile: UserProfile) -> None:
        """
        Persist the given UserProfile aggregate.

        This MUST overwrite any previously stored profile for the same user.
        """
        await self._store.save_profile(profile)

    async def delete_profile(self, user_id: str) -> None:
        """
        Remove any stored profile data for the given user_id.

        This operation is typically used for cleanup or account deletion flows.
        """
        await self._store.delete_profile(user_id)

    # ---------------------------------------------------------------------
    # System instructions management
    # ---------------------------------------------------------------------

    async def get_system_instructions_for_user(self, user_id: str) -> str:
        """
        Return a compact system-level instruction string for the given user.

        Behavior:
          - loads the user's profile from the store,
          - uses the profile's `system_instructions` if set,
          - otherwise builds a deterministic fallback based on identity
            and preferences via `UserProfile.build_default_system_instructions()`.

        This method does NOT call any LLM and does NOT use long-term memory.
        Higher-level components may choose to update `system_instructions`
        using LLMs and then persist the result via `update_system_instructions()`.
        """
        profile = await self._store.get_profile(user_id)
        return self._build_default_system_instructions(profile)

    async def update_system_instructions(
        self,
        user_id: str,
        instructions: str,
    ) -> UserProfile:
        """
        Update the `system_instructions` field of the user's profile.

        This method assumes that some higher-level component (e.g. the runtime
        or a batch job) has already decided *what* the new instructions should be,
        possibly by calling an LLM over `memory_entries` and other data.

        The manager is responsible only for:
          - loading the profile,
          - updating the field,
          - persisting the aggregate.

        Returns the updated UserProfile for convenience.
        """
        profile = await self._store.get_profile(user_id)
        normalized = instructions.strip()
        profile.system_instructions = normalized or None
        profile.modified=True
        await self._store.save_profile(profile)
        profile.modified=False
        return profile

    # ---------------------------------------------------------------------
    # Long-term memory management
    # ---------------------------------------------------------------------

    async def add_memory_entry(
        self,
        user_id: str,
        entry_or_content: Union[UserProfileMemoryEntry, str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> UserProfile:
        """
        Append a new long-term memory entry to the user's profile.

        This method only updates the profile aggregate and persists it via
        the store. It does NOT call any LLM and does NOT update
        `system_instructions` automatically.

        Returns the updated UserProfile for convenience.
        """
        profile = await self._store.get_profile(user_id)

        if isinstance(entry_or_content, UserProfileMemoryEntry):
            entry = entry_or_content
            # Ensure metadata dict exists (avoid None)
            if entry.metadata is None:
                entry.metadata = {}
        else:
            entry = UserProfileMemoryEntry(
                content=str(entry_or_content),
                metadata=metadata or {},
            )

        profile.memory_entries.append(entry)

        await self._store.save_profile(profile)
        
        return profile

    async def update_memory_entry(
        self,
        user_id: str,
        entry_id: int,
        *,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> UserProfile:
        """
        Update a single long-term memory entry identified by `entry_id`.
        """
        profile = await self._store.get_profile(user_id)

        for entry in profile.memory_entries:
            if entry.entry_id == entry_id:
                if content is not None:
                    entry.content = content
                if metadata is not None:
                    entry.metadata = metadata
                entry.modified=True                
                break

        await self._store.save_profile(profile)

        if entry:
            entry.modified=False

        return profile

    async def remove_memory_entry(
        self,
        user_id: str,
        entry_id: int,
    ) -> UserProfile:
        """
        Remove a single long-term memory entry identified by `entry_id`.
        """
        profile = await self._store.get_profile(user_id)

        found = False
        for entry in profile.memory_entries:
            if entry.entry_id == entry_id:
                entry.deleted = True
                found = True
                break

        if not found:
            return profile
       
        await self._store.save_profile(profile)

        # profile.memory_entries = [
        #     e for e in profile.memory_entries if e.entry_id != entry_id
        # ]

        return profile


    async def clear_memory(self, user_id: str) -> UserProfile:
        """
        Remove all long-term memory entries for the given user.

        This is usually used for privacy/cleanup flows or when the application
        decides to reset user-level memory.
        """
        profile = await self._store.get_profile(user_id)     
        
        changed = False
        for entry in profile.memory_entries:
            if not entry.deleted:
                entry.deleted=True  
                changed = True
        
        if changed:
            await self._store.save_profile(profile)
            # profile.memory_entries.clear()

        return profile
    

    def _build_default_system_instructions(self, profile: UserProfile) -> str:
        """
        Deterministic, non-LLM helper that builds system instructions
        from the given profile (identity + preferences) when the profile
        does not yet have explicit system_instructions.
        """
        if profile.system_instructions:
            return profile.system_instructions.strip()

        identity = profile.identity
        prefs = profile.preferences

        lines: list[str] = []

        # Identity
        if identity.display_name:
            lines.append(f"You are talking to {identity.display_name}.")
        else:
            lines.append(f"You are talking to a user with id '{identity.user_id}'.")

        if identity.role:
            lines.append(f"The user is: {identity.role}.")
        if identity.domain_expertise:
            lines.append(f"Domain expertise: {identity.domain_expertise}.")

        # Language / style
        if prefs.preferred_language:
            lines.append(
                f"Always answer in {prefs.preferred_language} unless explicitly asked otherwise."
            )
        if prefs.tone:
            lines.append(f"Default tone: {prefs.tone}.")
        if prefs.answer_length:
            lines.append(f"Default answer length: {prefs.answer_length}.")

        # Formatting rules
        if prefs.no_emojis_in_code:
            lines.append("Never use emojis in code blocks.")
        if prefs.no_emojis_in_docs:
            lines.append("Avoid emojis in technical documentation.")
        if prefs.default_project_context:
            lines.append(
                f"Assume the default project context is: {prefs.default_project_context}."
            )

        if not lines:
            lines.append(
                "You are talking to a user. Use a helpful, concise, and technical style by default."
            )

        return " ".join(lines)


    async def purge_deleted_memory_entries(self, user_id: str) -> UserProfile:
        """
        Permanently remove entries marked as deleted=True from the profile aggregate.

        This is a maintenance operation. Normal read flows should still ignore
        deleted entries even if purge is not called.
        """
        profile = await self._store.get_profile(user_id)

        before = len(profile.memory_entries)
        profile.memory_entries = [e for e in profile.memory_entries if not e.deleted]
        after = len(profile.memory_entries)

        if after != before:
            await self._store.save_profile(profile)

        return profile