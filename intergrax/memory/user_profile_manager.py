# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Optional

from intergrax.memory.user_profile_memory import (
    UserProfile,
    UserProfilePromptBundle,
    build_profile_prompt_bundle,
)
from intergrax.memory.user_profile_store import UserProfileStore


class UserProfileManager:
    """
    High-level facade for working with user profiles.

    Responsibilities:
      - provide convenient methods to:
          * load or create a UserProfile for a given user_id,
          * persist profile changes,
          * build a prompt-ready bundle for the LLM/runtime;
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
    # Prompt bundle API
    # ---------------------------------------------------------------------

    async def get_prompt_bundle(
        self,
        user_id: str,
        *,
        override_profile: Optional[UserProfile] = None,
    ) -> UserProfilePromptBundle:
        """
        Build a prompt-ready bundle for the given user.

        Behaviour:
          - if `override_profile` is provided, it is used directly;
          - otherwise the profile is loaded from the store for `user_id`.

        The resulting UserProfilePromptBundle is suitable to be injected
        into system instructions or used by the runtime to configure
        language, tone, formatting preferences, etc.
        """
        profile: UserProfile
        if override_profile is not None:
            profile = override_profile
        else:
            profile = await self._store.get_profile(user_id)

        return build_profile_prompt_bundle(profile)
