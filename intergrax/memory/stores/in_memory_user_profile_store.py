# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Dict

from intergrax.memory.user_profile_memory import UserProfile, UserIdentity, UserPreferences
from intergrax.memory.user_profile_store import UserProfileStore


class InMemoryUserProfileStore(UserProfileStore):
    """
    In-memory implementation of UserProfileStore.

    Use cases:
      - unit tests,
      - local development,
      - experiments and notebooks.

    This implementation does NOT provide durability or cross-process sharing.
    """

    def __init__(self) -> None:
        # user_id -> UserProfile
        self._profiles: Dict[str, UserProfile] = {}

    async def get_profile(self, user_id: str) -> UserProfile:
        """
        Return an existing profile or a default one if not present.
        """
        if user_id in self._profiles:
            return self._profiles[user_id]

        # Create a default, mostly empty profile for a new user.
        identity = UserIdentity(user_id=user_id)
        preferences = UserPreferences()
        profile = UserProfile(identity=identity, preferences=preferences)

        # Optionally store the default profile to make subsequent calls cheaper.
        self._profiles[user_id] = profile
        return profile

    async def save_profile(self, profile: UserProfile) -> None:
        """
        Persist or update the profile in memory.
        """
        self._profiles[profile.identity.user_id] = profile

    async def delete_profile(self, user_id: str) -> None:
        """
        Remove a stored profile, if present. Ignore unknown IDs.
        """
        self._profiles.pop(user_id, None)

    # Optional helper for debugging / tests
    def list_user_ids(self):
        return list(self._profiles.keys())
