# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Dict, Tuple

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
        self._profiles: Dict[Tuple[str, str], UserProfile] = {}

    async def get_profile(
        self,
        *,
        tenant_id: str,
        user_id: str,
    ) -> UserProfile:
        key = (tenant_id, user_id)
        if key in self._profiles:
            return self._profiles[key]

        identity = UserIdentity(user_id=user_id)
        preferences = UserPreferences()
        profile = UserProfile(identity=identity, preferences=preferences)

        self._profiles[key] = profile
        return profile

    async def save_profile(
        self,
        *,
        tenant_id: str,
        profile: UserProfile,
    ) -> None:
        key = (tenant_id, profile.identity.user_id)
        self._profiles[key] = profile

    async def delete_profile(
        self,
        *,
        tenant_id: str,
        user_id: str,
    ) -> None:
        key = (tenant_id, user_id)
        self._profiles.pop(key, None)

    # Optional helper for debugging / tests
    def list_user_ids(self):
        return list(self._profiles.keys())
