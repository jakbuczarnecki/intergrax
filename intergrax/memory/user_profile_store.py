# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Protocol

from intergrax.memory.user_profile_memory import UserProfile


class UserProfileStore(Protocol):
    """
    Persistent storage interface for user profiles.

    This store is responsible for:
    - loading and saving the `UserProfile` aggregate,
    - providing sane defaults for new users,
    - hiding backend-specific concerns (JSON files, SQL DB, etc.).

    It MUST NOT:
    - implement LLM prompt logic,
    - perform RAG operations,
    - decide how profile is injected into prompts.
    """

    async def get_profile(self, user_id: str) -> UserProfile:
        """
        Load user profile for the given user_id.

        Implementations SHOULD:
        - return an initialized profile even if no data exists yet
          (e.g. with default identity/preferences),
        - never return None.
        """
        ...

    async def save_profile(self, profile: UserProfile) -> None:
        """
        Persist the given profile aggregate for the associated user_id.

        This MUST overwrite any previously stored profile for that user.
        """
        ...

    async def delete_profile(self, user_id: str) -> None:
        """
        Remove any stored profile data for the given user_id.

        Implementations MUST tolerate unknown user_ids without error.
        """
        ...
