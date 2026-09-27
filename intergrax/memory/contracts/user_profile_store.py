# © Artur Czarnecki. All rights reserved.

"""Canonical UserProfileStore contract (Memory domain)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from intergrax.memory.user_profile_memory import UserProfile


@runtime_checkable
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

    Concurrency:
    - Implementations MUST document whether they are thread-safe, safe under
      overlapping async mutations, and process-safe.
    - Callers MUST serialize concurrent mutations unless the implementation
      explicitly documents a stronger guarantee.
    """

    async def get_profile(
        self,
        *,
        tenant_id: str,
        user_id: str,
    ) -> UserProfile:
        """
        Load user profile for the given user_id.

        Implementations SHOULD:
        - return an initialized profile even if no data exists yet
          (e.g. with default identity/preferences),
        - never return None.
        """
        ...

    async def save_profile(
        self,
        *,
        tenant_id: str,
        profile: UserProfile,
    ) -> None:
        """
        Persist the given profile aggregate for the associated user_id.

        This MUST overwrite any previously stored profile for that user.
        """
        ...

    async def delete_profile(
        self,
        *,
        tenant_id: str,
        user_id: str,
    ) -> None:
        """
        Remove any stored profile data for the given user_id.

        Implementations MUST tolerate unknown user_ids without error.
        """
        ...


__all__ = ["UserProfileStore"]
