# © Artur Czarnecki. All rights reserved.

"""Typed fault-injection helpers for MEM-ENT-14 resilience proofs (test-only)."""

from __future__ import annotations

from intergrax.memory.user_profile_memory import UserProfile
from intergrax.memory.stores.in_memory_user_profile_store import InMemoryUserProfileStore


class AmbiguousCommitUserProfileStore(InMemoryUserProfileStore):
    """Persists profile then raises (simulates timeout after provider commit)."""

    def __init__(
        self,
        *,
        error_type: type[BaseException] = TimeoutError,
        error_message: str = "ambiguous provider timeout",
    ) -> None:
        super().__init__()
        self.error_type = error_type
        self.error_message = error_message
        self.commit_count = 0

    async def save_profile(self, *, tenant_id: str, profile: UserProfile) -> None:
        await super().save_profile(tenant_id=tenant_id, profile=profile)
        self.commit_count += 1
        raise self.error_type(self.error_message)


class FailBeforeCommitUserProfileStore(InMemoryUserProfileStore):
    """Raises before any persistence (retry must not create duplicate)."""

    def __init__(
        self,
        *,
        error_type: type[BaseException] = TimeoutError,
        error_message: str = "fail before commit",
    ) -> None:
        super().__init__()
        self.error_type = error_type
        self.error_message = error_message
        self.attempts = 0

    async def save_profile(self, *, tenant_id: str, profile: UserProfile) -> None:
        self.attempts += 1
        raise self.error_type(self.error_message)
