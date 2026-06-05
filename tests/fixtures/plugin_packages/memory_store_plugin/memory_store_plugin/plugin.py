# Reference external memory store plugin for gate tests (Phase MEM-3.3).

from __future__ import annotations

from typing import Any

from intergrax.memory.stores.in_memory_user_profile_store import InMemoryUserProfileStore
from intergrax.memory.user_profile_store import UserProfileStore


class ExternalInMemoryUserProfileStorePlugin:
    @classmethod
    def plugin_id(cls) -> str:
        return "external.in_memory_user_profile"

    @classmethod
    def create_user_profile_store(cls, **kwargs: Any) -> UserProfileStore:
        _ = kwargs
        return InMemoryUserProfileStore()
