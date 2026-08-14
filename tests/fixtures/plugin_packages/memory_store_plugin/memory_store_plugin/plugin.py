# Reference external memory store plugin for gate tests (Phase MEM-3.3).

from __future__ import annotations

from typing import Any

from intergrax.memory.stores.in_memory_user_profile_store import InMemoryUserProfileStore
from intergrax.memory.user_profile_store import UserProfileStore
from intergrax.runtime.nexus.session.in_memory_session_storage import InMemorySessionStorage
from intergrax.runtime.nexus.session.session_storage import SessionStorage


class FixtureExternalUserProfileStore(InMemoryUserProfileStore):
    """Fixture marker store for integration tests."""


class FixtureExternalSessionStorage(InMemorySessionStorage):
    """Fixture marker store for integration tests."""


class ExternalInMemoryUserProfileStorePlugin:
    @classmethod
    def plugin_id(cls) -> str:
        return "external.in_memory_user_profile"

    @classmethod
    def create_user_profile_store(cls, **kwargs: Any) -> UserProfileStore:
        _ = kwargs
        return FixtureExternalUserProfileStore()


class ExternalInMemorySessionStoragePlugin:
    @classmethod
    def plugin_id(cls) -> str:
        return "external.in_memory_session_storage"

    @classmethod
    def create_session_storage(cls, **kwargs: Any) -> SessionStorage:
        _ = kwargs
        return FixtureExternalSessionStorage()
