# Reference external memory store plugin for gate tests (Phase MEM-3.3).

from __future__ import annotations

from intergrax.memory.contracts.memory_store_creation_context import (
    SessionStorageCreationContext,
    UserProfileStoreCreationContext,
)
from intergrax.memory.stores.in_memory_user_profile_store import InMemoryUserProfileStore
from intergrax.memory.user_profile_store import UserProfileStore
from intergrax.memory.contracts.session_storage import SessionStorage

from .fixture_session_storage import FixtureExternalSessionStorage


class FixtureExternalUserProfileStore(InMemoryUserProfileStore):
    """Fixture marker store for integration tests."""


class ExternalInMemoryUserProfileStorePlugin:
    @classmethod
    def plugin_id(cls) -> str:
        return "external.in_memory_user_profile"

    @classmethod
    def create_user_profile_store(
        cls,
        context: UserProfileStoreCreationContext,
    ) -> UserProfileStore:
        _ = context
        return FixtureExternalUserProfileStore()


class ExternalInMemorySessionStoragePlugin:
    @classmethod
    def plugin_id(cls) -> str:
        return "external.in_memory_session_storage"

    @classmethod
    def create_session_storage(
        cls,
        context: SessionStorageCreationContext,
    ) -> SessionStorage:
        _ = context
        return FixtureExternalSessionStorage()
