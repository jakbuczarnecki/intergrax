# © Artur Czarnecki. All rights reserved.

"""ENTERPRISE-5 / BLOCK D: Memory store plugin classifier unit tests."""

from __future__ import annotations

import pytest

from intergrax.memory.resolver import MemoryStorePluginKind, classify_memory_store_plugin
from intergrax.memory.session_turn_index_service import VectorSessionTurnIndexStore
from intergrax.memory.stores.in_memory_user_profile_store import InMemoryUserProfileStore
from intergrax.runtime.nexus.session.in_memory_session_storage import InMemorySessionStorage

pytestmark = pytest.mark.unit


class _UserProfilePlugin:
    @classmethod
    def plugin_id(cls) -> str:
        return "test.user_profile"

    @classmethod
    def create_user_profile_store(cls, **_kwargs):
        return InMemoryUserProfileStore()


class _SessionStoragePlugin:
    @classmethod
    def plugin_id(cls) -> str:
        return "test.session_storage"

    @classmethod
    def create_session_storage(cls, **_kwargs):
        return InMemorySessionStorage()


class _SessionTurnIndexPlugin:
    @classmethod
    def plugin_id(cls) -> str:
        return "test.session_turn_index"

    @classmethod
    def create_session_turn_index(cls, **_kwargs) -> VectorSessionTurnIndexStore:
        raise NotImplementedError


def test_classifier_user_profile_store_kind() -> None:
    assert classify_memory_store_plugin(_UserProfilePlugin) is MemoryStorePluginKind.USER_PROFILE_STORE


def test_classifier_session_storage_kind() -> None:
    assert classify_memory_store_plugin(_SessionStoragePlugin) is MemoryStorePluginKind.SESSION_STORAGE


def test_classifier_session_turn_index_kind() -> None:
    assert classify_memory_store_plugin(_SessionTurnIndexPlugin) is MemoryStorePluginKind.SESSION_TURN_INDEX


def test_classifier_prefers_session_turn_index_when_all_methods_present() -> None:
    class _MultiCapabilityPlugin:
        @classmethod
        def plugin_id(cls) -> str:
            return "test.multi"

        @classmethod
        def create_user_profile_store(cls, **_kwargs):
            return InMemoryUserProfileStore()

        @classmethod
        def create_session_storage(cls, **_kwargs):
            return InMemorySessionStorage()

        @classmethod
        def create_session_turn_index(cls, **_kwargs) -> VectorSessionTurnIndexStore:
            raise NotImplementedError

    assert (
        classify_memory_store_plugin(_MultiCapabilityPlugin)
        is MemoryStorePluginKind.SESSION_TURN_INDEX
    )
