# © Artur Czarnecki. All rights reserved.

"""Memory store plugin contracts (Phase MEM-3.1)."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from intergrax.memory.user_profile_store import UserProfileStore
from intergrax.runtime.nexus.session.session_storage import SessionStorage


@runtime_checkable
class UserProfileStorePlugin(Protocol):
    """Plugin that materializes a ``UserProfileStore`` backend."""

    @classmethod
    def plugin_id(cls) -> str: ...

    @classmethod
    def create_user_profile_store(cls, **kwargs: Any) -> UserProfileStore: ...


@runtime_checkable
class SessionStoragePlugin(Protocol):
    """Plugin that materializes a ``SessionStorage`` backend."""

    @classmethod
    def plugin_id(cls) -> str: ...

    @classmethod
    def create_session_storage(cls, **kwargs: Any) -> SessionStorage: ...
