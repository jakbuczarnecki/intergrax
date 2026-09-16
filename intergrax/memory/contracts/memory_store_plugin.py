# © Artur Czarnecki. All rights reserved.

"""Memory store plugin contracts (Phase MEM-3.1)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from intergrax.memory.contracts.entity_temporal_memory import EntityTemporalMemoryStore
    from intergrax.memory.contracts.long_horizon_memory import LongHorizonMemoryStore
    from intergrax.memory.contracts.procedural_memory import ProcedureMemoryStore
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


@runtime_checkable
class EntityTemporalMemoryStorePlugin(Protocol):
    """Plugin that materializes an ``EntityTemporalMemoryStore`` backend."""

    @classmethod
    def plugin_id(cls) -> str: ...

    @classmethod
    def create_entity_temporal_memory_store(
        cls,
        **kwargs: Any,
    ) -> EntityTemporalMemoryStore: ...


@runtime_checkable
class ProceduralMemoryStorePlugin(Protocol):
    """Plugin that materializes a ``ProcedureMemoryStore`` backend."""

    @classmethod
    def plugin_id(cls) -> str: ...

    @classmethod
    def create_procedural_memory_store(
        cls,
        **kwargs: Any,
    ) -> ProcedureMemoryStore: ...


@runtime_checkable
class LongHorizonMemoryStorePlugin(Protocol):
    """Plugin that materializes a ``LongHorizonMemoryStore`` backend."""

    @classmethod
    def plugin_id(cls) -> str: ...

    @classmethod
    def create_long_horizon_memory_store(
        cls,
        **kwargs: Any,
    ) -> LongHorizonMemoryStore: ...
