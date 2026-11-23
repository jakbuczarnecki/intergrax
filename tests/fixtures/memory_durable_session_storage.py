# © Artur Czarnecki. All rights reserved.

"""Durable-declaring session-storage plugin for tests (T-IGX-043-B, shared fixture).

The fixture gives a PRODUCT or STRICT memory wiring a session storage that passes the
session-storage admission through the plugin branch: the store declares durability
``DURABLE``, is not reference-only and declares a ``QUALIFIED`` status, the same self-declared
pattern as ``_SelfCertifiedDurableUserProfileStore`` of the 5A admission tests. It is a test
double over a dictionary: it does not subclass ``InMemorySessionStorage`` (the admission
refuses known-ephemeral classes) and it carries no platform qualification evidence.
"""

from __future__ import annotations

import uuid
from typing import Any

from intergrax.applications.contracts.environment_profile import MemoryProfile
from intergrax.llm.messages import ChatMessage
from intergrax.memory.contracts.chat_session import ChatSession
from intergrax.memory.contracts.memory_store_creation_context import SessionStorageCreationContext
from intergrax.memory.contracts.provider_admission import (
    MemoryProviderDurability,
    MemoryProviderQualificationStatus,
)
from intergrax.memory.contracts.session_storage import SessionStorage

DURABLE_SESSION_PLUGIN_ID = "test.durable_declared_session_storage"


class DurableDeclaredSessionStorage:
    """Dictionary-backed ``SessionStorage`` that declares durable provider metadata."""

    def __init__(self, provider_id: str = DURABLE_SESSION_PLUGIN_ID) -> None:
        self._provider_id = provider_id
        self._sessions: dict[tuple[str, str], ChatSession] = {}
        self._history: dict[tuple[str, str], list[ChatMessage]] = {}

    @property
    def memory_provider_id(self) -> str:
        return self._provider_id

    @property
    def memory_provider_durability(self) -> MemoryProviderDurability:
        return MemoryProviderDurability.DURABLE

    @property
    def memory_provider_reference_only(self) -> bool:
        return False

    @property
    def memory_provider_declared_qualification_status(self) -> MemoryProviderQualificationStatus:
        return MemoryProviderQualificationStatus.QUALIFIED

    @property
    def memory_provider_version(self) -> str | None:
        return None

    async def get_session(self, *, tenant_id: str, session_id: str) -> ChatSession | None:
        return self._sessions.get((tenant_id, session_id))

    async def create_session(
        self,
        *,
        tenant_id: str,
        session_id: str | None = None,
        user_id: str | None = None,
        workspace_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ChatSession:
        _ = metadata
        session = ChatSession(
            id=session_id or str(uuid.uuid4()),
            tenant_id=tenant_id,
            user_id=user_id,
            workspace_id=workspace_id,
        )
        self._sessions[(tenant_id, session.id)] = session
        self._history[(tenant_id, session.id)] = []
        return session

    async def save_session(self, session: ChatSession) -> None:
        self._sessions[(session.tenant_id, session.id)] = session

    async def list_sessions_for_user(
        self,
        *,
        tenant_id: str,
        user_id: str,
        limit: int | None = None,
    ) -> list[ChatSession]:
        sessions = [s for (tenant, _), s in self._sessions.items() if tenant == tenant_id and s.user_id == user_id]
        return sessions if limit is None else sessions[:limit]

    async def append_message(self, *, tenant_id: str, session_id: str, message: ChatMessage) -> ChatMessage:
        self._history.setdefault((tenant_id, session_id), []).append(message)
        return message

    async def get_history(
        self,
        *,
        tenant_id: str,
        session_id: str,
        native_tools: bool = False,
    ) -> list[ChatMessage]:
        _ = native_tools
        return list(self._history.get((tenant_id, session_id), []))


class DurableDeclaredSessionStoragePlugin:
    """Session-storage plugin that materializes ``DurableDeclaredSessionStorage``."""

    @classmethod
    def plugin_id(cls) -> str:
        return DURABLE_SESSION_PLUGIN_ID

    @classmethod
    def create_session_storage(cls, context: SessionStorageCreationContext) -> SessionStorage:
        _ = context
        return DurableDeclaredSessionStorage(provider_id=cls.plugin_id())


def with_durable_session_plugin(memory_profile: MemoryProfile) -> MemoryProfile:
    """Return the profile with the fixture's session-storage plugin selected."""
    return memory_profile.model_copy(update={"session_storage_plugin_id": DURABLE_SESSION_PLUGIN_ID})
