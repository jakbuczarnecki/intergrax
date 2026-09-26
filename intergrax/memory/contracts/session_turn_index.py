# © Artur Czarnecki. All rights reserved.

"""Session turn vector index contracts (Phase MEM-VEC-2.1)."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from intergrax.llm.messages import ChatMessage


@dataclass(frozen=True, slots=True)
class SessionTurnIndexHit:
    """One episodic session turn search hit (MEM-ENT-13B)."""

    entry_id: str
    tenant_id: str
    session_id: str
    user_id: str | None
    message: ChatMessage
    score: float

    def __post_init__(self) -> None:
        if not (self.entry_id or "").strip():
            raise ValueError("entry_id must be non-empty")
        if not (self.tenant_id or "").strip():
            raise ValueError("tenant_id must be non-empty")
        if not (self.session_id or "").strip():
            raise ValueError("session_id must be non-empty")
        if not math.isfinite(self.score):
            raise ValueError("score must be finite")


@runtime_checkable
class SessionTurnIndexEmbeddingPort(Protocol):
    """Neutral embedding port for session turn index materialization."""

    def embed_texts(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...


@runtime_checkable
class SessionTurnIndexVectorScope(Protocol):
    """Vector query scope for session turn index backends (read-only data view)."""

    @property
    def tenant_id(self) -> str: ...

    @property
    def namespace(self) -> str | None: ...

    @property
    def workspace_id(self) -> str | None: ...


@runtime_checkable
class SessionTurnIndexMetadataFilter(Protocol):
    """Metadata filter passed to vector query backends (read-only data view)."""

    @property
    def conditions(self) -> Mapping[str, str | int | float]: ...


@runtime_checkable
class SessionTurnIndexVectorQueryHit(Protocol):
    """One raw vector query hit before session turn normalization (read-only data view)."""

    @property
    def similarity_score(self) -> float: ...

    @property
    def document_content(self) -> str: ...

    @property
    def document_id(self) -> str: ...

    @property
    def document_metadata(self) -> Mapping[str, str | int | float]: ...


@runtime_checkable
class SessionTurnIndexVectorUpsertRecord(Protocol):
    """One vector upsert row for session turn indexing (read-only data view)."""

    @property
    def vector_id(self) -> str: ...

    @property
    def document_content(self) -> str: ...

    @property
    def document_metadata(self) -> Mapping[str, str | int | float]: ...

    @property
    def embedding(self) -> Sequence[float]: ...


@runtime_checkable
class SessionTurnIndexVectorstorePort(Protocol):
    """Neutral vectorstore port for session turn index materialization."""

    def add_records(
        self,
        records: Sequence[SessionTurnIndexVectorUpsertRecord],
        *,
        scope: SessionTurnIndexVectorScope,
    ) -> None: ...

    def query(
        self,
        embedding: Sequence[float],
        *,
        scope: SessionTurnIndexVectorScope,
        top_k: int,
        metadata_filter: SessionTurnIndexMetadataFilter | None = None,
    ) -> Sequence[SessionTurnIndexVectorQueryHit]: ...

    def delete(
        self,
        ids: Sequence[str],
        *,
        scope: SessionTurnIndexVectorScope,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class SessionTurnIndexStoreCreationContext:
    """Vendor-neutral plugin materialization inputs for session turn index stores."""

    tenant_id: str
    index_roles: tuple[str, ...] = ("user", "assistant")
    vector_index_namespace: str | None = None
    workspace_id: str | None = None
    embedding_manager: SessionTurnIndexEmbeddingPort | None = None
    vectorstore_manager: SessionTurnIndexVectorstorePort | None = None


@runtime_checkable
class SessionTurnIndexStore(Protocol):
    """Episodic vector index over session turns — index over ``SessionStorage``, not a replacement."""

    async def upsert_turn(
        self,
        *,
        tenant_id: str,
        session_id: str,
        user_id: str | None,
        message: ChatMessage,
    ) -> None: ...

    async def tombstone_turn(self, entry_id: str) -> None: ...

    async def search_turns(
        self,
        *,
        query: str,
        tenant_id: str,
        session_id: str | None = None,
        user_id: str | None = None,
        top_k: int = 8,
        score_threshold: float | None = None,
        include_cross_session: bool = False,
    ) -> list[SessionTurnIndexHit]: ...


@runtime_checkable
class SessionTurnIndexStorePlugin(Protocol):
    """Entry-point plugin for custom episodic index backends (MEM-VEC-3.1)."""

    @classmethod
    def plugin_id(cls) -> str: ...

    @classmethod
    def create_session_turn_index(
        cls,
        context: SessionTurnIndexStoreCreationContext,
    ) -> SessionTurnIndexStore: ...
