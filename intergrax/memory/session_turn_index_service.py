# © Artur Czarnecki. All rights reserved.

"""Default episodic session turn vector index (Phase MEM-VEC-2.1–2.2)."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from intergrax.llm.messages import ChatMessage, MessageRole
from intergrax.memory.contracts.session_turn_index import (
    SessionTurnIndexEmbeddingPort,
    SessionTurnIndexHit,
    SessionTurnIndexMetadataFilter,
    SessionTurnIndexStore,
    SessionTurnIndexVectorScope,
    SessionTurnIndexVectorUpsertRecord,
    SessionTurnIndexVectorstorePort,
)
from intergrax.memory.memory_vector_namespace import (
    EPISODIC_INDEX_DOMAIN,
    resolve_memory_index_collection,
)
from intergrax.memory.memory_vector_errors import MemoryTenantScopeViolationError


def _normalize_message_role(value: object) -> MessageRole:
    if value == "system":
        return "system"
    if value == "assistant":
        return "assistant"
    if value == "tool":
        return "tool"
    if value == "user":
        return "user"
    return "user"


def _sanitize_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in meta.items():
        if value is None or isinstance(value, (str, int, float, bool)):
            out[key] = value
        elif isinstance(value, (list, tuple)):
            out[key] = ",".join(str(item) for item in value)
        elif isinstance(value, dict):
            out[key] = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        else:
            out[key] = str(value)
    return out


def _metadata_for_port(meta: dict[str, Any]) -> dict[str, str | int | float]:
    out: dict[str, str | int | float] = {}
    for key, value in meta.items():
        if isinstance(value, bool):
            out[key] = int(value)
        elif isinstance(value, (str, int, float)):
            out[key] = value
        else:
            out[key] = str(value)
    return out


@dataclass(frozen=True, slots=True)
class _BoundVectorScope:
    tenant_id: str
    namespace: str | None
    workspace_id: str | None


@dataclass(frozen=True, slots=True)
class _TurnMetadataFilter:
    conditions: Mapping[str, str | int | float]


@dataclass(frozen=True, slots=True)
class _TurnUpsertRecord:
    vector_id: str
    document_content: str
    document_metadata: Mapping[str, str | int | float]
    embedding: Sequence[float]


class VectorSessionTurnIndexStore(SessionTurnIndexStore):
    """Vectorstore-backed episodic index with ``index_domain=episodic`` metadata."""

    def __init__(
        self,
        *,
        embedding_port: SessionTurnIndexEmbeddingPort,
        vectorstore_port: SessionTurnIndexVectorstorePort,
        index_roles: Sequence[str] = ("user", "assistant"),
        tenant_id: str = "default",
        vector_index_namespace: str | None = None,
        workspace_id: str | None = None,
    ) -> None:
        self._embedding_port = embedding_port
        self._vectorstore_port = vectorstore_port
        self._index_roles = tuple(index_roles)
        self._tenant_id = tenant_id
        self._vector_index_namespace = vector_index_namespace
        self._workspace_id = workspace_id
        self._collection_name = resolve_memory_index_collection(
            vector_index_namespace=vector_index_namespace,
            tenant_id=tenant_id,
            domain=EPISODIC_INDEX_DOMAIN,
        )

    async def upsert_turn(
        self,
        *,
        tenant_id: str,
        session_id: str,
        user_id: str | None,
        message: ChatMessage,
    ) -> None:
        if message.deleted:
            self._resolve_bound_tenant(tenant_id)
            await self.tombstone_turn(message.entry_id)
            return
        if message.role not in self._index_roles:
            return
        text = (message.content or "").strip()
        if not text:
            return
        scope = self._bound_scope(tenant_id=tenant_id)
        meta = _metadata_for_port(
            _sanitize_metadata(
                {
                    "session_id": session_id,
                    "user_id": user_id or "",
                    "entry_id": message.entry_id,
                    "role": message.role,
                    "deleted": 0,
                    "index_domain": EPISODIC_INDEX_DOMAIN,
                    "collection_name": self._collection_name,
                }
            )
        )
        embeddings = self._embedding_port.embed_texts([text])
        embedding = tuple(float(x) for x in embeddings[0])
        self._vectorstore_port.add_records(
            [
                _TurnUpsertRecord(
                    vector_id=message.entry_id,
                    document_content=text,
                    document_metadata=meta,
                    embedding=embedding,
                )
            ],
            scope=scope,
        )

    def _resolve_bound_tenant(self, tenant_id: str | None) -> str:
        requested = tenant_id if tenant_id is not None else self._tenant_id
        if requested != self._tenant_id:
            raise MemoryTenantScopeViolationError(
                expected_tenant_id=self._tenant_id,
                requested_tenant_id=requested,
            )
        return self._tenant_id

    def _bound_scope(self, *, tenant_id: str) -> SessionTurnIndexVectorScope:
        bound_tenant = self._resolve_bound_tenant(tenant_id)
        return _BoundVectorScope(
            tenant_id=bound_tenant,
            namespace=self._vector_index_namespace,
            workspace_id=self._workspace_id,
        )

    async def tombstone_turn(self, entry_id: str) -> None:
        if not entry_id:
            return
        self._vectorstore_port.delete(
            [entry_id],
            scope=self._bound_scope(tenant_id=self._tenant_id),
        )

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
    ) -> list[SessionTurnIndexHit]:
        q = (query or "").strip()
        if not q:
            return []
        scope = self._bound_scope(tenant_id=tenant_id)
        where: dict[str, str | int | float] = {
            "deleted": 0,
            "index_domain": EPISODIC_INDEX_DOMAIN,
            "collection_name": self._collection_name,
        }
        if include_cross_session:
            if user_id:
                where["user_id"] = user_id
        else:
            if session_id:
                where["session_id"] = session_id
            if user_id:
                where["user_id"] = user_id

        q_emb = self._embedding_port.embed_texts([q])
        embedding = tuple(float(x) for x in q_emb[0])
        raw_hits = self._vectorstore_port.query(
            embedding,
            scope=scope,
            top_k=top_k,
            metadata_filter=_TurnMetadataFilter(conditions=where),
        )

        hits: list[SessionTurnIndexHit] = []
        for hit in raw_hits:
            score = float(hit.similarity_score)
            if score_threshold is not None and score < score_threshold:
                continue
            meta = dict(hit.document_metadata)
            entry_id = hit.document_id
            session_id_value = str(meta.get("session_id") or "")
            user_id_value = str(meta.get("user_id") or "") or None
            role = _normalize_message_role(meta.get("role"))
            hits.append(
                SessionTurnIndexHit(
                    entry_id=entry_id,
                    tenant_id=tenant_id,
                    session_id=session_id_value,
                    user_id=user_id_value,
                    message=ChatMessage(
                        role=role,
                        content=hit.document_content,
                        entry_id=entry_id,
                    ),
                    score=score,
                )
            )
        return hits
