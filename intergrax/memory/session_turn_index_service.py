# © Artur Czarnecki. All rights reserved.

"""Default episodic session turn vector index (Phase MEM-VEC-2.1–2.2)."""

from __future__ import annotations

import json
from typing import Any, Sequence

from intergrax.llm.messages import ChatMessage
from intergrax.memory.contracts.session_turn_index import SessionTurnIndexStore
from intergrax.memory.memory_vector_namespace import (
    EPISODIC_INDEX_DOMAIN,
    LTM_INDEX_DOMAIN,
    resolve_memory_index_collection,
)
from intergrax.rag.embedding.contracts.base_embedding_manager import BaseEmbeddingManager
from intergrax.knowledge.contracts import KnowledgeDocument
from intergrax.rag.vectorstore.contracts.base_vectorstore_manager import BaseVectorstoreManager
from intergrax.rag.vectorstore.contracts.native_vectorstore import (
    MetadataFilter,
    VectorStoreRecord,
    VectorStoreScope,
)


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


class VectorSessionTurnIndexStore(SessionTurnIndexStore):
    """Vectorstore-backed episodic index with ``index_domain=episodic`` metadata."""

    def __init__(
        self,
        *,
        embedding_manager: BaseEmbeddingManager,
        vectorstore_manager: BaseVectorstoreManager,
        index_roles: Sequence[str] = ("user", "assistant"),
        tenant_id: str = "default",
        vector_index_namespace: str | None = None,
        workspace_id: str | None = None,
    ) -> None:
        self._embedding_manager = embedding_manager
        self._vectorstore_manager = vectorstore_manager
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
        namespace: str | None = None,
        workspace_id: str | None = None,
    ) -> None:
        if message.deleted:
            await self.tombstone_turn(
                message.entry_id,
                tenant_id=tenant_id,
                namespace=namespace,
                workspace_id=workspace_id,
            )
            return
        if message.role not in self._index_roles:
            return
        text = (message.content or "").strip()
        if not text:
            return
        scope = self._scope(
            tenant_id=tenant_id,
            namespace=namespace,
            workspace_id=workspace_id,
        )
        meta = _sanitize_metadata(
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
        doc = KnowledgeDocument.model_validate(
            {
                "schema_version": 1,
                "identity": {
                    "document_id": message.entry_id,
                    "root_document_id": message.entry_id,
                },
                "scope": {
                    "tenant_id": scope.tenant_id,
                    "namespace": scope.namespace,
                    "workspace_id": scope.workspace_id,
                },
                "content": text,
                "metadata": meta,
                "provenance": {
                    "source_kind": "conversation_turn",
                    "source_id": message.entry_id,
                    "source_parent_id": session_id,
                },
            }
        )
        embeddings = self._embedding_manager.embed_texts([text])
        self._vectorstore_manager.add_records(
            [
                VectorStoreRecord(
                    document=doc,
                    embedding=embeddings[0],
                    vector_id=message.entry_id,
                )
            ],
            scope=scope,
        )

    def _scope(
        self,
        *,
        tenant_id: str | None = None,
        namespace: str | None = None,
        workspace_id: str | None = None,
    ) -> VectorStoreScope:
        return VectorStoreScope(
            tenant_id=tenant_id or self._tenant_id,
            namespace=namespace if namespace is not None else self._vector_index_namespace,
            workspace_id=workspace_id if workspace_id is not None else self._workspace_id,
        )

    async def tombstone_turn(
        self,
        entry_id: str,
        *,
        tenant_id: str | None = None,
        namespace: str | None = None,
        workspace_id: str | None = None,
    ) -> None:
        if not entry_id:
            return
        self._vectorstore_manager.delete(
            [entry_id],
            scope=self._scope(
                tenant_id=tenant_id,
                namespace=namespace,
                workspace_id=workspace_id,
            ),
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
        namespace: str | None = None,
        workspace_id: str | None = None,
    ) -> list[dict[str, Any]]:
        q = (query or "").strip()
        if not q:
            return []
        scope = self._scope(
            tenant_id=tenant_id,
            namespace=namespace,
            workspace_id=workspace_id,
        )
        where: dict[str, Any] = {
            "deleted": 0,
            "index_domain": EPISODIC_INDEX_DOMAIN,
            "collection_name": self._collection_name,
        }
        if not include_cross_session and session_id:
            where["session_id"] = session_id
        elif include_cross_session and user_id:
            where["user_id"] = user_id

        q_emb = self._embedding_manager.embed_texts([q])
        embedding = q_emb[0].tolist() if hasattr(q_emb[0], "tolist") else list(q_emb[0])
        raw_hits = self._vectorstore_manager.query(
            embedding,
            scope=scope,
            top_k=top_k,
            metadata_filter=MetadataFilter(conditions=where),
        )

        hits: list[dict[str, Any]] = []
        for hit in raw_hits:
            score = float(hit.similarity_score)
            if score_threshold is not None and score < score_threshold:
                continue
            document = hit.document
            meta = dict(document.metadata)
            hits.append(
                {
                    "text": document.content,
                    "score": score,
                    "message_id": document.identity.document_id,
                    "session_id": str(meta.get("session_id") or ""),
                    "role": str(meta.get("role") or ""),
                }
            )
        return hits
