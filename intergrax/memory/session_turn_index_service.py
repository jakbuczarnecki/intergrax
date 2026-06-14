# © Artur Czarnecki. All rights reserved.

"""Default episodic session turn vector index (Phase MEM-VEC-2.1–2.2)."""

from __future__ import annotations

import json
from typing import Any, Sequence

from langchain_core.documents import Document

from intergrax.llm.messages import ChatMessage
from intergrax.memory.contracts.session_turn_index import SessionTurnIndexStore
from intergrax.rag.embedding.contracts.base_embedding_manager import BaseEmbeddingManager
from intergrax.rag.vectorstore.contracts.base_vectorstore_manager import BaseVectorstoreManager

EPISODIC_INDEX_DOMAIN = "episodic"
LTM_INDEX_DOMAIN = "ltm"


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
    ) -> None:
        self._embedding_manager = embedding_manager
        self._vectorstore_manager = vectorstore_manager
        self._index_roles = tuple(index_roles)

    async def upsert_turn(
        self,
        *,
        tenant_id: str,
        session_id: str,
        user_id: str | None,
        message: ChatMessage,
    ) -> None:
        if message.deleted:
            await self.tombstone_turn(message.entry_id)
            return
        if message.role not in self._index_roles:
            return
        text = (message.content or "").strip()
        if not text:
            return
        meta = _sanitize_metadata(
            {
                "tenant_id": tenant_id,
                "session_id": session_id,
                "user_id": user_id or "",
                "entry_id": message.entry_id,
                "role": message.role,
                "deleted": 0,
                "index_domain": EPISODIC_INDEX_DOMAIN,
            }
        )
        doc = Document(page_content=text, metadata=meta)
        embeddings = self._embedding_manager.embed_texts([text])
        self._vectorstore_manager.add_documents(
            documents=[doc],
            embeddings=embeddings,
            ids=[message.entry_id],
        )

    async def tombstone_turn(self, entry_id: str) -> None:
        if not entry_id:
            return
        self._vectorstore_manager.delete([entry_id])

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
    ) -> list[dict[str, Any]]:
        q = (query or "").strip()
        if not q:
            return []
        where: dict[str, Any] = {
            "tenant_id": tenant_id,
            "deleted": 0,
            "index_domain": EPISODIC_INDEX_DOMAIN,
        }
        if not include_cross_session and session_id:
            where["session_id"] = session_id
        elif include_cross_session and user_id:
            where["user_id"] = user_id

        q_emb = self._embedding_manager.embed_texts([q])
        embedding = q_emb[0].tolist() if hasattr(q_emb[0], "tolist") else list(q_emb[0])
        from intergrax.rag.vectorstore.contracts.vector_store import MetadataFilter

        raw_hits = self._vectorstore_manager.query(
            embedding,
            top_k=top_k,
            metadata_filter=MetadataFilter(conditions=where),
        )

        hits: list[dict[str, Any]] = []
        for hit in raw_hits:
            score = float(getattr(hit, "similarity_score", None) or getattr(hit, "score", 0.0) or 0.0)
            if score_threshold is not None and score < score_threshold:
                continue
            meta = dict(hit.metadata or {})
            text = str(getattr(hit, "content", None) or getattr(hit, "text", "") or "")
            hits.append(
                {
                    "text": text,
                    "score": score,
                    "message_id": str(hit.id or meta.get("entry_id") or ""),
                    "session_id": str(meta.get("session_id") or ""),
                    "role": str(meta.get("role") or ""),
                }
            )
        return hits
