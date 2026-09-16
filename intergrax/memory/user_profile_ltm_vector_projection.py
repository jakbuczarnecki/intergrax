# © Artur Czarnecki. All rights reserved.

"""LTM vector derived projection for user profile memory (MEM-ENT-2)."""

from __future__ import annotations

import json
from collections.abc import Sequence

from intergrax.knowledge.contracts import KnowledgeDocument
from intergrax.memory.contracts.memory_lifecycle import (
    MemoryProjectionReconciliationDisposition,
    MemoryProjectionReconciliationResult,
    UserProfileMemoryProjectionContext,
    UserProfileMemoryReconciliationContext,
)
from intergrax.memory.memory_temporal import filter_active_memory_entries
from intergrax.memory.memory_vector_namespace import LTM_INDEX_DOMAIN, resolve_memory_index_collection
from intergrax.memory.user_profile_memory import MemoryKind, UserProfileMemoryEntry
from intergrax.rag.embedding.embedding_manager import EmbeddingManager
from intergrax.rag.vectorstore.contracts.native_vectorstore import MetadataFilter, VectorStoreRecord, VectorStoreScope
from intergrax.rag.vectorstore.vectorstore_manager import VectorstoreManager

__all__ = ["UserProfileLtmVectorProjection", "LTM_VECTOR_PROJECTION_ID"]

LTM_VECTOR_PROJECTION_ID = "user_profile_ltm_vector"


class UserProfileLtmVectorProjection:
    def __init__(
        self,
        *,
        embedding_manager: EmbeddingManager,
        vectorstore_manager: VectorstoreManager,
        tenant_id: str,
        vector_index_namespace: str | None,
        workspace_id: str | None,
    ) -> None:
        self._embedding_manager = embedding_manager
        self._vectorstore_manager = vectorstore_manager
        self._tenant_id = tenant_id
        self._vector_index_namespace = vector_index_namespace
        self._workspace_id = workspace_id
        self._ltm_collection_name = resolve_memory_index_collection(
            vector_index_namespace=vector_index_namespace,
            tenant_id=tenant_id,
            domain=LTM_INDEX_DOMAIN,
        )

    @property
    def projection_id(self) -> str:
        return LTM_VECTOR_PROJECTION_ID

    def _vector_scope(self) -> VectorStoreScope:
        return VectorStoreScope(
            tenant_id=self._tenant_id,
            namespace=self._vector_index_namespace,
            workspace_id=self._workspace_id,
        )

    async def upsert_memory_entry(
        self,
        context: UserProfileMemoryProjectionContext,
        entry: UserProfileMemoryEntry,
    ) -> None:
        user_id = context.user_id
        if entry.deleted:
            return
        text = (entry.content or "").strip()
        if not text:
            return
        meta = dict(entry.metadata or {})
        kind_value = entry.kind.value if isinstance(entry.kind, MemoryKind) else str(entry.kind)
        meta.update(
            {
                "user_id": user_id,
                "entry_id": entry.entry_id,
                "memory_id": entry.memory_id,
                "revision": entry.revision,
                "kind": kind_value,
                "deleted": 0,
                "index_domain": LTM_INDEX_DOMAIN,
                "collection_name": self._ltm_collection_name,
            }
        )
        meta = self._sanitize_vectorstore_metadata(meta)
        scope = self._vector_scope()
        doc = KnowledgeDocument.model_validate(
            {
                "schema_version": 1,
                "identity": {
                    "document_id": entry.entry_id,
                    "root_document_id": entry.entry_id,
                },
                "scope": {
                    "tenant_id": scope.tenant_id,
                    "namespace": scope.namespace,
                    "workspace_id": scope.workspace_id,
                },
                "content": text,
                "metadata": meta,
                "provenance": {
                    "source_kind": "user_profile_memory",
                    "source_id": entry.entry_id,
                    "source_parent_id": user_id,
                },
            }
        )
        emb = self._embedding_manager.embed_texts([text])
        self._vectorstore_manager.add_records(
            [
                VectorStoreRecord(
                    document=doc,
                    embedding=emb[0],
                    vector_id=entry.entry_id,
                )
            ],
            scope=scope,
        )

    async def delete_memory_entries(
        self,
        context: UserProfileMemoryProjectionContext,
        entry_ids: Sequence[str],
    ) -> None:
        _ = context
        ids = [entry_id for entry_id in entry_ids if entry_id]
        if not ids:
            return
        self._vectorstore_manager.delete(ids, scope=self._vector_scope())

    async def reconcile(
        self,
        context: UserProfileMemoryReconciliationContext,
    ) -> MemoryProjectionReconciliationResult:
        indexed_ids = self._indexed_entry_ids_for_user(context.user_id)
        expected_ids = set(context.authoritative_active_entry_ids)
        orphan_ids = indexed_ids - expected_ids
        missing_ids = expected_ids - indexed_ids
        changed = False
        if orphan_ids:
            await self.delete_memory_entries(
                UserProfileMemoryProjectionContext(identity=context.identity),
                sorted(orphan_ids),
            )
            changed = True
        if context.profile is not None:
            active_entries = filter_active_memory_entries(context.profile.memory_entries)
            for entry in active_entries:
                if entry.entry_id in missing_ids:
                    await self.upsert_memory_entry(
                        UserProfileMemoryProjectionContext(identity=context.identity),
                        entry,
                    )
                    changed = True
        disposition = (
            MemoryProjectionReconciliationDisposition.REPAIRED
            if changed
            else MemoryProjectionReconciliationDisposition.CONSISTENT
        )
        return MemoryProjectionReconciliationResult(
            projection_id=self.projection_id,
            disposition=disposition,
        )

    def _indexed_entry_ids_for_user(self, user_id: str) -> set[str]:
        scope = self._vector_scope()
        metadata_filter = MetadataFilter(
            conditions={
                "user_id": user_id,
                "index_domain": LTM_INDEX_DOMAIN,
                "collection_name": self._ltm_collection_name,
            }
        )
        vector_ids = self._vectorstore_manager.list_vector_ids_by_metadata(
            scope=scope,
            metadata_filter=metadata_filter,
            limit=10_000,
        )
        return set(vector_ids)

    def _sanitize_vectorstore_metadata(self, meta: dict[str, object]) -> dict[str, object]:
        out: dict[str, object] = {}
        for key, value in meta.items():
            if value is None or isinstance(value, (str, int, float, bool)):
                out[key] = value
                continue
            if isinstance(value, (list, tuple)):
                out[key] = ",".join(str(item) for item in value)
                continue
            if isinstance(value, dict):
                out[key] = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
                continue
            out[key] = str(value)
        return out
