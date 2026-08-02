# © Artur Czarnecki. All rights reserved.

"""Durable sink that materializes vendor knowledge batches into LKW documents."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

from intergrax.runtime.vendor_knowledge.models import KnowledgeChangeKind, KnowledgeContentMode
from intergrax.runtime.vendor_knowledge.sync_models import KnowledgeSyncBatch
from local_workspace_application.workspaces.connected_source_delivery import (
    begin_delivery_receipt,
    complete_delivery_receipt,
    delivery_receipt_already_applied,
)
from local_workspace_application.workspaces.connected_source_materializer import (
    ConnectedSourceContentMaterializerRegistry,
    default_connected_source_materializer_registry,
)
from local_workspace_application.workspaces.connected_source_models import ConnectedSourceSyncSinkError
from local_workspace_application.workspaces.document_indexing import (
    WorkspaceDocumentIndexingError,
    WorkspaceDocumentIndexingService,
)
from local_workspace_application.workspaces.repository import ManagedWorkspaceRepository

_ACTIVE_CHANGE_KINDS = frozenset(
    {
        KnowledgeChangeKind.UPSERT,
        KnowledgeChangeKind.METADATA_CHANGED,
        KnowledgeChangeKind.PERMISSIONS_CHANGED,
    }
)


@dataclass(frozen=True, slots=True)
class ConnectedSourceSyncSinkContext:
    tenant_id: str
    workspace_id: str
    source_id: str
    indexed_source_binding_id: str
    knowledge_source_binding_ref: str
    operation_id: str


class WorkspaceConnectedSourceKnowledgeSyncSink:
    def __init__(
        self,
        *,
        repository: ManagedWorkspaceRepository,
        indexing_service: WorkspaceDocumentIndexingService,
        context: ConnectedSourceSyncSinkContext,
        materializer_registry: ConnectedSourceContentMaterializerRegistry | None = None,
    ) -> None:
        self._repository = repository
        self._indexing_service = indexing_service
        self._context = context
        self._materializers = materializer_registry or default_connected_source_materializer_registry()

    async def apply_batch(self, *, batch: KnowledgeSyncBatch) -> None:
        self._validate_batch(batch)
        if delivery_receipt_already_applied(
            repository=self._repository,
            tenant_id=self._context.tenant_id,
            workspace_id=self._context.workspace_id,
            source_id=self._context.source_id,
            delivery_id=batch.delivery_id,
        ):
            return

        receipt = begin_delivery_receipt(
            repository=self._repository,
            tenant_id=self._context.tenant_id,
            workspace_id=self._context.workspace_id,
            source_id=self._context.source_id,
            indexed_source_binding_id=self._context.indexed_source_binding_id,
            knowledge_source_binding_ref=self._context.knowledge_source_binding_ref,
            delivery_id=batch.delivery_id,
            binding_configuration_version=batch.binding_configuration_version,
            operation_id=self._context.operation_id,
        )
        if receipt is None:
            raise ConnectedSourceSyncSinkError("connected_source_delivery_receipt_conflict")

        documents_indexed = 0
        documents_unchanged = 0
        items_failed = 0

        for envelope in batch.envelopes:
            if envelope.change_kind not in _ACTIVE_CHANGE_KINDS:
                continue
            if envelope.content is None:
                items_failed += 1
                continue
            if envelope.content.mode is not KnowledgeContentMode.STRUCTURED_RECORD:
                items_failed += 1
                continue
            record = envelope.content.structured_record
            if not isinstance(record, dict):
                items_failed += 1
                continue
            schema_name = record.get("schema")
            if not isinstance(schema_name, str) or not schema_name:
                items_failed += 1
                continue
            try:
                materializer = self._materializers.resolve(schema_name)
                materialized = materializer.materialize(
                    source_id=self._context.source_id,
                    remote_id=envelope.remote_id,
                    content=envelope.content,
                )
                result = await self._index_materialized_document(materialized)
            except (ConnectedSourceSyncSinkError, WorkspaceDocumentIndexingError):
                items_failed += 1
                continue
            if result.indexed:
                documents_indexed += 1
            elif result.unchanged:
                documents_unchanged += 1
            else:
                items_failed += 1

        complete_delivery_receipt(
            repository=self._repository,
            receipt=receipt,
            documents_indexed=documents_indexed,
            documents_unchanged=documents_unchanged,
            items_failed=items_failed,
        )

    def _validate_batch(self, batch: KnowledgeSyncBatch) -> None:
        if batch.tenant_id != self._context.tenant_id:
            raise ConnectedSourceSyncSinkError("connected_source_batch_tenant_mismatch")
        if batch.binding_id != self._context.knowledge_source_binding_ref:
            raise ConnectedSourceSyncSinkError("connected_source_batch_binding_mismatch")

    async def _index_materialized_document(self, materialized):
        fd, temp_name = tempfile.mkstemp(
            prefix="lkw-connected-source-",
            suffix=".md",
        )
        temp_path = Path(temp_name)
        with open(fd, "w", encoding="utf-8") as handle:
            handle.write(materialized.markdown)
        return await self._indexing_service.index_one(
            tenant_id=self._context.tenant_id,
            workspace_id=self._context.workspace_id,
            source_id=self._context.source_id,
            operation_id=self._context.operation_id,
            physical_path=temp_path,
            logical_source_path=materialized.logical_source_path,
            safe_file_name=materialized.safe_file_name,
            content_hash=materialized.content_hash,
        )
