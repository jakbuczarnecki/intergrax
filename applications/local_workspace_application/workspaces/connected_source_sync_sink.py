# © Artur Czarnecki. All rights reserved.

"""Durable sink that materializes vendor knowledge batches into LKW documents."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol

from local_workspace_application.workspaces.connected_source_delivery import (
    ConnectedSourceDeliveryApplyResult,
    begin_delivery_receipt,
    complete_delivery_receipt,
    delivery_receipt_completed,
    mark_delivery_prepared,
)
from local_workspace_application.workspaces.connected_source_manifest import (
    ConnectedSourceMaterializationManifestEntryV1,
    ConnectedSourceMaterializationManifestRepository,
    ConnectedSourceMaterializationManifestV1,
    ManifestCommitStatus,
    materialization_manifest_payload_fingerprint,
    publication_fence_token_fingerprint,
)
from local_workspace_application.workspaces.connected_source_materializer import (
    ConnectedSourceContentMaterializerRegistry,
    default_connected_source_materializer_registry,
)
from local_workspace_application.workspaces.connected_source_models import (
    ConnectedSourceDeliveryPublicationState,
    ConnectedSourceDeliveryStatus,
    ConnectedSourceSyncSinkError,
)
from local_workspace_application.workspaces.connected_source_source_projection import (
    ConnectedSourceOriginValidationError,
    validate_connected_source_durable_origin,
)
from local_workspace_application.workspaces.document_indexing import (
    WorkspaceDocumentIndexingError,
    WorkspaceDocumentIndexingService,
)
from local_workspace_application.workspaces.knowledge_configuration_models import (
    IndexedSourceAudienceEligibilityV1,
    WorkspaceIndexedSourceBindingStatusV1,
)
from local_workspace_application.workspaces.knowledge_configuration_service import (
    WorkspaceKnowledgeConfigurationService,
    is_workspace_source_product_visible,
)
from local_workspace_application.workspaces.materialization_visibility import (
    KnowledgeMaterializationActivePointerV1,
    KnowledgeMaterializationOwnershipV1,
)
from local_workspace_application.workspaces.models import (
    WorkspaceOperationStatus,
    WorkspaceOperationType,
    WorkspaceSourceType,
)
from local_workspace_application.workspaces.repository import ManagedWorkspaceRepository

from intergrax.runtime.vendor_knowledge.bindings import (
    KnowledgeSourceBinding,
    KnowledgeSourceBindingStatus,
    to_source_ref,
)
from intergrax.runtime.vendor_knowledge.models import (
    KnowledgeChangeKind,
    KnowledgeContentMode,
)
from intergrax.runtime.vendor_knowledge.sync_models import (
    KnowledgeSyncBatch,
    KnowledgeSyncSinkReceipt,
    KnowledgeSyncSinkReceiptStatus,
)
from intergrax.runtime.vendor_knowledge.sync_publication_fence import (
    KnowledgeSyncPublicationFencePort,
)

_ALLOWED_CHANGE_KINDS = frozenset(
    {
        KnowledgeChangeKind.UPSERT,
        KnowledgeChangeKind.METADATA_CHANGED,
    }
)


class TenantKnowledgeSourceBindingPort(Protocol):
    def get_binding(
        self,
        *,
        tenant_id: str,
        binding_id: str,
    ) -> KnowledgeSourceBinding | None:
        ...


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
        configuration_reader: WorkspaceKnowledgeConfigurationService,
        tenant_binding_port: TenantKnowledgeSourceBindingPort,
        context: ConnectedSourceSyncSinkContext,
        materializer_registry: ConnectedSourceContentMaterializerRegistry | None = None,
        publication_fence_port: KnowledgeSyncPublicationFencePort | None = None,
    ) -> None:
        self._repository = repository
        self._indexing_service = indexing_service
        self._configuration_reader = configuration_reader
        self._tenant_binding_port = tenant_binding_port
        self._context = context
        self._materializers = materializer_registry or default_connected_source_materializer_registry()
        self._publication_fence_port = publication_fence_port
        self._manifest_repository = ConnectedSourceMaterializationManifestRepository(
            repository.document_store,
            publication_authority=publication_fence_port,
        )
        self._publication_validator = None

    def set_publication_validator(self, validator) -> None:
        """Install the coordinator guard that also checks the active source lease."""
        self._publication_validator = validator

    async def apply_batch(self, *, batch: KnowledgeSyncBatch) -> ConnectedSourceDeliveryApplyResult:
        try:
            return await self._apply_batch(batch=batch)
        except ConnectedSourceSyncSinkError:
            raise
        except Exception as exc:
            raise ConnectedSourceSyncSinkError(
                f"connected_source_sink_internal_error:{exc.__class__.__name__}"
            ) from exc

    def inspect_receipt(
        self,
        *,
        tenant_id: str,
        binding_id: str,
        delivery_id: str,
        prepared_batch_payload_fingerprint: str,
    ) -> KnowledgeSyncSinkReceipt:
        if (
            tenant_id != self._context.tenant_id
            or binding_id != self._context.knowledge_source_binding_ref
        ):
            return KnowledgeSyncSinkReceipt(
                status=KnowledgeSyncSinkReceiptStatus.CONFLICT,
                delivery_id=delivery_id,
                prepared_batch_payload_fingerprint=prepared_batch_payload_fingerprint,
            )
        receipt = delivery_receipt_completed(
            repository=self._repository,
            tenant_id=self._context.tenant_id,
            workspace_id=self._context.workspace_id,
            source_id=self._context.source_id,
            delivery_id=delivery_id,
            indexed_source_binding_id=self._context.indexed_source_binding_id,
            knowledge_source_binding_ref=self._context.knowledge_source_binding_ref,
            binding_configuration_version=self._binding_configuration_version(),
            operation_id=self._context.operation_id,
        )
        if receipt is None:
            existing = self._repository.get_connected_source_delivery_receipt(
                tenant_id=self._context.tenant_id,
                workspace_id=self._context.workspace_id,
                source_id=self._context.source_id,
                delivery_id=delivery_id,
            )
            if existing is None:
                return KnowledgeSyncSinkReceipt(status=KnowledgeSyncSinkReceiptStatus.ABSENT)
            return KnowledgeSyncSinkReceipt(
                status=KnowledgeSyncSinkReceiptStatus.UNKNOWN,
                delivery_id=delivery_id,
                prepared_batch_payload_fingerprint=prepared_batch_payload_fingerprint,
            )
        return KnowledgeSyncSinkReceipt(
            status=KnowledgeSyncSinkReceiptStatus.APPLIED,
            delivery_id=receipt.delivery_id,
            prepared_batch_payload_fingerprint=prepared_batch_payload_fingerprint,
        )

    async def _apply_batch(self, *, batch: KnowledgeSyncBatch) -> ConnectedSourceDeliveryApplyResult:
        if (
            batch.tenant_id != self._context.tenant_id
            or batch.binding_id != self._context.knowledge_source_binding_ref
        ):
            self._validate_authoritative_state(batch)
        payload_fingerprint = materialization_manifest_payload_fingerprint(batch)
        completed = delivery_receipt_completed(
            repository=self._repository,
            tenant_id=self._context.tenant_id,
            workspace_id=self._context.workspace_id,
            source_id=self._context.source_id,
            delivery_id=batch.delivery_id,
            indexed_source_binding_id=self._context.indexed_source_binding_id,
            knowledge_source_binding_ref=self._context.knowledge_source_binding_ref,
            binding_configuration_version=batch.binding_configuration_version,
            operation_id=self._context.operation_id,
            payload_fingerprint=payload_fingerprint,
        )
        if completed is not None:
            if completed.payload_fingerprint is not None:
                current_manifest = self._manifest_repository.get_current(
                    tenant_id=self._context.tenant_id,
                    workspace_id=self._context.workspace_id,
                    source_id=self._context.source_id,
                    indexed_source_binding_id=self._context.indexed_source_binding_id,
                    knowledge_source_binding_ref=self._context.knowledge_source_binding_ref,
                )
                if current_manifest is None:
                    raise ConnectedSourceSyncSinkError(
                        "connected_source_materialization_manifest_missing"
                    )
            self._activate_delivery_documents(delivery_id=completed.delivery_id)
            return ConnectedSourceDeliveryApplyResult(
                documents_indexed=completed.documents_indexed,
                documents_unchanged=completed.documents_unchanged,
                items_processed=completed.documents_indexed + completed.documents_unchanged,
                items_failed=completed.items_failed,
                replayed=True,
            )

        self._validate_authoritative_state(batch)
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
            payload_fingerprint=payload_fingerprint,
        )
        if (
            receipt.status is ConnectedSourceDeliveryStatus.COMPLETED
            and receipt.completed_at is not None
            and receipt.items_failed == 0
        ):
            return ConnectedSourceDeliveryApplyResult(
                documents_indexed=receipt.documents_indexed,
                documents_unchanged=receipt.documents_unchanged,
                items_processed=receipt.documents_indexed + receipt.documents_unchanged,
                items_failed=receipt.items_failed,
                replayed=True,
            )
        if receipt.materialization_sequence is None:
            raise ConnectedSourceSyncSinkError(
                "connected_source_delivery_sequence_missing"
            )
        if receipt.publication_state is ConnectedSourceDeliveryPublicationState.PREPARED:
            try:
                committed_manifest = self._manifest_repository.get_committed_for_delivery(
                    tenant_id=self._context.tenant_id,
                    workspace_id=self._context.workspace_id,
                    source_id=self._context.source_id,
                    indexed_source_binding_id=self._context.indexed_source_binding_id,
                    delivery_id=receipt.delivery_id,
                )
            except Exception as exc:
                raise ConnectedSourceSyncSinkError(
                    "connected_source_materialization_manifest_recovery_failed"
                ) from exc
            if (
                committed_manifest is not None
                and committed_manifest.materialization_sequence
                == receipt.materialization_sequence
                and committed_manifest.payload_fingerprint == receipt.payload_fingerprint
            ):
                recovered = complete_delivery_receipt(
                    repository=self._repository,
                    receipt=receipt,
                    documents_indexed=len(committed_manifest.document_entries),
                    documents_unchanged=0,
                    items_processed=len(committed_manifest.document_entries),
                    items_failed=0,
                )
                self._activate_delivery_documents(delivery_id=recovered.delivery_id)
                return ConnectedSourceDeliveryApplyResult(
                    documents_indexed=recovered.documents_indexed,
                    documents_unchanged=recovered.documents_unchanged,
                    items_processed=(
                        recovered.documents_indexed + recovered.documents_unchanged
                    ),
                    items_failed=recovered.items_failed,
                    replayed=True,
                )
            prepared_manifest = self._manifest_repository.get_prepared_for_delivery(
                tenant_id=self._context.tenant_id,
                workspace_id=self._context.workspace_id,
                source_id=self._context.source_id,
                indexed_source_binding_id=self._context.indexed_source_binding_id,
                delivery_id=receipt.delivery_id,
            )
            if (
                prepared_manifest is not None
                and prepared_manifest.materialization_sequence
                == receipt.materialization_sequence
                and prepared_manifest.payload_fingerprint == receipt.payload_fingerprint
                and batch.publication_fence is not None
                and batch.publication_permit is not None
            ):
                try:
                    status = self._manifest_repository.commit(
                        prepared_manifest,
                        expected_fence=batch.publication_fence,
                        publication_permit=batch.publication_permit,
                        publication_authority=self._publication_fence_port,
                    )
                except Exception as exc:
                    raise ConnectedSourceSyncSinkError(
                        "connected_source_materialization_manifest_recovery_failed"
                    ) from exc
                if status is ManifestCommitStatus.STALE:
                    raise ConnectedSourceSyncSinkError(
                        "connected_source_materialization_manifest_stale"
                    )
                recovered = complete_delivery_receipt(
                    repository=self._repository,
                    receipt=receipt,
                    documents_indexed=len(prepared_manifest.document_entries),
                    documents_unchanged=0,
                    items_processed=len(prepared_manifest.document_entries),
                    items_failed=0,
                )
                self._activate_delivery_documents(delivery_id=recovered.delivery_id)
                return ConnectedSourceDeliveryApplyResult(
                    documents_indexed=recovered.documents_indexed,
                    documents_unchanged=recovered.documents_unchanged,
                    items_processed=(
                        recovered.documents_indexed + recovered.documents_unchanged
                    ),
                    items_failed=recovered.items_failed,
                    replayed=True,
                )
            if prepared_manifest is not None:
                raise ConnectedSourceSyncSinkError(
                    "connected_source_materialization_manifest_recovery_conflict"
                )

        documents_indexed = 0
        documents_unchanged = 0
        items_processed = 0
        materialized_documents: list[
            tuple[KnowledgeMaterializationOwnershipV1, str, str]
        ] = []

        for envelope in batch.envelopes:
            if envelope.change_kind not in _ALLOWED_CHANGE_KINDS:
                raise ConnectedSourceSyncSinkError("connected_source_change_kind_rejected")
            if envelope.content is None:
                raise ConnectedSourceSyncSinkError("connected_source_content_missing")
            if envelope.content.mode is not KnowledgeContentMode.STRUCTURED_RECORD:
                raise ConnectedSourceSyncSinkError("connected_source_content_mode_invalid")
            record = envelope.content.structured_record
            if not isinstance(record, dict):
                raise ConnectedSourceSyncSinkError("connected_source_structured_record_invalid")
            schema_name = record.get("schema")
            if not isinstance(schema_name, str) or not schema_name:
                raise ConnectedSourceSyncSinkError("connected_source_schema_unsupported")
            materializer = self._materializers.resolve(schema_name)
            materialized = materializer.materialize(
                source_id=self._context.source_id,
                remote_id=envelope.remote_id,
                content=envelope.content,
            )
            result = await self._index_materialized_document(
                materialized,
                delivery_id=batch.delivery_id,
                remote_id=envelope.remote_id,
                materialization_sequence=receipt.materialization_sequence,
            )
            materialized_documents.append(
                (
                    KnowledgeMaterializationOwnershipV1.connected(
                        tenant_id=self._context.tenant_id,
                        workspace_id=self._context.workspace_id,
                        source_id=self._context.source_id,
                        indexed_source_binding_id=self._context.indexed_source_binding_id,
                        knowledge_source_binding_ref=self._context.knowledge_source_binding_ref,
                        delivery_id=batch.delivery_id,
                        materialization_sequence=receipt.materialization_sequence,
                        remote_id=envelope.remote_id,
                    ),
                    result.document_id,
                    materialized.content_hash,
                )
            )
            items_processed += 1
            if result.indexed:
                documents_indexed += 1
            elif result.unchanged:
                documents_unchanged += 1
            else:
                raise ConnectedSourceSyncSinkError("connected_source_indexing_failed")

        mark_delivery_prepared(repository=self._repository, receipt=receipt)
        manifest = self._build_manifest(
            batch=batch,
            receipt=receipt,
            materialized_documents=materialized_documents,
            payload_fingerprint=payload_fingerprint,
        )
        expected_fence = batch.publication_fence
        publication_permit = batch.publication_permit
        if expected_fence is None or publication_permit is None:
            raise ConnectedSourceSyncSinkError(
                "connected_source_publication_permit_required"
            )
        try:
            manifest_status = self._manifest_repository.commit(
                manifest,
                expected_fence=expected_fence,
                publication_permit=publication_permit,
                publication_authority=self._publication_fence_port,
            )
        except ConnectedSourceSyncSinkError:
            raise
        except Exception as exc:
            raise ConnectedSourceSyncSinkError(
                "connected_source_materialization_manifest_commit_failed"
            ) from exc
        if manifest_status is ManifestCommitStatus.STALE:
            raise ConnectedSourceSyncSinkError(
                "connected_source_materialization_manifest_stale"
            )
        completed_receipt = complete_delivery_receipt(
            repository=self._repository,
            receipt=receipt,
            documents_indexed=documents_indexed,
            documents_unchanged=documents_unchanged,
            items_processed=items_processed,
            items_failed=0,
        )
        for ownership, document_id, _content_hash in materialized_documents:
            try:
                self._activate_materialization(
                    ownership=ownership,
                    document_id=document_id,
                    committed_at=completed_receipt.completed_at or completed_receipt.created_at,
                )
            except (
                ConnectedSourceSyncSinkError,
                AssertionError,
                TypeError,
                ValueError,
                AttributeError,
            ):
                # Active pointers are a rebuildable accelerator, never visibility authority.
                pass
        return ConnectedSourceDeliveryApplyResult(
            documents_indexed=completed_receipt.documents_indexed,
            documents_unchanged=completed_receipt.documents_unchanged,
            items_processed=items_processed,
            items_failed=0,
            replayed=False,
        )

    def _build_manifest(
        self,
        *,
        batch: KnowledgeSyncBatch,
        receipt,
        materialized_documents: list[tuple[KnowledgeMaterializationOwnershipV1, str, str]],
        payload_fingerprint: str,
    ) -> ConnectedSourceMaterializationManifestV1:
        if batch.publication_fence is None or batch.publication_permit is None:
            raise ConnectedSourceSyncSinkError(
                "connected_source_publication_permit_required"
            )
        if receipt.materialization_sequence is None:
            raise ConnectedSourceSyncSinkError(
                "connected_source_delivery_sequence_missing"
            )
        if (
            receipt.tenant_id != self._context.tenant_id
            or receipt.workspace_id != self._context.workspace_id
            or receipt.source_id != self._context.source_id
            or receipt.indexed_source_binding_id
            != self._context.indexed_source_binding_id
            or receipt.knowledge_source_binding_ref
            != self._context.knowledge_source_binding_ref
            or receipt.delivery_id != batch.delivery_id
            or receipt.binding_configuration_version
            != batch.binding_configuration_version
            or receipt.payload_fingerprint != payload_fingerprint
        ):
            raise ConnectedSourceSyncSinkError(
                "connected_source_manifest_receipt_identity_conflict"
            )
        entries = tuple(
            sorted(
                self._manifest_entries(materialized_documents),
                key=lambda entry: (entry.remote_id, entry.document_id),
            )
        )
        return ConnectedSourceMaterializationManifestV1(
            tenant_id=self._context.tenant_id,
            workspace_id=self._context.workspace_id,
            source_id=self._context.source_id,
            indexed_source_binding_id=self._context.indexed_source_binding_id,
            knowledge_source_binding_ref=self._context.knowledge_source_binding_ref,
            delivery_id=batch.delivery_id,
            materialization_sequence=receipt.materialization_sequence,
            binding_configuration_version=batch.binding_configuration_version,
            publication_fence_revision=batch.publication_fence.lifecycle_revision,
            publication_fence_token_fingerprint=publication_fence_token_fingerprint(
                batch.publication_fence.lifecycle_token
            ),
            document_entries=entries,
            payload_fingerprint=payload_fingerprint,
            committed_at=receipt.created_at,
        )

    @staticmethod
    def _manifest_entries(
        materialized_documents: list[tuple[KnowledgeMaterializationOwnershipV1, str, str]],
    ) -> list[ConnectedSourceMaterializationManifestEntryV1]:
        entries: list[ConnectedSourceMaterializationManifestEntryV1] = []
        for ownership, document_id, content_hash in materialized_documents:
            assert ownership.remote_id is not None
            assert ownership.materialization_generation is not None
            entries.append(
                ConnectedSourceMaterializationManifestEntryV1(
                    remote_id=ownership.remote_id,
                    document_id=document_id,
                    materialization_generation=ownership.materialization_generation,
                    content_hash=content_hash,
                )
            )
        return entries

    def _validate_publication_at_commit(self, expected_fence, publication_permit) -> None:
        if expected_fence is None or publication_permit is None:
            raise ConnectedSourceSyncSinkError(
                "connected_source_publication_permit_required"
            )
        if self._publication_validator is not None:
            try:
                self._publication_validator(expected_fence, publication_permit)
            except Exception as exc:
                raise ConnectedSourceSyncSinkError(
                    "connected_source_publication_permit_invalid"
                ) from exc
            return
        if self._publication_fence_port is None:
            raise ConnectedSourceSyncSinkError(
                "connected_source_publication_fence_not_configured"
            )
        try:
            if (
                publication_permit.tenant_id != expected_fence.tenant_id
                or publication_permit.binding_id != expected_fence.binding_id
                or publication_permit.lifecycle_revision
                != expected_fence.lifecycle_revision
                or publication_permit.lifecycle_token != expected_fence.lifecycle_token
            ):
                raise ConnectedSourceSyncSinkError(
                    "connected_source_publication_permit_invalid"
                )
            if not self._publication_fence_port.is_current_publication_permit(
                permit=publication_permit
            ):
                raise ConnectedSourceSyncSinkError(
                    "connected_source_publication_permit_invalid"
                )
        except ConnectedSourceSyncSinkError:
            raise
        except Exception as exc:
            raise ConnectedSourceSyncSinkError(
                "connected_source_publication_permit_invalid"
            ) from exc

    def _binding_configuration_version(self) -> int:
        binding = self._tenant_binding_port.get_binding(
            tenant_id=self._context.tenant_id,
            binding_id=self._context.knowledge_source_binding_ref,
        )
        if binding is None:
            raise ConnectedSourceSyncSinkError("connected_source_tenant_binding_not_found")
        return binding.configuration_version

    def _validate_authoritative_state(self, batch: KnowledgeSyncBatch) -> None:
        if batch.tenant_id != self._context.tenant_id:
            raise ConnectedSourceSyncSinkError("connected_source_batch_tenant_mismatch")
        if batch.binding_id != self._context.knowledge_source_binding_ref:
            raise ConnectedSourceSyncSinkError("connected_source_batch_binding_mismatch")

        tenant_binding = self._tenant_binding_port.get_binding(
            tenant_id=self._context.tenant_id,
            binding_id=self._context.knowledge_source_binding_ref,
        )
        if tenant_binding is None:
            raise ConnectedSourceSyncSinkError("connected_source_tenant_binding_not_found")
        if tenant_binding.status is not KnowledgeSourceBindingStatus.ACTIVE:
            raise ConnectedSourceSyncSinkError("connected_source_tenant_binding_inactive")
        if batch.source.tenant_id != tenant_binding.tenant_id:
            raise ConnectedSourceSyncSinkError("connected_source_batch_source_mismatch")
        ref = to_source_ref(tenant_binding)
        if batch.source.provider_id != ref.provider_id:
            raise ConnectedSourceSyncSinkError("connected_source_batch_source_mismatch")
        if batch.source.integration_kind != ref.integration_kind:
            raise ConnectedSourceSyncSinkError("connected_source_batch_source_mismatch")
        if batch.source.source_kind != ref.source_kind:
            raise ConnectedSourceSyncSinkError("connected_source_batch_source_mismatch")
        if batch.source.connection_ref != ref.connection_ref:
            raise ConnectedSourceSyncSinkError("connected_source_batch_source_mismatch")
        if batch.source.scope.remote_scope_id != ref.scope.remote_scope_id:
            raise ConnectedSourceSyncSinkError("connected_source_batch_source_mismatch")
        if batch.source.scope.remote_scope_type != ref.scope.remote_scope_type:
            raise ConnectedSourceSyncSinkError("connected_source_batch_source_mismatch")
        if batch.source.scope.parameters != ref.scope.parameters:
            raise ConnectedSourceSyncSinkError("connected_source_batch_source_mismatch")
        if batch.binding_configuration_version != tenant_binding.configuration_version:
            raise ConnectedSourceSyncSinkError("connected_source_batch_version_mismatch")

        configuration = self._configuration_reader.get_configuration(
            tenant_id=self._context.tenant_id,
            workspace_id=self._context.workspace_id,
        )
        if configuration is None:
            raise ConnectedSourceSyncSinkError("connected_source_configuration_not_found")
        indexed_binding = None
        for item in configuration.indexed_sources:
            if item.indexed_source_binding_id == self._context.indexed_source_binding_id:
                indexed_binding = item
                break
        if indexed_binding is None:
            raise ConnectedSourceSyncSinkError("connected_source_indexed_binding_not_found")
        if indexed_binding.status is not WorkspaceIndexedSourceBindingStatusV1.ACTIVE:
            raise ConnectedSourceSyncSinkError("connected_source_indexed_binding_inactive")
        if indexed_binding.knowledge_source_binding_ref != self._context.knowledge_source_binding_ref:
            raise ConnectedSourceSyncSinkError("connected_source_indexed_binding_ref_mismatch")
        if indexed_binding.source_id != self._context.source_id:
            raise ConnectedSourceSyncSinkError("connected_source_indexed_binding_source_mismatch")
        if indexed_binding.audience_eligibility is not IndexedSourceAudienceEligibilityV1.PERSONAL_ONLY:
            raise ConnectedSourceSyncSinkError("connected_source_audience_eligibility_rejected")

        source = self._repository.get_source(
            tenant_id=self._context.tenant_id,
            workspace_id=self._context.workspace_id,
            source_id=self._context.source_id,
        )
        if source is None:
            raise ConnectedSourceSyncSinkError("connected_source_workspace_source_not_found")
        if source.source_type is not WorkspaceSourceType.CONNECTED_SOURCE:
            raise ConnectedSourceSyncSinkError("connected_source_workspace_source_type_mismatch")
        if not is_workspace_source_product_visible(
            source,
            committed_configuration_revision=configuration.configuration_revision,
        ):
            raise ConnectedSourceSyncSinkError("connected_source_workspace_source_uncommitted")
        try:
            validate_connected_source_durable_origin(
                repository=self._repository,
                tenant_id=self._context.tenant_id,
                workspace_id=self._context.workspace_id,
                source_id=self._context.source_id,
                binding=indexed_binding,
                committed_configuration_revision=configuration.configuration_revision,
            )
        except ConnectedSourceOriginValidationError:
            raise ConnectedSourceSyncSinkError(
                "connected_source_workspace_source_uncommitted"
            ) from None

        operation = self._repository.get_operation(
            tenant_id=self._context.tenant_id,
            operation_id=self._context.operation_id,
        )
        if operation is None:
            raise ConnectedSourceSyncSinkError("connected_source_operation_not_found")
        if operation.workspace_id != self._context.workspace_id:
            raise ConnectedSourceSyncSinkError("connected_source_operation_workspace_mismatch")
        if operation.source_id != self._context.source_id:
            raise ConnectedSourceSyncSinkError("connected_source_operation_source_mismatch")
        if operation.operation_type is not WorkspaceOperationType.SOURCE_SYNC:
            raise ConnectedSourceSyncSinkError("connected_source_operation_type_mismatch")
        if operation.status is not WorkspaceOperationStatus.RUNNING:
            raise ConnectedSourceSyncSinkError("connected_source_operation_not_running")

    async def _index_materialized_document(
        self,
        materialized,
        *,
        delivery_id: str,
        remote_id: str,
        materialization_sequence: int | None,
    ):
        allowed_roots = tuple(
            item.strip()
            for item in os.environ.get("INTERGRAX_ALLOWED_READ_ROOTS", "").split(os.pathsep)
            if item.strip()
        )
        temp_dir = allowed_roots[0] if allowed_roots else None
        fd, temp_name = tempfile.mkstemp(
            prefix="lkw-connected-source-",
            suffix=".md",
            dir=temp_dir,
        )
        temp_path = Path(temp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(materialized.markdown)
            return await self._indexing_service.index_connected_source_one(
                tenant_id=self._context.tenant_id,
                workspace_id=self._context.workspace_id,
                source_id=self._context.source_id,
                operation_id=self._context.operation_id,
                physical_path=temp_path,
                logical_source_path=materialized.logical_source_path,
                safe_file_name=materialized.safe_file_name,
                content_hash=materialized.content_hash,
                materialization_ownership=KnowledgeMaterializationOwnershipV1.connected(
                    tenant_id=self._context.tenant_id,
                    workspace_id=self._context.workspace_id,
                    source_id=self._context.source_id,
                    indexed_source_binding_id=self._context.indexed_source_binding_id,
                    knowledge_source_binding_ref=self._context.knowledge_source_binding_ref,
                    delivery_id=delivery_id,
                    materialization_sequence=materialization_sequence,
                    remote_id=remote_id,
                ),
            )
        except WorkspaceDocumentIndexingError as exc:
            raise ConnectedSourceSyncSinkError("connected_source_indexing_failed") from exc
        finally:
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)

    def _activate_delivery_documents(self, *, delivery_id: str) -> None:
        for ref in self._repository.list_document_refs(
            tenant_id=self._context.tenant_id,
            workspace_id=self._context.workspace_id,
        ):
            ownership = ref.materialization_ownership
            if ownership is None or ownership.delivery_id != delivery_id:
                continue
            try:
                self._activate_materialization(
                    ownership=ownership,
                    document_id=ref.document_id,
                    committed_at=ref.indexed_at,
                )
            except (
                ConnectedSourceSyncSinkError,
                AssertionError,
                TypeError,
                ValueError,
                AttributeError,
            ):
                # Recovery and retrieval remain correct without this projection.
                pass

    def _activate_materialization(
        self,
        *,
        ownership: KnowledgeMaterializationOwnershipV1,
        document_id: str,
        committed_at: datetime,
    ) -> None:
        assert ownership.indexed_source_binding_id is not None
        assert ownership.knowledge_source_binding_ref is not None
        assert ownership.delivery_id is not None
        assert ownership.remote_id is not None
        receipt = self._repository.get_connected_source_delivery_receipt(
            tenant_id=ownership.tenant_id,
            workspace_id=ownership.workspace_id,
            source_id=ownership.source_id,
            delivery_id=ownership.delivery_id,
        )
        materialization_sequence = (
            None if receipt is None else receipt.materialization_sequence
        )
        if (
            receipt is None
            or receipt.tenant_id != ownership.tenant_id
            or receipt.workspace_id != ownership.workspace_id
            or receipt.source_id != ownership.source_id
            or receipt.indexed_source_binding_id != ownership.indexed_source_binding_id
            or receipt.knowledge_source_binding_ref != ownership.knowledge_source_binding_ref
            or receipt.status is not ConnectedSourceDeliveryStatus.COMPLETED
            or receipt.completed_at is None
            or receipt.items_failed != 0
            or materialization_sequence is None
        ):
            raise ConnectedSourceSyncSinkError(
                "connected_source_active_pointer_receipt_invalid"
            )
        assignment = self._repository.get_connected_source_delivery_sequence_assignment(
            tenant_id=ownership.tenant_id,
            workspace_id=ownership.workspace_id,
            source_id=ownership.source_id,
            indexed_source_binding_id=ownership.indexed_source_binding_id,
            delivery_id=ownership.delivery_id,
        )
        if (
            assignment is not None
            and assignment.materialization_sequence != materialization_sequence
        ):
            raise ConnectedSourceSyncSinkError(
                "connected_source_active_pointer_receipt_invalid"
            )
        pointer = KnowledgeMaterializationActivePointerV1.for_ownership(
            ownership=ownership,
            document_id=document_id,
            materialization_revision=materialization_sequence,
            committed_at=committed_at,
        )
        for _ in range(3):
            current = self._repository.get_active_materialization_pointer(
                tenant_id=ownership.tenant_id,
                workspace_id=ownership.workspace_id,
                source_id=ownership.source_id,
                indexed_source_binding_id=ownership.indexed_source_binding_id,
                remote_id=ownership.remote_id,
            )
            if current is None:
                if self._repository.put_active_materialization_pointer_if_absent(pointer):
                    return
                continue
            if current == pointer:
                return
            if current.materialization_revision > pointer.materialization_revision:
                return
            if current.materialization_revision == pointer.materialization_revision:
                raise ConnectedSourceSyncSinkError(
                    "connected_source_active_pointer_revision_conflict"
                )
            if self._repository.replace_active_materialization_pointer(
                expected=current,
                replacement=pointer,
            ):
                return
        raise ConnectedSourceSyncSinkError("connected_source_active_pointer_conflict")
