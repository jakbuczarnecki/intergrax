# © Artur Czarnecki. All rights reserved.

"""DocumentStore-backed persistence for managed workspaces (LKW-PRODUCT-1)."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, TypeVar, cast

from local_workspace_application.workspaces.connected_source_models import (
    ConnectedSourceDeliveryReceipt,
    ConnectedSourceDeliverySequenceAssignment,
    ConnectedSourceDeliverySequenceHead,
    ConnectedSourceOperationDeliveryAccounting,
    ConnectedSourceSyncEnqueueIntent,
)
from local_workspace_application.workspaces.connected_source_purge_completion_contracts import (
    ConnectedSourceDeliveryReceiptOwnershipIndexEntryV1,
    ConnectedSourceDeliveryReceiptOwnershipIndexError,
    ConnectedSourceDeliveryReceiptOwnershipPageV1,
    ConnectedSourceRecoveryMigrationGateError,
    ConnectedSourceRecoveryMigrationGateStatusV1,
    ConnectedSourceRecoveryMigrationGateV1,
    delivery_receipt_ownership_index_partition,
    delivery_receipt_ownership_scope_prefix,
    parse_delivery_receipt_ownership_index_entry,
    parse_recovery_migration_gate,
    recovery_migration_gate_partition,
)
from local_workspace_application.workspaces.connected_source_recovery_ownership_index import (
    ConnectedSourceRecoveryOwnershipIndexEntryV1,
    ConnectedSourceRecoveryOwnershipIndexError,
    ConnectedSourceRecoveryOwnershipPageV1,
    RecoveryRecordKindV1,
    canonical_record_fingerprint,
    index_entry_for_delivery_accounting,
    index_entry_for_enqueue_intent,
    parse_recovery_ownership_index_entry,
    recovery_ownership_index_partition,
    recovery_ownership_scope_prefix,
)
from local_workspace_application.workspaces.document_ownership_index import (
    DocumentOwnershipIndexError,
    DocumentReferenceOwnershipPageV1,
    WorkspaceDocumentOwnershipIndexEntryV1,
    ownership_index_partition,
    parse_index_entry,
    reference_fingerprint,
)
from local_workspace_application.workspaces.knowledge_configuration_models import (
    WorkspaceCommittedQueryPolicy,
    WorkspaceConnectionAttachment,
    WorkspaceIndexedSourceBinding,
    WorkspaceKnowledgeConfigurationHead,
    WorkspaceKnowledgeMutationOperationV1,
    WorkspaceKnowledgeMutationRecord,
    WorkspaceLiveAccessBinding,
    parse_workspace_query_policy,
)
from local_workspace_application.workspaces.materialization_visibility import (
    KnowledgeMaterializationActivePointerV1,
    KnowledgeMaterializationOwnershipModeV1,
)
from local_workspace_application.workspaces.models import (
    ActiveKnowledgeIngestionLocator,
    IntakeBatch,
    KnowledgeInput,
    ManagedFileObject,
    WebUrlSourceLocator,
    Workspace,
    WorkspaceDocumentReference,
    WorkspaceOperation,
    WorkspaceOperationStatus,
    WorkspaceOperationType,
    WorkspaceSource,
)
from pydantic import BaseModel, ValidationError

from intergrax.integrations.contracts.document_store import (
    ConditionalDocumentStore,
    DocumentQueryPageV1,
    DocumentRecord,
    DocumentStore,
    validate_document_query_limit,
)

T = TypeVar("T", bound=BaseModel)

_IDEMPOTENCY_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_QUERY_POLICY_ENTITY_ID = "query-policy"
_REVISION_ROW_KEY_MARKER = ":rev:"
_KNOWLEDGE_CONFIGURATION_MAX_REVISION = 10**20 - 1
_KNOWLEDGE_CONFIGURATION_REVISION_SCAN_LIMIT = 2000
_KNOWLEDGE_CONFIGURATION_MUTATION_SCAN_LIMIT = 2000
_KNOWLEDGE_CONFIGURATION_MUTATION_SCAN_PROBE_LIMIT = 2001

_ENTITY_KNOWLEDGE_CONFIGURATION_HEAD = "knowledge_configuration_head"
_ENTITY_KNOWLEDGE_CONFIGURATION_MUTATION = "knowledge_configuration_mutation"
_ENTITY_KNOWLEDGE_CONFIGURATION_CONNECTION_ATTACHMENT = (
    "knowledge_configuration_connection_attachment"
)
_ENTITY_KNOWLEDGE_CONFIGURATION_INDEXED_SOURCE = "knowledge_configuration_indexed_source"
_ENTITY_KNOWLEDGE_CONFIGURATION_LIVE_ACCESS = "knowledge_configuration_live_access"
_ENTITY_KNOWLEDGE_CONFIGURATION_QUERY_POLICY = "knowledge_configuration_query_policy"
_ENTITY_CONNECTED_SOURCE_DELIVERY = "connected_source_delivery"
_ENTITY_CONNECTED_SOURCE_SYNC_ENQUEUE = "connected_source_sync_enqueue"
_ENTITY_CONNECTED_SOURCE_DELIVERY_SEQUENCE_LEGACY = "connected_source_delivery_sequence"
_ENTITY_CONNECTED_SOURCE_DELIVERY_SEQUENCE_HEAD = "connected_source_delivery_sequence_head"
_ENTITY_CONNECTED_SOURCE_DELIVERY_SEQUENCE_ASSIGNMENT = (
    "connected_source_delivery_sequence_assignment"
)


class WorkspaceKnowledgeConfigurationRepositoryError(RuntimeError):
    def __init__(self, error_code: str) -> None:
        super().__init__(error_code)
        self.error_code = error_code


@dataclass(frozen=True)
class ActiveKnowledgeIngestionLocatorScan:
    locators: tuple[ActiveKnowledgeIngestionLocator, ...]
    malformed_seen: int
    malformed_removed: int

_ENTITY_WORKSPACE = "workspace"
_ENTITY_SOURCE = "source"
_ENTITY_OPERATION = "operation"
_ENTITY_SOURCE_SYNC_OPERATION_INDEX = "source_sync_operation_index"
_ENTITY_DOCUMENT = "document"
_ENTITY_DOCUMENT_OWNERSHIP_INDEX = "document_ownership_index"
_ENTITY_KNOWLEDGE_INPUT = "knowledge_input"
_ENTITY_MANAGED_FILE = "managed_file"
_ENTITY_WEB_URL_LOCATOR = "web_url_locator"
_ENTITY_INTAKE_BATCH = "intake_batch"
_ENTITY_CONNECTED_SOURCE_DELIVERY_RECEIPT = "connected_source_delivery_receipt"
_ENTITY_MATERIALIZATION_ACTIVE_POINTER = "materialization_active_pointer"
_ACTIVE_KNOWLEDGE_INGESTION_PARTITION = "lkw.managed_workspace:active_knowledge_ingestion"


def _partition(tenant_id: str, entity: str) -> str:
    return f"lkw.managed_workspace:{tenant_id}:{entity}"


def _source_sync_operation_index_row_key(*, workspace_id: str, source_id: str) -> str:
    return f"{workspace_id}:{source_id}:latest"


def _revision_row_key(
    *,
    workspace_id: str,
    entity_id: str,
    revision: int,
) -> str:
    if revision < 1 or revision > _KNOWLEDGE_CONFIGURATION_MAX_REVISION:
        raise ValueError("knowledge_configuration_revision_invalid")
    return f"{workspace_id}:{entity_id}{_REVISION_ROW_KEY_MARKER}{revision:020d}"


def _mutation_row_key(
    *,
    workspace_id: str,
    operation: str,
    idempotency_key_hash: str,
) -> str:
    return f"{workspace_id}:{operation}:{idempotency_key_hash}"


def _parse_mutation_row_key(row_key: str) -> tuple[str, str, str]:
    hash_sep = row_key.rfind(":")
    if hash_sep < 0:
        raise ValueError("knowledge_configuration_record_identity_mismatch")
    idempotency_key_hash = row_key[hash_sep + 1 :]
    prefix = row_key[:hash_sep]
    colon_idx = prefix.find(":")
    if colon_idx < 0:
        raise ValueError("knowledge_configuration_record_identity_mismatch")
    workspace_id = prefix[:colon_idx]
    operation = prefix[colon_idx + 1 :]
    return workspace_id, operation, idempotency_key_hash


def _parse_revision_row_key(row_key: str) -> tuple[str, str, int]:
    if _REVISION_ROW_KEY_MARKER not in row_key:
        raise ValueError("knowledge_configuration_record_identity_mismatch")
    prefix, rev_part = row_key.rsplit(_REVISION_ROW_KEY_MARKER, 1)
    if len(rev_part) != 20 or not rev_part.isdigit():
        raise ValueError("knowledge_configuration_record_identity_mismatch")
    revision = int(rev_part)
    if revision < 1:
        raise ValueError("knowledge_configuration_record_identity_mismatch")
    colon_idx = prefix.find(":")
    if colon_idx < 0:
        raise ValueError("knowledge_configuration_record_identity_mismatch")
    workspace_id = prefix[:colon_idx]
    entity_id = prefix[colon_idx + 1 :]
    return workspace_id, entity_id, revision


def _assert_record_identity(
    *,
    tenant_id: str,
    workspace_id: str,
    model_tenant_id: str,
    model_workspace_id: str,
    entity_id: str | None = None,
    model_entity_id: str | None = None,
    revision: int | None = None,
    model_revision: int | None = None,
) -> None:
    if (
        model_tenant_id != tenant_id
        or model_workspace_id != workspace_id
        or (entity_id is not None and model_entity_id != entity_id)
        or (revision is not None and model_revision != revision)
    ):
        raise ValueError("knowledge_configuration_record_identity_mismatch")


def _validate_idempotency_key_hash(idempotency_key_hash: str) -> None:
    if _IDEMPOTENCY_HASH_RE.fullmatch(idempotency_key_hash) is None:
        raise WorkspaceKnowledgeConfigurationRepositoryError(
            "knowledge_configuration_idempotency_hash_invalid"
        )


def _to_document_record(
    model: BaseModel,
    *,
    partition_key: str,
    row_key: str,
) -> DocumentRecord:
    return DocumentRecord(
        partition_key=partition_key,
        row_key=row_key,
        data=model.model_dump(mode="json"),
    )


class ManagedWorkspaceRepository:
    """Tier-3 repository over the provider-neutral DocumentStore contract."""

    def __init__(self, document_store: DocumentStore) -> None:
        self._store = document_store

    @property
    def document_store(self) -> DocumentStore:
        return self._store

    def _put(self, partition_key: str, row_key: str, model: BaseModel) -> None:
        self._store.put(
            DocumentRecord(
                partition_key=partition_key,
                row_key=row_key,
                data=model.model_dump(mode="json"),
            )
        )

    def _get(self, partition_key: str, row_key: str, model_type: type[T]) -> T | None:
        record = self._store.get(partition_key, row_key)
        if record is None:
            return None
        return model_type.model_validate(dict(record.data), strict=False)

    def _list(self, partition_key: str, model_type: type[T], *, limit: int = 500) -> list[T]:
        result = self._store.query(partition_key, limit=limit)
        return [
            model_type.model_validate(dict(doc.data), strict=False)
            for doc in result.documents
        ]

    # --- Workspace ---

    def put_workspace(self, workspace: Workspace) -> Workspace:
        self._put(
            _partition(workspace.tenant_id, _ENTITY_WORKSPACE),
            workspace.workspace_id,
            workspace,
        )
        return workspace

    def get_workspace(self, *, tenant_id: str, workspace_id: str) -> Workspace | None:
        return self._get(
            _partition(tenant_id, _ENTITY_WORKSPACE),
            workspace_id,
            Workspace,
        )

    def delete_workspace(self, *, tenant_id: str, workspace_id: str) -> None:
        self._store.delete(_partition(tenant_id, _ENTITY_WORKSPACE), workspace_id)

    def list_workspaces(self, *, tenant_id: str) -> list[Workspace]:
        items = self._list(_partition(tenant_id, _ENTITY_WORKSPACE), Workspace)
        return sorted(items, key=lambda item: item.created_at, reverse=True)

    # --- Source ---

    def put_source(self, source: WorkspaceSource) -> WorkspaceSource:
        self._put(
            _partition(source.tenant_id, _ENTITY_SOURCE),
            f"{source.workspace_id}:{source.source_id}",
            source,
        )
        return source

    def put_source_if_absent(self, source: WorkspaceSource) -> bool:
        return self._put_if_absent(
            source,
            partition_key=_partition(source.tenant_id, _ENTITY_SOURCE),
            row_key=f"{source.workspace_id}:{source.source_id}",
        )

    def delete_source_if_match(self, source: WorkspaceSource) -> bool:
        return self._delete_if_match(
            source,
            partition_key=_partition(source.tenant_id, _ENTITY_SOURCE),
            row_key=f"{source.workspace_id}:{source.source_id}",
        )

    def get_source(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        source_id: str,
    ) -> WorkspaceSource | None:
        return self._get(
            _partition(tenant_id, _ENTITY_SOURCE),
            f"{workspace_id}:{source_id}",
            WorkspaceSource,
        )

    def list_sources(self, *, tenant_id: str, workspace_id: str) -> list[WorkspaceSource]:
        prefix = f"{workspace_id}:"
        result = self._store.query(
            _partition(tenant_id, _ENTITY_SOURCE),
            limit=500,
            row_key_prefix=prefix,
        )
        items = [WorkspaceSource.model_validate(dict(doc.data)) for doc in result.documents]
        return sorted(items, key=lambda item: item.created_at)

    def delete_source(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        source_id: str,
    ) -> None:
        self._store.delete(
            _partition(tenant_id, _ENTITY_SOURCE),
            f"{workspace_id}:{source_id}",
        )

    def delete_sources_for_workspace(self, *, tenant_id: str, workspace_id: str) -> int:
        deleted = 0
        for source in self.list_sources(tenant_id=tenant_id, workspace_id=workspace_id):
            self.delete_source(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                source_id=source.source_id,
            )
            deleted += 1
        return deleted

    # --- Connected source delivery receipts ---

    def put_connected_source_delivery_receipt(
        self,
        receipt: ConnectedSourceDeliveryReceipt,
    ) -> ConnectedSourceDeliveryReceipt:
        partition_key = _partition(receipt.tenant_id, _ENTITY_CONNECTED_SOURCE_DELIVERY_RECEIPT)
        row_key = (
            f"{receipt.workspace_id}:{receipt.source_id}:{receipt.delivery_id}"
        )
        self._put(partition_key, row_key, receipt)
        self.put_connected_source_delivery_receipt_ownership_index_entry(
            ConnectedSourceDeliveryReceiptOwnershipIndexEntryV1.for_receipt(receipt)
        )
        return receipt

    def put_connected_source_delivery_receipt_if_absent(
        self,
        receipt: ConnectedSourceDeliveryReceipt,
    ) -> bool:
        partition_key = _partition(receipt.tenant_id, _ENTITY_CONNECTED_SOURCE_DELIVERY_RECEIPT)
        row_key = (
            f"{receipt.workspace_id}:{receipt.source_id}:{receipt.delivery_id}"
        )
        created = self._put_if_absent(
            receipt, partition_key=partition_key, row_key=row_key
        )
        stored = (
            receipt
            if created
            else self.get_connected_source_delivery_receipt(
                tenant_id=receipt.tenant_id,
                workspace_id=receipt.workspace_id,
                source_id=receipt.source_id,
                delivery_id=receipt.delivery_id,
            )
        )
        if stored is not None:
            self.put_connected_source_delivery_receipt_ownership_index_entry(
                ConnectedSourceDeliveryReceiptOwnershipIndexEntryV1.for_receipt(stored)
            )
        return created

    def complete_connected_source_delivery_receipt_if_in_progress(
        self,
        *,
        expected: ConnectedSourceDeliveryReceipt,
        replacement: ConnectedSourceDeliveryReceipt,
    ) -> bool:
        partition_key = _partition(expected.tenant_id, _ENTITY_CONNECTED_SOURCE_DELIVERY_RECEIPT)
        row_key = f"{expected.workspace_id}:{expected.source_id}:{expected.delivery_id}"
        replaced = self._replace_if_match(
            expected=expected,
            replacement=replacement,
            partition_key=partition_key,
            row_key=row_key,
        )
        if replaced:
            self.put_connected_source_delivery_receipt_ownership_index_entry(
                ConnectedSourceDeliveryReceiptOwnershipIndexEntryV1.for_receipt(
                    replacement
                )
            )
        return replaced

    def get_connected_source_delivery_receipt(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        source_id: str,
        delivery_id: str,
    ) -> ConnectedSourceDeliveryReceipt | None:
        return self._get(
            _partition(tenant_id, _ENTITY_CONNECTED_SOURCE_DELIVERY_RECEIPT),
            f"{workspace_id}:{source_id}:{delivery_id}",
            ConnectedSourceDeliveryReceipt,
        )

    def get_connected_source_delivery_sequence_head(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        source_id: str,
        indexed_source_binding_id: str,
    ) -> ConnectedSourceDeliverySequenceHead | None:
        row_key = self._delivery_sequence_head_key(
            workspace_id=workspace_id,
            source_id=source_id,
            indexed_source_binding_id=indexed_source_binding_id,
        )
        partition_key = _partition(
            tenant_id, _ENTITY_CONNECTED_SOURCE_DELIVERY_SEQUENCE_HEAD
        )
        record = self._store.get(partition_key, row_key)
        if record is None:
            legacy_record = self._store.get(
                _partition(tenant_id, _ENTITY_CONNECTED_SOURCE_DELIVERY_SEQUENCE_LEGACY),
                row_key,
            )
            if legacy_record is not None:
                raise WorkspaceKnowledgeConfigurationRepositoryError(
                    "connected_source_delivery_sequence_migration_required"
                )
            return None
        head = ConnectedSourceDeliverySequenceHead.model_validate(dict(record.data))
        if (
            head.tenant_id != tenant_id
            or head.workspace_id != workspace_id
            or head.source_id != source_id
            or head.indexed_source_binding_id != indexed_source_binding_id
        ):
            raise ValueError("connected_source_delivery_sequence_identity_mismatch")
        return head

    def get_connected_source_delivery_sequence_ledger(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        source_id: str,
        indexed_source_binding_id: str,
    ) -> ConnectedSourceDeliverySequenceHead | None:
        """Deprecated compatibility name for the bounded sequence head."""
        return self.get_connected_source_delivery_sequence_head(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            source_id=source_id,
            indexed_source_binding_id=indexed_source_binding_id,
        )

    def put_connected_source_delivery_sequence_head_if_absent(
        self,
        head: ConnectedSourceDeliverySequenceHead,
    ) -> bool:
        return self._put_if_absent(
            head,
            partition_key=_partition(
                head.tenant_id, _ENTITY_CONNECTED_SOURCE_DELIVERY_SEQUENCE_HEAD
            ),
            row_key=self._delivery_sequence_head_key(
                workspace_id=head.workspace_id,
                source_id=head.source_id,
                indexed_source_binding_id=head.indexed_source_binding_id,
            ),
        )

    def replace_connected_source_delivery_sequence_head_if_match(
        self,
        *,
        expected: ConnectedSourceDeliverySequenceHead,
        replacement: ConnectedSourceDeliverySequenceHead,
    ) -> bool:
        if (
            expected.tenant_id != replacement.tenant_id
            or expected.workspace_id != replacement.workspace_id
            or expected.source_id != replacement.source_id
            or expected.indexed_source_binding_id != replacement.indexed_source_binding_id
        ):
            raise ValueError("connected_source_delivery_sequence_identity_mismatch")
        return self._replace_if_match(
            expected=expected,
            replacement=replacement,
            partition_key=_partition(
                expected.tenant_id, _ENTITY_CONNECTED_SOURCE_DELIVERY_SEQUENCE_HEAD
            ),
            row_key=self._delivery_sequence_head_key(
                workspace_id=expected.workspace_id,
                source_id=expected.source_id,
                indexed_source_binding_id=expected.indexed_source_binding_id,
            ),
        )

    def get_connected_source_delivery_sequence_assignment(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        source_id: str,
        indexed_source_binding_id: str,
        delivery_id: str,
    ) -> ConnectedSourceDeliverySequenceAssignment | None:
        if _IDEMPOTENCY_HASH_RE.fullmatch(delivery_id) is None:
            raise ValueError("connected_source_delivery_id_invalid")
        row_key = self._delivery_sequence_assignment_key(
            workspace_id=workspace_id,
            source_id=source_id,
            indexed_source_binding_id=indexed_source_binding_id,
            delivery_id=delivery_id,
        )
        record = self._store.get(
            _partition(tenant_id, _ENTITY_CONNECTED_SOURCE_DELIVERY_SEQUENCE_ASSIGNMENT),
            row_key,
        )
        if record is None:
            return None
        assignment = ConnectedSourceDeliverySequenceAssignment.model_validate(
            dict(record.data)
        )
        if (
            assignment.tenant_id != tenant_id
            or assignment.workspace_id != workspace_id
            or assignment.source_id != source_id
            or assignment.indexed_source_binding_id != indexed_source_binding_id
            or assignment.delivery_id != delivery_id
        ):
            raise ValueError("connected_source_delivery_sequence_identity_mismatch")
        return assignment

    def put_connected_source_delivery_sequence_assignment_if_absent(
        self,
        assignment: ConnectedSourceDeliverySequenceAssignment,
    ) -> bool:
        return self._put_if_absent(
            assignment,
            partition_key=_partition(
                assignment.tenant_id, _ENTITY_CONNECTED_SOURCE_DELIVERY_SEQUENCE_ASSIGNMENT
            ),
            row_key=self._delivery_sequence_assignment_key(
                workspace_id=assignment.workspace_id,
                source_id=assignment.source_id,
                indexed_source_binding_id=assignment.indexed_source_binding_id,
                delivery_id=assignment.delivery_id,
            ),
        )

    def list_connected_source_delivery_sequence_assignments(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        source_id: str,
        indexed_source_binding_id: str,
    ) -> list[ConnectedSourceDeliverySequenceAssignment]:
        prefix = self._delivery_sequence_head_key(
            workspace_id=workspace_id,
            source_id=source_id,
            indexed_source_binding_id=indexed_source_binding_id,
        ) + ":"
        result = self._store.query(
            _partition(tenant_id, _ENTITY_CONNECTED_SOURCE_DELIVERY_SEQUENCE_ASSIGNMENT),
            limit=5000,
            row_key_prefix=prefix,
        )
        assignments: list[ConnectedSourceDeliverySequenceAssignment] = []
        for record in result.documents:
            assignment = ConnectedSourceDeliverySequenceAssignment.model_validate(
                dict(record.data)
            )
            if (
                assignment.tenant_id != tenant_id
                or assignment.workspace_id != workspace_id
                or assignment.source_id != source_id
                or assignment.indexed_source_binding_id != indexed_source_binding_id
                or record.row_key
                != self._delivery_sequence_assignment_key(
                    workspace_id=workspace_id,
                    source_id=source_id,
                    indexed_source_binding_id=indexed_source_binding_id,
                    delivery_id=assignment.delivery_id,
                )
            ):
                raise ValueError("connected_source_delivery_sequence_identity_mismatch")
            assignments.append(assignment)
        return sorted(assignments, key=lambda item: item.materialization_sequence)

    def allocate_connected_source_delivery_sequence(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        source_id: str,
        indexed_source_binding_id: str,
        delivery_id: str,
    ) -> int:
        if _IDEMPOTENCY_HASH_RE.fullmatch(delivery_id) is None:
            raise ValueError("connected_source_delivery_id_invalid")
        while True:
            assignment = self.get_connected_source_delivery_sequence_assignment(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                source_id=source_id,
                indexed_source_binding_id=indexed_source_binding_id,
                delivery_id=delivery_id,
            )
            if assignment is not None:
                return assignment.materialization_sequence

            current = self.get_connected_source_delivery_sequence_head(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                source_id=source_id,
                indexed_source_binding_id=indexed_source_binding_id,
            )
            if current is None:
                candidate = ConnectedSourceDeliverySequenceHead(
                    tenant_id=tenant_id,
                    workspace_id=workspace_id,
                    source_id=source_id,
                    indexed_source_binding_id=indexed_source_binding_id,
                    next_sequence=2,
                )
                sequence = 1
                if not self.put_connected_source_delivery_sequence_head_if_absent(candidate):
                    continue
            else:
                sequence = current.next_sequence
                replacement = current.model_copy(update={"next_sequence": sequence + 1})
                if not self.replace_connected_source_delivery_sequence_head_if_match(
                    expected=current,
                    replacement=replacement,
                ):
                    continue

            assignment = ConnectedSourceDeliverySequenceAssignment(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                source_id=source_id,
                indexed_source_binding_id=indexed_source_binding_id,
                delivery_id=delivery_id,
                materialization_sequence=sequence,
                assigned_at=datetime.now(UTC),
            )
            if self.put_connected_source_delivery_sequence_assignment_if_absent(
                assignment
            ):
                return sequence
            winner = self.get_connected_source_delivery_sequence_assignment(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                source_id=source_id,
                indexed_source_binding_id=indexed_source_binding_id,
                delivery_id=delivery_id,
            )
            if winner is not None:
                return winner.materialization_sequence

    @staticmethod
    def _delivery_sequence_head_key(
        *,
        workspace_id: str,
        source_id: str,
        indexed_source_binding_id: str,
    ) -> str:
        return f"{workspace_id}:{source_id}:{indexed_source_binding_id}"

    @staticmethod
    def _delivery_sequence_assignment_key(
        *,
        workspace_id: str,
        source_id: str,
        indexed_source_binding_id: str,
        delivery_id: str,
    ) -> str:
        return (
            f"{workspace_id}:{source_id}:{indexed_source_binding_id}:"
            f"{delivery_id}"
        )

    def put_active_materialization_pointer_if_absent(
        self,
        pointer: KnowledgeMaterializationActivePointerV1,
    ) -> bool:
        return self._put_if_absent(
            pointer,
            partition_key=_partition(
                pointer.tenant_id, _ENTITY_MATERIALIZATION_ACTIVE_POINTER
            ),
            row_key=self._active_materialization_pointer_key(
                workspace_id=pointer.workspace_id,
                source_id=pointer.source_id,
                indexed_source_binding_id=pointer.indexed_source_binding_id,
                remote_id=pointer.remote_id,
            ),
        )

    def get_active_materialization_pointer(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        source_id: str,
        indexed_source_binding_id: str,
        remote_id: str,
    ) -> KnowledgeMaterializationActivePointerV1 | None:
        return self._get(
            _partition(tenant_id, _ENTITY_MATERIALIZATION_ACTIVE_POINTER),
            self._active_materialization_pointer_key(
                workspace_id=workspace_id,
                source_id=source_id,
                indexed_source_binding_id=indexed_source_binding_id,
                remote_id=remote_id,
            ),
            KnowledgeMaterializationActivePointerV1,
        )

    def replace_active_materialization_pointer(
        self,
        *,
        expected: KnowledgeMaterializationActivePointerV1,
        replacement: KnowledgeMaterializationActivePointerV1,
    ) -> bool:
        if (
            expected.tenant_id != replacement.tenant_id
            or self._active_materialization_pointer_key(
                workspace_id=expected.workspace_id,
                source_id=expected.source_id,
                indexed_source_binding_id=expected.indexed_source_binding_id,
                remote_id=expected.remote_id,
            )
            != self._active_materialization_pointer_key(
                workspace_id=replacement.workspace_id,
                source_id=replacement.source_id,
                indexed_source_binding_id=replacement.indexed_source_binding_id,
                remote_id=replacement.remote_id,
            )
        ):
            raise ValueError("active_materialization_pointer_identity_mismatch")
        return self._replace_if_match(
            expected=expected,
            replacement=replacement,
            partition_key=_partition(
                expected.tenant_id, _ENTITY_MATERIALIZATION_ACTIVE_POINTER
            ),
            row_key=self._active_materialization_pointer_key(
                workspace_id=expected.workspace_id,
                source_id=expected.source_id,
                indexed_source_binding_id=expected.indexed_source_binding_id,
                remote_id=expected.remote_id,
            ),
        )

    @staticmethod
    def _active_materialization_pointer_key(
        *,
        workspace_id: str,
        source_id: str,
        indexed_source_binding_id: str,
        remote_id: str,
    ) -> str:
        return (
            f"{workspace_id}:{source_id}:"
            f"{indexed_source_binding_id}:{remote_id}"
        )

    def delete_connected_source_delivery_receipt(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        source_id: str,
        delivery_id: str,
    ) -> None:
        receipt = self.get_connected_source_delivery_receipt(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            source_id=source_id,
            delivery_id=delivery_id,
        )
        self._store.delete(
            _partition(tenant_id, _ENTITY_CONNECTED_SOURCE_DELIVERY_RECEIPT),
            f"{workspace_id}:{source_id}:{delivery_id}",
        )
        if receipt is not None:
            self.delete_connected_source_delivery_receipt_ownership_index_entry(
                ConnectedSourceDeliveryReceiptOwnershipIndexEntryV1.for_receipt(receipt)
            )

    def list_connected_source_delivery_receipts_page(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        source_id: str,
        limit: int,
        cursor: str | None = None,
    ) -> DocumentQueryPageV1:
        validate_document_query_limit(limit)
        return self._store.query(
            _partition(tenant_id, _ENTITY_CONNECTED_SOURCE_DELIVERY_RECEIPT),
            limit=limit,
            row_key_prefix=f"{workspace_id}:{source_id}:",
            cursor=cursor,
        )

    def put_connected_source_delivery_receipt_ownership_index_entry(
        self,
        entry: ConnectedSourceDeliveryReceiptOwnershipIndexEntryV1,
    ) -> ConnectedSourceDeliveryReceiptOwnershipIndexEntryV1:
        record = entry.to_document()
        existing = self._store.get(record.partition_key, record.row_key)
        if existing is not None:
            parsed = parse_delivery_receipt_ownership_index_entry(existing)
            if parsed.ownership_scope != entry.ownership_scope or (
                parsed.delivery_id != entry.delivery_id
            ):
                raise ConnectedSourceDeliveryReceiptOwnershipIndexError(
                    "delivery_receipt_ownership_index_conflict"
                )
            if parsed != entry:
                if not isinstance(self._store, ConditionalDocumentStore):
                    raise ConnectedSourceDeliveryReceiptOwnershipIndexError(
                        "delivery_receipt_ownership_index_store_unavailable"
                    )
                if not self._store.replace_if_match(
                    expected=existing,
                    replacement=record,
                ):
                    retry = self._store.get(record.partition_key, record.row_key)
                    if (
                        retry is None
                        or parse_delivery_receipt_ownership_index_entry(retry) != entry
                    ):
                        raise ConnectedSourceDeliveryReceiptOwnershipIndexError(
                            "delivery_receipt_ownership_index_conflict"
                        )
            return entry
        if not isinstance(self._store, ConditionalDocumentStore):
            raise ConnectedSourceDeliveryReceiptOwnershipIndexError(
                "delivery_receipt_ownership_index_store_unavailable"
            )
        if not self._store.put_if_absent(record):
            retry = self._store.get(record.partition_key, record.row_key)
            if (
                retry is None
                or parse_delivery_receipt_ownership_index_entry(retry) != entry
            ):
                raise ConnectedSourceDeliveryReceiptOwnershipIndexError(
                    "delivery_receipt_ownership_index_conflict"
                )
        return entry

    def delete_connected_source_delivery_receipt_ownership_index_entry(
        self,
        entry: ConnectedSourceDeliveryReceiptOwnershipIndexEntryV1,
    ) -> bool:
        record = self._store.get(
            delivery_receipt_ownership_index_partition(entry.tenant_id),
            entry.row_key,
        )
        if record is None:
            return False
        if parse_delivery_receipt_ownership_index_entry(record) != entry:
            raise ConnectedSourceDeliveryReceiptOwnershipIndexError(
                "delivery_receipt_ownership_index_delete_conflict"
            )
        self._store.delete(record.partition_key, record.row_key)
        return True

    def list_connected_source_delivery_receipts_by_owner(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        source_id: str,
        indexed_source_binding_id: str,
        knowledge_source_binding_ref: str,
        limit: int,
        cursor: str | None = None,
    ) -> ConnectedSourceDeliveryReceiptOwnershipPageV1:
        """Exact binding-scoped receipt enumeration via derived ownership index."""
        validate_document_query_limit(limit)
        scope = (
            tenant_id,
            workspace_id,
            source_id,
            indexed_source_binding_id,
            knowledge_source_binding_ref,
        )
        prefix = delivery_receipt_ownership_scope_prefix(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            source_id=source_id,
            indexed_source_binding_id=indexed_source_binding_id,
            knowledge_source_binding_ref=knowledge_source_binding_ref,
        )
        page = self._store.query(
            delivery_receipt_ownership_index_partition(tenant_id),
            limit=limit,
            row_key_prefix=prefix,
            cursor=cursor,
        )
        documents: list[DocumentRecord] = []
        orphan_entries: list[ConnectedSourceDeliveryReceiptOwnershipIndexEntryV1] = []
        for record in page.documents:
            entry = parse_delivery_receipt_ownership_index_entry(record)
            if entry.ownership_scope != scope:
                raise ConnectedSourceDeliveryReceiptOwnershipIndexError(
                    "delivery_receipt_ownership_index_scope_mismatch"
                )
            canonical = self._store.get(
                entry.canonical_partition_key,
                entry.canonical_row_key,
            )
            if canonical is None:
                orphan_entries.append(entry)
                continue
            try:
                receipt = ConnectedSourceDeliveryReceipt.model_validate(
                    dict(canonical.data), strict=False
                )
            except (TypeError, ValueError):
                raise ConnectedSourceDeliveryReceiptOwnershipIndexError(
                    "delivery_receipt_ownership_index_reference_mismatch"
                ) from None
            if (
                receipt.tenant_id != tenant_id
                or receipt.workspace_id != workspace_id
                or receipt.source_id != source_id
                or receipt.indexed_source_binding_id != indexed_source_binding_id
                or receipt.knowledge_source_binding_ref
                != knowledge_source_binding_ref
                or receipt.delivery_id != entry.delivery_id
                or canonical.partition_key != entry.canonical_partition_key
                or canonical.row_key != entry.canonical_row_key
            ):
                raise ConnectedSourceDeliveryReceiptOwnershipIndexError(
                    "delivery_receipt_ownership_index_reference_mismatch"
                )
            documents.append(canonical)
        return ConnectedSourceDeliveryReceiptOwnershipPageV1(
            documents=tuple(documents),
            orphan_index_entries=tuple(orphan_entries),
            next_cursor=page.next_cursor,
        )

    def put_connected_source_recovery_migration_gate(
        self,
        gate: ConnectedSourceRecoveryMigrationGateV1,
    ) -> ConnectedSourceRecoveryMigrationGateV1:
        record = gate.to_document()
        existing = self._store.get(record.partition_key, record.row_key)
        if existing is not None:
            parsed = parse_recovery_migration_gate(existing)
            if parsed.ownership_scope != gate.ownership_scope:
                raise ConnectedSourceRecoveryMigrationGateError(
                    "migration_gate_conflict"
                )
            if parsed != gate:
                if not isinstance(self._store, ConditionalDocumentStore):
                    raise ConnectedSourceRecoveryMigrationGateError(
                        "migration_gate_store_unavailable"
                    )
                if not self._store.replace_if_match(
                    expected=existing,
                    replacement=record,
                ):
                    retry = self._store.get(record.partition_key, record.row_key)
                    if retry is None or parse_recovery_migration_gate(retry) != gate:
                        raise ConnectedSourceRecoveryMigrationGateError(
                            "migration_gate_conflict"
                        )
            return gate
        if not isinstance(self._store, ConditionalDocumentStore):
            raise ConnectedSourceRecoveryMigrationGateError(
                "migration_gate_store_unavailable"
            )
        if not self._store.put_if_absent(record):
            retry = self._store.get(record.partition_key, record.row_key)
            if retry is None or parse_recovery_migration_gate(retry) != gate:
                raise ConnectedSourceRecoveryMigrationGateError(
                    "migration_gate_conflict"
                )
        return gate

    def get_connected_source_recovery_migration_gate(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        source_id: str,
        indexed_source_binding_id: str,
        knowledge_source_binding_ref: str,
    ) -> ConnectedSourceRecoveryMigrationGateV1 | None:
        probe = ConnectedSourceRecoveryMigrationGateV1(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            source_id=source_id,
            indexed_source_binding_id=indexed_source_binding_id,
            knowledge_source_binding_ref=knowledge_source_binding_ref,
            status=ConnectedSourceRecoveryMigrationGateStatusV1.REQUIRED,
            schema_version=1,
        )
        record = self._store.get(
            recovery_migration_gate_partition(tenant_id),
            probe.row_key,
        )
        if record is None:
            return None
        gate = parse_recovery_migration_gate(record)
        if gate.ownership_scope != (
            tenant_id,
            workspace_id,
            source_id,
            indexed_source_binding_id,
            knowledge_source_binding_ref,
        ):
            raise ConnectedSourceRecoveryMigrationGateError(
                "migration_gate_identity_mismatch"
            )
        return gate

    # --- Connected source delivery accounting ---

    def put_connected_source_delivery_accounting_if_absent(
        self,
        accounting: ConnectedSourceOperationDeliveryAccounting,
    ) -> bool:
        partition_key = _partition(accounting.tenant_id, _ENTITY_CONNECTED_SOURCE_DELIVERY)
        row_key = f"{accounting.operation_id}:{accounting.delivery_id}"
        created = self._put_if_absent(
            accounting,
            partition_key=partition_key,
            row_key=row_key,
        )
        stored = (
            accounting
            if created
            else self.get_connected_source_delivery_accounting(
                tenant_id=accounting.tenant_id,
                operation_id=accounting.operation_id,
                delivery_id=accounting.delivery_id,
            )
        )
        if (
            stored is not None
            and stored.ownership_classification == "COMPLETE_OWNERSHIP"
        ):
            self.put_connected_source_recovery_ownership_index_entry(
                index_entry_for_delivery_accounting(
                    stored,
                    canonical_partition_key=partition_key,
                    canonical_row_key=row_key,
                )
            )
        return created

    def get_connected_source_delivery_accounting(
        self,
        *,
        tenant_id: str,
        operation_id: str,
        delivery_id: str,
    ) -> ConnectedSourceOperationDeliveryAccounting | None:
        return self._get(
            _partition(tenant_id, _ENTITY_CONNECTED_SOURCE_DELIVERY),
            f"{operation_id}:{delivery_id}",
            ConnectedSourceOperationDeliveryAccounting,
        )

    def delete_connected_source_delivery_accounting(
        self,
        *,
        tenant_id: str,
        operation_id: str,
        delivery_id: str,
    ) -> None:
        self._store.delete(
            _partition(tenant_id, _ENTITY_CONNECTED_SOURCE_DELIVERY),
            f"{operation_id}:{delivery_id}",
        )

    def list_connected_source_delivery_accounting(
        self,
        *,
        tenant_id: str,
        operation_id: str,
    ) -> list[ConnectedSourceOperationDeliveryAccounting]:
        result = self._store.query(
            _partition(tenant_id, _ENTITY_CONNECTED_SOURCE_DELIVERY),
            limit=500,
            row_key_prefix=f"{operation_id}:",
        )
        items = [
            ConnectedSourceOperationDeliveryAccounting.model_validate(dict(doc.data))
            for doc in result.documents
        ]
        return sorted(items, key=lambda item: (item.accounted_at, item.delivery_id))

    def list_connected_source_delivery_accounting_page(
        self,
        *,
        tenant_id: str,
        limit: int,
        cursor: str | None = None,
    ) -> DocumentQueryPageV1:
        validate_document_query_limit(limit)
        return self._store.query(
            _partition(tenant_id, _ENTITY_CONNECTED_SOURCE_DELIVERY),
            limit=limit,
            cursor=cursor,
        )

    # --- Connected source sync enqueue intent ---

    def put_connected_source_sync_enqueue_intent(
        self,
        intent: ConnectedSourceSyncEnqueueIntent,
    ) -> ConnectedSourceSyncEnqueueIntent:
        partition_key = _partition(intent.tenant_id, _ENTITY_CONNECTED_SOURCE_SYNC_ENQUEUE)
        self._put(partition_key, intent.operation_id, intent)
        if intent.ownership_classification == "COMPLETE_OWNERSHIP":
            self.put_connected_source_recovery_ownership_index_entry(
                index_entry_for_enqueue_intent(
                    intent,
                    canonical_partition_key=partition_key,
                    canonical_row_key=intent.operation_id,
                )
            )
        return intent

    def put_connected_source_sync_enqueue_intent_if_absent(
        self,
        intent: ConnectedSourceSyncEnqueueIntent,
    ) -> bool:
        partition_key = _partition(intent.tenant_id, _ENTITY_CONNECTED_SOURCE_SYNC_ENQUEUE)
        created = self._put_if_absent(
            intent,
            partition_key=partition_key,
            row_key=intent.operation_id,
        )
        stored = (
            intent
            if created
            else self.get_connected_source_sync_enqueue_intent(
                tenant_id=intent.tenant_id,
                operation_id=intent.operation_id,
            )
        )
        if (
            stored is not None
            and stored.ownership_classification == "COMPLETE_OWNERSHIP"
        ):
            self.put_connected_source_recovery_ownership_index_entry(
                index_entry_for_enqueue_intent(
                    stored,
                    canonical_partition_key=partition_key,
                    canonical_row_key=intent.operation_id,
                )
            )
        return created

    def allocate_connected_source_sync_enqueue_generation(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        source_id: str,
        operation_id: str,
        indexed_source_binding_id: str | None = None,
        knowledge_source_binding_ref: str | None = None,
        max_attempts: int = 3,
    ) -> ConnectedSourceSyncEnqueueIntent:
        from datetime import UTC, datetime

        partition_key = _partition(tenant_id, _ENTITY_CONNECTED_SOURCE_SYNC_ENQUEUE)
        for _ in range(max_attempts):
            existing = self.get_connected_source_sync_enqueue_intent(
                tenant_id=tenant_id,
                operation_id=operation_id,
            )
            now = datetime.now(UTC)
            if existing is None:
                complete_ownership = (
                    indexed_source_binding_id is not None
                    and knowledge_source_binding_ref is not None
                )
                intent = ConnectedSourceSyncEnqueueIntent(
                    tenant_id=tenant_id,
                    workspace_id=workspace_id,
                    source_id=source_id,
                    indexed_source_binding_id=indexed_source_binding_id,
                    knowledge_source_binding_ref=knowledge_source_binding_ref,
                    operation_id=operation_id,
                    enqueue_generation=1,
                    last_enqueued_generation=0,
                    updated_at=now,
                    ownership_classification=(
                        "COMPLETE_OWNERSHIP"
                        if complete_ownership
                        else "LEGACY_MIGRATION_REQUIRED"
                    ),
                )
                if self.put_connected_source_sync_enqueue_intent_if_absent(intent):
                    return intent
                continue
            if (
                (
                    indexed_source_binding_id is not None
                    or knowledge_source_binding_ref is not None
                )
                and (
                    existing.ownership_classification != "COMPLETE_OWNERSHIP"
                    or existing.indexed_source_binding_id != indexed_source_binding_id
                    or existing.knowledge_source_binding_ref
                    != knowledge_source_binding_ref
                )
            ):
                raise ValueError("connected_source_enqueue_migration_required")
            updated = existing.model_copy(
                update={
                    "enqueue_generation": existing.enqueue_generation + 1,
                    "updated_at": now,
                }
            )
            if self._replace_if_match(
                expected=existing,
                replacement=updated,
                partition_key=partition_key,
                row_key=operation_id,
            ):
                if updated.ownership_classification == "COMPLETE_OWNERSHIP":
                    self.put_connected_source_recovery_ownership_index_entry(
                        index_entry_for_enqueue_intent(
                            updated,
                            canonical_partition_key=partition_key,
                            canonical_row_key=operation_id,
                        )
                    )
                return updated
        raise RuntimeError("connected_source_enqueue_generation_allocation_failed")

    def resolve_connected_source_ownership_for_source(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        source_id: str,
    ) -> tuple[str, str] | None:
        versions = self.list_knowledge_indexed_source_versions(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
        )
        matches = [
            item
            for item in versions
            if item.source_id == source_id
        ]
        if not matches:
            return None
        binding_ids = {
            (
                item.indexed_source_binding_id,
                item.knowledge_source_binding_ref,
            )
            for item in matches
        }
        if len(binding_ids) != 1:
            raise ValueError("connected_source_ownership_ambiguous")
        return next(iter(binding_ids))

    def get_connected_source_sync_enqueue_intent(
        self,
        *,
        tenant_id: str,
        operation_id: str,
    ) -> ConnectedSourceSyncEnqueueIntent | None:
        return self._get(
            _partition(tenant_id, _ENTITY_CONNECTED_SOURCE_SYNC_ENQUEUE),
            operation_id,
            ConnectedSourceSyncEnqueueIntent,
        )

    def mark_connected_source_sync_enqueued(
        self,
        *,
        tenant_id: str,
        operation_id: str,
        expected_generation: int,
        task_id: str,
        queue_provider: str,
    ) -> bool:
        intent = self.get_connected_source_sync_enqueue_intent(
            tenant_id=tenant_id,
            operation_id=operation_id,
        )
        if intent is None or intent.enqueue_generation != expected_generation:
            return False
        if intent.last_enqueued_generation >= expected_generation:
            return True
        from datetime import UTC, datetime

        updated = intent.model_copy(
            update={
                "last_enqueued_generation": expected_generation,
                "last_task_id": task_id,
                "last_queue_provider": queue_provider,
                "updated_at": datetime.now(UTC),
            }
        )
        return self._replace_if_match(
            expected=intent,
            replacement=updated,
            partition_key=_partition(tenant_id, _ENTITY_CONNECTED_SOURCE_SYNC_ENQUEUE),
            row_key=operation_id,
        )

    def delete_connected_source_sync_enqueue_intent(
        self,
        *,
        tenant_id: str,
        operation_id: str,
    ) -> None:
        self._store.delete(
            _partition(tenant_id, _ENTITY_CONNECTED_SOURCE_SYNC_ENQUEUE),
            operation_id,
        )

    def list_connected_source_sync_enqueue_intents_page(
        self,
        *,
        tenant_id: str,
        limit: int,
        cursor: str | None = None,
    ) -> DocumentQueryPageV1:
        validate_document_query_limit(limit)
        return self._store.query(
            _partition(tenant_id, _ENTITY_CONNECTED_SOURCE_SYNC_ENQUEUE),
            limit=limit,
            cursor=cursor,
        )

    # --- Operation ---

    def put_operation(self, operation: WorkspaceOperation) -> WorkspaceOperation:
        if operation.operation_type is WorkspaceOperationType.SOURCE_SYNC:
            # Publish the bounded source pointer first.  A crash before the
            # canonical write leaves only a stale pointer, which readers
            # validate and ignore; replaying this write repairs the pointer.
            is_new_operation = (
                self.get_operation(
                    tenant_id=operation.tenant_id,
                    operation_id=operation.operation_id,
                )
                is None
            )
            self._put_source_sync_operation_index(
                operation,
                force_latest=is_new_operation,
            )
        self._put(
            _partition(operation.tenant_id, _ENTITY_OPERATION),
            operation.operation_id,
            operation,
        )
        return operation

    @staticmethod
    def _source_sync_operation_sort_key(operation: WorkspaceOperation) -> tuple[str, str]:
        timestamp = operation.created_at or operation.started_at or operation.completed_at
        if timestamp is None:
            return ("", operation.operation_id)
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=UTC)
        else:
            timestamp = timestamp.astimezone(UTC)
        return (timestamp.isoformat(), operation.operation_id)

    def _put_source_sync_operation_index(
        self,
        operation: WorkspaceOperation,
        *,
        force_latest: bool,
    ) -> None:
        partition_key = _partition(operation.tenant_id, _ENTITY_SOURCE_SYNC_OPERATION_INDEX)
        row_key = _source_sync_operation_index_row_key(
            workspace_id=operation.workspace_id,
            source_id=operation.source_id,
        )
        sort_timestamp, sort_operation_id = self._source_sync_operation_sort_key(operation)
        record = DocumentRecord(
            partition_key=partition_key,
            row_key=row_key,
            data={
                "tenant_id": operation.tenant_id,
                "workspace_id": operation.workspace_id,
                "source_id": operation.source_id,
                "operation_type": operation.operation_type.value,
                "operation_id": operation.operation_id,
                "sort_timestamp": sort_timestamp,
                "sort_operation_id": sort_operation_id,
            },
        )
        for _ in range(4):
            existing = self._store.get(partition_key, row_key)
            if existing is None:
                if not isinstance(self._store, ConditionalDocumentStore):
                    self._store.put(record)
                    return
                if self._store.put_if_absent(record):
                    return
                continue
            existing_data = dict(existing.data)
            if any(
                existing_data.get(field) != expected
                for field, expected in (
                    ("tenant_id", operation.tenant_id),
                    ("workspace_id", operation.workspace_id),
                    ("source_id", operation.source_id),
                    ("operation_type", WorkspaceOperationType.SOURCE_SYNC.value),
                )
            ):
                raise RuntimeError("source_sync_operation_index_identity_conflict")
            if existing_data.get("operation_id") == operation.operation_id:
                return
            if force_latest:
                should_replace = True
            else:
                existing_sort_key = (
                    str(existing_data.get("sort_timestamp") or ""),
                    str(existing_data.get("sort_operation_id") or ""),
                )
                should_replace = (sort_timestamp, sort_operation_id) > existing_sort_key
            if not should_replace:
                return
            if not isinstance(self._store, ConditionalDocumentStore):
                raise TypeError("source_sync_operation_index_store_unavailable")
            if self._store.replace_if_match(expected=existing, replacement=record):
                return
        raise RuntimeError("source_sync_operation_index_concurrency_conflict")

    def replace_operation_if_match(
        self,
        *,
        expected: WorkspaceOperation,
        replacement: WorkspaceOperation,
    ) -> bool:
        return self._replace_if_match(
            expected=expected,
            replacement=replacement,
            partition_key=_partition(expected.tenant_id, _ENTITY_OPERATION),
            row_key=expected.operation_id,
        )

    def claim_operation_if_queued(
        self,
        *,
        expected: WorkspaceOperation,
        replacement: WorkspaceOperation,
    ) -> bool:
        partition_key = _partition(expected.tenant_id, _ENTITY_OPERATION)
        return self._replace_if_match(
            expected=expected,
            replacement=replacement,
            partition_key=partition_key,
            row_key=expected.operation_id,
        )

    def get_operation(self, *, tenant_id: str, operation_id: str) -> WorkspaceOperation | None:
        return self._get(
            _partition(tenant_id, _ENTITY_OPERATION),
            operation_id,
            WorkspaceOperation,
        )

    def delete_operation(self, *, tenant_id: str, operation_id: str) -> None:
        self._store.delete(_partition(tenant_id, _ENTITY_OPERATION), operation_id)

    def delete_operations_for_workspace(self, *, tenant_id: str, workspace_id: str) -> int:
        deleted = 0
        for operation in self.list_operations(tenant_id=tenant_id):
            if operation.workspace_id != workspace_id:
                continue
            self.delete_operation(tenant_id=tenant_id, operation_id=operation.operation_id)
            deleted += 1
        return deleted

    # --- Document references ---

    def put_document_ref(self, ref: WorkspaceDocumentReference) -> WorkspaceDocumentReference:
        self._put(
            _partition(ref.tenant_id, _ENTITY_DOCUMENT),
            f"{ref.workspace_id}:{ref.document_id}",
            ref,
        )
        # Secondary index by path identity for idempotent sync lookups.
        path_row = f"path:{ref.workspace_id}:{ref.source_id}:{ref.source_path}"
        self._store.put(
            DocumentRecord(
                partition_key=_partition(ref.tenant_id, _ENTITY_DOCUMENT),
                row_key=path_row,
                data={"document_id": ref.document_id, "content_hash": ref.content_hash},
            )
        )
        if (
            ref.materialization_ownership is not None
            and ref.materialization_ownership.ownership_mode
            is KnowledgeMaterializationOwnershipModeV1.CONNECTED_SOURCE
        ):
            self.put_document_ownership_index_entry(
                WorkspaceDocumentOwnershipIndexEntryV1.for_reference(ref)
            )
        return ref

    def put_document_ownership_index_entry(
        self,
        entry: WorkspaceDocumentOwnershipIndexEntryV1,
    ) -> WorkspaceDocumentOwnershipIndexEntryV1:
        record = entry.to_document()
        existing = self._store.get(record.partition_key, record.row_key)
        if existing is not None:
            parsed = parse_index_entry(existing)
            same_identity = (
                parsed.tenant_id == entry.tenant_id
                and parsed.workspace_id == entry.workspace_id
                and parsed.source_id == entry.source_id
                and parsed.indexed_source_binding_id == entry.indexed_source_binding_id
                and parsed.knowledge_source_binding_ref
                == entry.knowledge_source_binding_ref
                and parsed.document_id == entry.document_id
            )
            if not same_identity:
                raise DocumentOwnershipIndexError(
                    "document_ownership_index_conflict"
                )
            if parsed != entry:
                if not isinstance(self._store, ConditionalDocumentStore):
                    raise DocumentOwnershipIndexError(
                        "document_ownership_index_store_unavailable"
                    )
                if not self._store.replace_if_match(
                    expected=existing,
                    replacement=record,
                ):
                    retry = self._store.get(record.partition_key, record.row_key)
                    if retry is None or parse_index_entry(retry) != entry:
                        raise DocumentOwnershipIndexError(
                            "document_ownership_index_conflict"
                        )
            return entry
        if not isinstance(self._store, ConditionalDocumentStore):
            raise DocumentOwnershipIndexError("document_ownership_index_store_unavailable")
        if not self._store.put_if_absent(record):
            retry = self._store.get(record.partition_key, record.row_key)
            if retry is None or parse_index_entry(retry) != entry:
                raise DocumentOwnershipIndexError(
                    "document_ownership_index_conflict"
                )
        return entry

    def repair_document_ownership_index_entry(
        self,
        reference: WorkspaceDocumentReference,
    ) -> WorkspaceDocumentOwnershipIndexEntryV1:
        try:
            entry = WorkspaceDocumentOwnershipIndexEntryV1.for_reference(reference)
        except ValueError as exc:
            raise DocumentOwnershipIndexError(str(exc)) from exc
        return self.put_document_ownership_index_entry(entry)

    def delete_document_ownership_index_entry(
        self,
        entry: WorkspaceDocumentOwnershipIndexEntryV1,
    ) -> bool:
        """Delete one validated derived ownership row, idempotently."""
        record = self._store.get(
            ownership_index_partition(entry.tenant_id),
            entry.row_key,
        )
        if record is None:
            return False
        if parse_index_entry(record) != entry:
            raise DocumentOwnershipIndexError(
                "document_ownership_index_delete_conflict"
            )
        self._store.delete(record.partition_key, record.row_key)
        return True

    def list_document_refs_by_materialization_owner(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        source_id: str,
        indexed_source_binding_id: str,
        knowledge_source_binding_ref: str,
        limit: int,
        cursor: str | None = None,
    ) -> DocumentReferenceOwnershipPageV1:
        scope = {
            "tenant_id": tenant_id,
            "workspace_id": workspace_id,
            "source_id": source_id,
            "indexed_source_binding_id": indexed_source_binding_id,
            "knowledge_source_binding_ref": knowledge_source_binding_ref,
        }
        prefix_entry = WorkspaceDocumentOwnershipIndexEntryV1(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            source_id=source_id,
            indexed_source_binding_id=indexed_source_binding_id,
            knowledge_source_binding_ref=knowledge_source_binding_ref,
            document_id="scope-probe",
            reference_fingerprint="0" * 64,
            indexed_at=datetime.now(UTC),
        )
        page = self._store.query(
            ownership_index_partition(tenant_id),
            limit=limit,
            row_key_prefix=prefix_entry.scope_prefix,
            cursor=cursor,
        )
        references: list[WorkspaceDocumentReference] = []
        orphan_entries: list[WorkspaceDocumentOwnershipIndexEntryV1] = []
        for record in page.documents:
            entry = parse_index_entry(record)
            if any(
                getattr(entry, key) != value
                for key, value in scope.items()
            ):
                raise DocumentOwnershipIndexError(
                    "document_ownership_index_scope_mismatch"
                )
            reference = self.get_document_ref(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                document_id=entry.document_id,
            )
            if reference is None:
                orphan_entries.append(entry)
                continue
            ownership = reference.materialization_ownership
            if (
                ownership is None
                or ownership.ownership_mode
                is not KnowledgeMaterializationOwnershipModeV1.CONNECTED_SOURCE
                or (
                    ownership.tenant_id,
                    ownership.workspace_id,
                    ownership.source_id,
                    ownership.indexed_source_binding_id,
                    ownership.knowledge_source_binding_ref,
                )
                != (
                    tenant_id,
                    workspace_id,
                    source_id,
                    indexed_source_binding_id,
                    knowledge_source_binding_ref,
                )
                or reference_fingerprint(reference) != entry.reference_fingerprint
            ):
                raise DocumentOwnershipIndexError(
                    "document_ownership_index_reference_mismatch"
                )
            references.append(reference)
        return DocumentReferenceOwnershipPageV1(
            references=tuple(references),
            orphan_index_entries=tuple(orphan_entries),
            next_cursor=page.next_cursor,
        )

    def put_connected_source_recovery_ownership_index_entry(
        self,
        entry: ConnectedSourceRecoveryOwnershipIndexEntryV1,
    ) -> ConnectedSourceRecoveryOwnershipIndexEntryV1:
        record = entry.to_document()
        existing = self._store.get(record.partition_key, record.row_key)
        if existing is not None:
            parsed = parse_recovery_ownership_index_entry(existing)
            same_identity = (
                parsed.tenant_id == entry.tenant_id
                and parsed.workspace_id == entry.workspace_id
                and parsed.source_id == entry.source_id
                and parsed.indexed_source_binding_id
                == entry.indexed_source_binding_id
                and parsed.knowledge_source_binding_ref
                == entry.knowledge_source_binding_ref
                and parsed.record_kind == entry.record_kind
                and parsed.operation_id == entry.operation_id
                and parsed.delivery_id == entry.delivery_id
                and parsed.document_id == entry.document_id
                and parsed.canonical_partition_key == entry.canonical_partition_key
                and parsed.canonical_row_key == entry.canonical_row_key
            )
            if not same_identity:
                raise ConnectedSourceRecoveryOwnershipIndexError(
                    "recovery_ownership_index_conflict"
                )
            if parsed == entry:
                return entry
            if not isinstance(self._store, ConditionalDocumentStore):
                raise ConnectedSourceRecoveryOwnershipIndexError(
                    "recovery_ownership_index_store_unavailable"
                )
            if not self._store.replace_if_match(
                expected=existing,
                replacement=record,
            ):
                retry = self._store.get(record.partition_key, record.row_key)
                if (
                    retry is None
                    or parse_recovery_ownership_index_entry(retry) != entry
                ):
                    raise ConnectedSourceRecoveryOwnershipIndexError(
                        "recovery_ownership_index_conflict"
                    )
            return entry
        if not isinstance(self._store, ConditionalDocumentStore):
            raise ConnectedSourceRecoveryOwnershipIndexError(
                "recovery_ownership_index_store_unavailable"
            )
        if not self._store.put_if_absent(record):
            retry = self._store.get(record.partition_key, record.row_key)
            if retry is None or parse_recovery_ownership_index_entry(retry) != entry:
                raise ConnectedSourceRecoveryOwnershipIndexError(
                    "recovery_ownership_index_conflict"
                )
        return entry

    def repair_connected_source_recovery_ownership_index_entry(
        self,
        entry: ConnectedSourceRecoveryOwnershipIndexEntryV1,
    ) -> ConnectedSourceRecoveryOwnershipIndexEntryV1:
        """Idempotent repair from a known complete-ownership canonical record."""
        return self.put_connected_source_recovery_ownership_index_entry(entry)

    def delete_connected_source_recovery_ownership_index_entry(
        self,
        entry: ConnectedSourceRecoveryOwnershipIndexEntryV1,
    ) -> bool:
        record = self._store.get(
            recovery_ownership_index_partition(entry.tenant_id),
            entry.row_key,
        )
        if record is None:
            return False
        if parse_recovery_ownership_index_entry(record) != entry:
            raise ConnectedSourceRecoveryOwnershipIndexError(
                "recovery_ownership_index_delete_conflict"
            )
        self._store.delete(record.partition_key, record.row_key)
        return True

    def list_connected_source_recovery_records_by_owner(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        source_id: str,
        indexed_source_binding_id: str,
        knowledge_source_binding_ref: str,
        record_kind: RecoveryRecordKindV1,
        limit: int,
        cursor: str | None = None,
    ) -> ConnectedSourceRecoveryOwnershipPageV1:
        validate_document_query_limit(limit)
        scope = (
            tenant_id,
            workspace_id,
            source_id,
            indexed_source_binding_id,
            knowledge_source_binding_ref,
        )
        prefix = recovery_ownership_scope_prefix(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            source_id=source_id,
            indexed_source_binding_id=indexed_source_binding_id,
            knowledge_source_binding_ref=knowledge_source_binding_ref,
            record_kind=record_kind,
        )
        page = self._store.query(
            recovery_ownership_index_partition(tenant_id),
            limit=limit,
            row_key_prefix=prefix,
            cursor=cursor,
        )
        index_entries: list[ConnectedSourceRecoveryOwnershipIndexEntryV1] = []
        orphan_entries: list[ConnectedSourceRecoveryOwnershipIndexEntryV1] = []
        for record in page.documents:
            entry = parse_recovery_ownership_index_entry(record)
            if entry.ownership_scope != scope or entry.record_kind != record_kind:
                raise ConnectedSourceRecoveryOwnershipIndexError(
                    "recovery_ownership_index_scope_mismatch"
                )
            canonical = self._store.get(
                entry.canonical_partition_key,
                entry.canonical_row_key,
            )
            if canonical is None:
                orphan_entries.append(entry)
                continue
            if not self._recovery_canonical_matches_index(entry, canonical):
                raise ConnectedSourceRecoveryOwnershipIndexError(
                    "recovery_ownership_index_reference_mismatch"
                )
            index_entries.append(entry)
        return ConnectedSourceRecoveryOwnershipPageV1(
            index_entries=tuple(index_entries),
            orphan_index_entries=tuple(orphan_entries),
            next_cursor=page.next_cursor,
        )

    def _recovery_canonical_matches_index(
        self,
        entry: ConnectedSourceRecoveryOwnershipIndexEntryV1,
        canonical: DocumentRecord,
    ) -> bool:
        if (
            canonical.partition_key != entry.canonical_partition_key
            or canonical.row_key != entry.canonical_row_key
        ):
            return False
        if entry.record_kind is RecoveryRecordKindV1.ENQUEUE_INTENT:
            try:
                item = ConnectedSourceSyncEnqueueIntent.model_validate(
                    dict(canonical.data), strict=False
                )
            except (TypeError, ValueError):
                return False
            return (
                item.ownership_classification == "COMPLETE_OWNERSHIP"
                and (
                    item.tenant_id,
                    item.workspace_id,
                    item.source_id,
                    item.indexed_source_binding_id,
                    item.knowledge_source_binding_ref,
                )
                == entry.ownership_scope
                and item.operation_id == entry.operation_id
                and canonical_record_fingerprint(item) == entry.canonical_fingerprint
            )
        if entry.record_kind is RecoveryRecordKindV1.DELIVERY_ACCOUNTING:
            try:
                item = ConnectedSourceOperationDeliveryAccounting.model_validate(
                    dict(canonical.data), strict=False
                )
            except (TypeError, ValueError):
                return False
            return (
                item.ownership_classification == "COMPLETE_OWNERSHIP"
                and (
                    item.tenant_id,
                    item.workspace_id,
                    item.source_id,
                    item.indexed_source_binding_id,
                    item.knowledge_source_binding_ref,
                )
                == entry.ownership_scope
                and item.operation_id == entry.operation_id
                and item.delivery_id == entry.delivery_id
                and canonical_record_fingerprint(item) == entry.canonical_fingerprint
            )
        if entry.record_kind is RecoveryRecordKindV1.INDEX_RECEIPT:
            from local_workspace_application.workspaces.document_indexing import (
                _WorkspaceDocumentIndexReceipt,
            )

            try:
                item = _WorkspaceDocumentIndexReceipt.model_validate(
                    dict(canonical.data), strict=False
                )
            except (TypeError, ValueError):
                return False
            ownership = item.materialization_ownership
            if ownership is None:
                return False
            return (
                ownership.ownership_mode
                is KnowledgeMaterializationOwnershipModeV1.CONNECTED_SOURCE
                and (
                    ownership.tenant_id,
                    ownership.workspace_id,
                    ownership.source_id,
                    ownership.indexed_source_binding_id,
                    ownership.knowledge_source_binding_ref,
                )
                == entry.ownership_scope
                and item.operation_id == entry.operation_id
                and item.document_id == entry.document_id
                and canonical_record_fingerprint(item) == entry.canonical_fingerprint
            )
        return False

    def get_document_ref_by_path(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        source_id: str,
        source_path: str,
    ) -> WorkspaceDocumentReference | None:
        path_row = f"path:{workspace_id}:{source_id}:{source_path}"
        index = self._store.get(_partition(tenant_id, _ENTITY_DOCUMENT), path_row)
        if index is None:
            return None
        document_id = str(index.data.get("document_id") or "").strip()
        if not document_id:
            return None
        return self._get(
            _partition(tenant_id, _ENTITY_DOCUMENT),
            f"{workspace_id}:{document_id}",
            WorkspaceDocumentReference,
        )

    def list_document_refs(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
    ) -> list[WorkspaceDocumentReference]:
        result = self._store.query(
            _partition(tenant_id, _ENTITY_DOCUMENT),
            limit=2000,
            row_key_prefix=f"{workspace_id}:",
        )
        refs: list[WorkspaceDocumentReference] = []
        for doc in result.documents:
            if (
                str(doc.row_key).startswith(f"{workspace_id}:")
                and not str(doc.row_key).startswith(f"{workspace_id}:path:")
                and "document_id" in doc.data
                and "source_path" in doc.data
            ):
                try:
                    refs.append(
                        WorkspaceDocumentReference.model_validate(
                            dict(doc.data),
                            strict=False,
                        )
                    )
                except ValidationError:
                    continue
        return refs

    def get_document_ref(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        document_id: str,
    ) -> WorkspaceDocumentReference | None:
        return self._get(
            _partition(tenant_id, _ENTITY_DOCUMENT),
            f"{workspace_id}:{document_id}",
            WorkspaceDocumentReference,
        )

    def delete_document_refs_for_workspace(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
    ) -> int:
        """Remove primary refs and path-index rows for one workspace."""
        partition = _partition(tenant_id, _ENTITY_DOCUMENT)
        deleted = 0
        refs = self.list_document_refs(tenant_id=tenant_id, workspace_id=workspace_id)
        for ref in refs:
            self._store.delete(partition, f"{workspace_id}:{ref.document_id}")
            path_row = f"path:{workspace_id}:{ref.source_id}:{ref.source_path}"
            self._store.delete(partition, path_row)
            deleted += 1
        # Sweep any leftover path-index rows for this workspace.
        path_prefix = f"path:{workspace_id}:"
        leftover = self._store.query(partition, limit=2000, row_key_prefix=path_prefix)
        for doc in leftover.documents:
            self._store.delete(partition, doc.row_key)
            deleted += 1
        return deleted

    def list_operations(self, *, tenant_id: str) -> list[WorkspaceOperation]:
        return self._list(_partition(tenant_id, _ENTITY_OPERATION), WorkspaceOperation)

    def list_source_sync_operations_page(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        source_id: str,
        limit: int,
        cursor: str | None = None,
    ) -> DocumentQueryPageV1:
        """Read the exact latest-operation pointer for one source."""
        validate_document_query_limit(limit)
        return self._store.query(
            _partition(tenant_id, _ENTITY_SOURCE_SYNC_OPERATION_INDEX),
            limit=limit,
            row_key_prefix=_source_sync_operation_index_row_key(
                workspace_id=workspace_id,
                source_id=source_id,
            ),
            cursor=cursor,
        )

    def list_ingestion_operations(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        statuses: set[WorkspaceOperationStatus] | None = None,
    ) -> list[WorkspaceOperation]:
        items = [
            op
            for op in self.list_operations(tenant_id=tenant_id)
            if op.workspace_id == workspace_id
            and op.operation_type is WorkspaceOperationType.KNOWLEDGE_INGESTION
            and (statuses is None or op.status in statuses)
        ]
        return sorted(items, key=lambda item: item.created_at or item.operation_id)

    # --- Knowledge Input ---

    def put_knowledge_input(self, knowledge_input: KnowledgeInput) -> KnowledgeInput:
        self._put(
            _partition(knowledge_input.tenant_id, _ENTITY_KNOWLEDGE_INPUT),
            f"{knowledge_input.workspace_id}:{knowledge_input.input_id}",
            knowledge_input,
        )
        return knowledge_input

    def get_knowledge_input(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        input_id: str,
    ) -> KnowledgeInput | None:
        return self._get(
            _partition(tenant_id, _ENTITY_KNOWLEDGE_INPUT),
            f"{workspace_id}:{input_id}",
            KnowledgeInput,
        )

    def list_knowledge_inputs(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
    ) -> list[KnowledgeInput]:
        result = self._store.query(
            _partition(tenant_id, _ENTITY_KNOWLEDGE_INPUT),
            limit=500,
            row_key_prefix=f"{workspace_id}:",
        )
        items = [KnowledgeInput.model_validate(dict(doc.data)) for doc in result.documents]
        return sorted(items, key=lambda item: item.created_at)

    def delete_knowledge_inputs_for_workspace(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
    ) -> int:
        deleted = 0
        for knowledge_input in self.list_knowledge_inputs(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
        ):
            self._store.delete(
                _partition(tenant_id, _ENTITY_KNOWLEDGE_INPUT),
                f"{workspace_id}:{knowledge_input.input_id}",
            )
            deleted += 1
        return deleted

    def find_active_sync_operation(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        source_id: str,
    ) -> WorkspaceOperation | None:
        active = {
            WorkspaceOperationStatus.QUEUED,
            WorkspaceOperationStatus.RUNNING,
        }
        page = self.list_source_sync_operations_page(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            source_id=source_id,
            limit=1,
        )
        for record in page.documents:
            data = dict(record.data)
            if any(
                data.get(field) != expected
                for field, expected in (
                    ("tenant_id", tenant_id),
                    ("workspace_id", workspace_id),
                    ("source_id", source_id),
                    ("operation_type", WorkspaceOperationType.SOURCE_SYNC.value),
                )
            ):
                continue
            operation_id = data.get("operation_id")
            if not isinstance(operation_id, str) or not operation_id.strip():
                continue
            operation = self.get_operation(
                tenant_id=tenant_id,
                operation_id=operation_id,
            )
            if (
                operation is not None
                and operation.tenant_id == tenant_id
                and operation.workspace_id == workspace_id
                and operation.source_id == source_id
                and operation.operation_type is WorkspaceOperationType.SOURCE_SYNC
                and operation.status in active
            ):
                return operation
        return None

    def mark_running_operations_failed_for_tenant(
        self,
        *,
        tenant_id: str,
        error: str = "interrupted_by_host_restart",
    ) -> int:
        from datetime import UTC, datetime

        now = datetime.now(UTC)
        recovered = 0
        for operation in self.list_operations(tenant_id=tenant_id):
            if operation.status is not WorkspaceOperationStatus.RUNNING:
                continue
            self.put_operation(
                operation.model_copy(
                    update={
                        "status": WorkspaceOperationStatus.FAILED,
                        "error": error,
                        "completed_at": now,
                    }
                )
            )
            recovered += 1
        return recovered

    # --- Managed file ---

    def put_managed_file(self, managed_file: ManagedFileObject) -> ManagedFileObject:
        self._put(
            _partition(managed_file.tenant_id, _ENTITY_MANAGED_FILE),
            f"{managed_file.workspace_id}:{managed_file.input_id}",
            managed_file,
        )
        return managed_file

    def get_managed_file(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        input_id: str,
    ) -> ManagedFileObject | None:
        return self._get(
            _partition(tenant_id, _ENTITY_MANAGED_FILE),
            f"{workspace_id}:{input_id}",
            ManagedFileObject,
        )

    def list_managed_files(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
    ) -> list[ManagedFileObject]:
        result = self._store.query(
            _partition(tenant_id, _ENTITY_MANAGED_FILE),
            limit=2000,
            row_key_prefix=f"{workspace_id}:",
        )
        items = [ManagedFileObject.model_validate(dict(doc.data)) for doc in result.documents]
        return sorted(items, key=lambda item: item.created_at)

    def delete_managed_file(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        input_id: str,
    ) -> None:
        self._store.delete(
            _partition(tenant_id, _ENTITY_MANAGED_FILE),
            f"{workspace_id}:{input_id}",
        )

    def delete_managed_files_for_workspace(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
    ) -> int:
        deleted = 0
        for managed_file in self.list_managed_files(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
        ):
            self.delete_managed_file(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                input_id=managed_file.input_id,
            )
            deleted += 1
        return deleted

    # --- Web URL locators ---

    def put_web_url_locator(self, locator: WebUrlSourceLocator) -> WebUrlSourceLocator:
        self._put(
            _partition(locator.tenant_id, _ENTITY_WEB_URL_LOCATOR),
            f"{locator.workspace_id}:{locator.requested_url_fingerprint}",
            locator,
        )
        return locator

    def get_web_url_locator(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        requested_url_fingerprint: str,
    ) -> WebUrlSourceLocator | None:
        return self._get(
            _partition(tenant_id, _ENTITY_WEB_URL_LOCATOR),
            f"{workspace_id}:{requested_url_fingerprint}",
            WebUrlSourceLocator,
        )

    def list_web_url_locators(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
    ) -> list[WebUrlSourceLocator]:
        result = self._store.query(
            _partition(tenant_id, _ENTITY_WEB_URL_LOCATOR),
            limit=2000,
            row_key_prefix=f"{workspace_id}:",
        )
        items: list[WebUrlSourceLocator] = []
        for doc in result.documents:
            try:
                items.append(WebUrlSourceLocator.model_validate(dict(doc.data)))
            except Exception as exc:
                raise ValueError("web_url_locator_malformed") from exc
        return sorted(items, key=lambda item: item.created_at)

    def delete_web_url_locator(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        requested_url_fingerprint: str,
    ) -> None:
        self._store.delete(
            _partition(tenant_id, _ENTITY_WEB_URL_LOCATOR),
            f"{workspace_id}:{requested_url_fingerprint}",
        )

    def delete_web_url_locators_for_workspace(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
    ) -> int:
        deleted = 0
        for locator in self.list_web_url_locators(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
        ):
            self.delete_web_url_locator(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                requested_url_fingerprint=locator.requested_url_fingerprint,
            )
            deleted += 1
        return deleted

    # --- Intake batch ---

    def put_intake_batch(self, batch: IntakeBatch) -> IntakeBatch:
        self._put(
            _partition(batch.tenant_id, _ENTITY_INTAKE_BATCH),
            f"{batch.workspace_id}:{batch.batch_id}",
            batch,
        )
        return batch

    def get_intake_batch(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        batch_id: str,
    ) -> IntakeBatch | None:
        return self._get(
            _partition(tenant_id, _ENTITY_INTAKE_BATCH),
            f"{workspace_id}:{batch_id}",
            IntakeBatch,
        )

    def list_intake_batches(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
    ) -> list[IntakeBatch]:
        result = self._store.query(
            _partition(tenant_id, _ENTITY_INTAKE_BATCH),
            limit=500,
            row_key_prefix=f"{workspace_id}:",
        )
        items = [IntakeBatch.model_validate(dict(doc.data)) for doc in result.documents]
        return sorted(items, key=lambda item: item.created_at)

    def delete_intake_batches_for_workspace(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
    ) -> int:
        deleted = 0
        for batch in self.list_intake_batches(tenant_id=tenant_id, workspace_id=workspace_id):
            self._store.delete(
                _partition(tenant_id, _ENTITY_INTAKE_BATCH),
                f"{workspace_id}:{batch.batch_id}",
            )
            deleted += 1
        return deleted

    # --- Active knowledge-ingestion locators ---

    def put_active_ingestion_locator(
        self,
        locator: ActiveKnowledgeIngestionLocator,
    ) -> ActiveKnowledgeIngestionLocator:
        self._put(
            _ACTIVE_KNOWLEDGE_INGESTION_PARTITION,
            locator.operation_id,
            locator,
        )
        return locator

    def scan_active_ingestion_locators(
        self,
        *,
        limit: int = 5000,
    ) -> ActiveKnowledgeIngestionLocatorScan:
        result = self._store.query(_ACTIVE_KNOWLEDGE_INGESTION_PARTITION, limit=limit)
        locators: list[ActiveKnowledgeIngestionLocator] = []
        malformed_seen = 0
        malformed_removed = 0
        for doc in result.documents:
            try:
                locators.append(
                    ActiveKnowledgeIngestionLocator.model_validate(dict(doc.data))
                )
            except Exception:  # noqa: BLE001 - isolate malformed locator rows
                malformed_seen += 1
                try:
                    self._store.delete(_ACTIVE_KNOWLEDGE_INGESTION_PARTITION, doc.row_key)
                    malformed_removed += 1
                except Exception:  # noqa: BLE001, S112 - continue scan after delete failure
                    continue
        locators.sort(key=lambda item: (item.created_at, item.operation_id))
        return ActiveKnowledgeIngestionLocatorScan(
            locators=tuple(locators),
            malformed_seen=malformed_seen,
            malformed_removed=malformed_removed,
        )

    def list_active_ingestion_locators(
        self,
        *,
        limit: int = 5000,
    ) -> list[ActiveKnowledgeIngestionLocator]:
        return list(self.scan_active_ingestion_locators(limit=limit).locators)

    def delete_active_ingestion_locator(self, operation_id: str) -> None:
        self._store.delete(_ACTIVE_KNOWLEDGE_INGESTION_PARTITION, operation_id)

    # --- Workspace Knowledge Configuration ---

    def _require_conditional_store(self) -> ConditionalDocumentStore:
        if not isinstance(self._store, ConditionalDocumentStore):
            raise WorkspaceKnowledgeConfigurationRepositoryError(
                "configuration_conditional_store_required"
            )
        return self._store

    def _put_if_absent(self, model: BaseModel, *, partition_key: str, row_key: str) -> bool:
        store = self._require_conditional_store()
        return store.put_if_absent(
            _to_document_record(model, partition_key=partition_key, row_key=row_key)
        )

    def _replace_if_match(
        self,
        *,
        expected: BaseModel,
        replacement: BaseModel,
        partition_key: str,
        row_key: str,
    ) -> bool:
        store = self._require_conditional_store()
        return store.replace_if_match(
            expected=_to_document_record(expected, partition_key=partition_key, row_key=row_key),
            replacement=_to_document_record(
                replacement,
                partition_key=partition_key,
                row_key=row_key,
            ),
        )

    def _delete_if_match(self, model: BaseModel, *, partition_key: str, row_key: str) -> bool:
        store = self._require_conditional_store()
        return store.delete_if_match(
            expected=_to_document_record(model, partition_key=partition_key, row_key=row_key)
        )

    def _list_revision_versions(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        partition_key: str,
        model_type: type[T],
        entity_id_field: str | None = None,
        fixed_entity_id: str | None = None,
        sort_key: Callable[[T], tuple[object, ...]],
    ) -> list[T]:
        result = self._store.query(
            partition_key,
            limit=_KNOWLEDGE_CONFIGURATION_REVISION_SCAN_LIMIT + 1,
            row_key_prefix=f"{workspace_id}:",
        )
        if len(result.documents) > _KNOWLEDGE_CONFIGURATION_REVISION_SCAN_LIMIT:
            raise WorkspaceKnowledgeConfigurationRepositoryError(
                "knowledge_configuration_revision_scan_limit_exceeded"
            )
        items: list[T] = []
        for doc in result.documents:
            model = cast(T, model_type.model_validate(dict(doc.data)))
            model_data = cast(Any, model)
            row_workspace_id, row_entity_id, row_revision = _parse_revision_row_key(doc.row_key)
            if row_workspace_id != workspace_id:
                raise ValueError("knowledge_configuration_record_identity_mismatch")
            if fixed_entity_id is not None:
                model_entity_id = fixed_entity_id
            else:
                assert entity_id_field is not None
                model_entity_id = getattr(model, entity_id_field)
            _assert_record_identity(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                model_tenant_id=model_data.tenant_id,
                model_workspace_id=model_data.workspace_id,
                entity_id=row_entity_id,
                model_entity_id=model_entity_id,
                revision=row_revision,
                model_revision=model_data.effective_revision,
            )
            items.append(model)
        return sorted(items, key=sort_key)

    def get_knowledge_configuration_head(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
    ) -> WorkspaceKnowledgeConfigurationHead | None:
        record = self._store.get(
            _partition(tenant_id, _ENTITY_KNOWLEDGE_CONFIGURATION_HEAD),
            workspace_id,
        )
        if record is None:
            return None
        head = WorkspaceKnowledgeConfigurationHead.model_validate(dict(record.data))
        _assert_record_identity(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            model_tenant_id=head.tenant_id,
            model_workspace_id=head.workspace_id,
        )
        return head

    def put_knowledge_configuration_head_if_absent(
        self,
        head: WorkspaceKnowledgeConfigurationHead,
    ) -> bool:
        partition_key = _partition(head.tenant_id, _ENTITY_KNOWLEDGE_CONFIGURATION_HEAD)
        return self._put_if_absent(head, partition_key=partition_key, row_key=head.workspace_id)

    def replace_knowledge_configuration_head_if_match(
        self,
        *,
        expected: WorkspaceKnowledgeConfigurationHead,
        replacement: WorkspaceKnowledgeConfigurationHead,
    ) -> bool:
        if (
            expected.tenant_id != replacement.tenant_id
            or expected.workspace_id != replacement.workspace_id
        ):
            raise ValueError("knowledge_configuration_conditional_key_mismatch")
        partition_key = _partition(expected.tenant_id, _ENTITY_KNOWLEDGE_CONFIGURATION_HEAD)
        row_key = expected.workspace_id
        return self._replace_if_match(
            expected=expected,
            replacement=replacement,
            partition_key=partition_key,
            row_key=row_key,
        )

    def get_knowledge_configuration_mutation(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        operation: WorkspaceKnowledgeMutationOperationV1,
        idempotency_key_hash: str,
    ) -> WorkspaceKnowledgeMutationRecord | None:
        _validate_idempotency_key_hash(idempotency_key_hash)
        row_key = _mutation_row_key(
            workspace_id=workspace_id,
            operation=operation.value,
            idempotency_key_hash=idempotency_key_hash,
        )
        record = self._store.get(
            _partition(tenant_id, _ENTITY_KNOWLEDGE_CONFIGURATION_MUTATION),
            row_key,
        )
        if record is None:
            return None
        mutation = WorkspaceKnowledgeMutationRecord.model_validate(dict(record.data))
        _assert_record_identity(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            model_tenant_id=mutation.tenant_id,
            model_workspace_id=mutation.workspace_id,
            entity_id=operation.value,
            model_entity_id=mutation.operation.value,
            revision=None,
            model_revision=None,
        )
        if mutation.idempotency_key_hash != idempotency_key_hash:
            raise ValueError("knowledge_configuration_record_identity_mismatch")
        return mutation

    def put_knowledge_configuration_mutation_if_absent(
        self,
        mutation: WorkspaceKnowledgeMutationRecord,
    ) -> bool:
        partition_key = _partition(mutation.tenant_id, _ENTITY_KNOWLEDGE_CONFIGURATION_MUTATION)
        row_key = _mutation_row_key(
            workspace_id=mutation.workspace_id,
            operation=mutation.operation.value,
            idempotency_key_hash=mutation.idempotency_key_hash,
        )
        return self._put_if_absent(mutation, partition_key=partition_key, row_key=row_key)

    def replace_knowledge_configuration_mutation_if_match(
        self,
        *,
        expected: WorkspaceKnowledgeMutationRecord,
        replacement: WorkspaceKnowledgeMutationRecord,
    ) -> bool:
        if (
            expected.tenant_id != replacement.tenant_id
            or expected.workspace_id != replacement.workspace_id
            or expected.operation != replacement.operation
            or expected.idempotency_key_hash != replacement.idempotency_key_hash
        ):
            raise ValueError("knowledge_configuration_conditional_key_mismatch")
        partition_key = _partition(expected.tenant_id, _ENTITY_KNOWLEDGE_CONFIGURATION_MUTATION)
        row_key = _mutation_row_key(
            workspace_id=expected.workspace_id,
            operation=expected.operation.value,
            idempotency_key_hash=expected.idempotency_key_hash,
        )
        return self._replace_if_match(
            expected=expected,
            replacement=replacement,
            partition_key=partition_key,
            row_key=row_key,
        )

    def list_knowledge_configuration_mutations(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
    ) -> list[WorkspaceKnowledgeMutationRecord]:
        partition_key = _partition(tenant_id, _ENTITY_KNOWLEDGE_CONFIGURATION_MUTATION)
        result = self._store.query(
            partition_key,
            limit=_KNOWLEDGE_CONFIGURATION_MUTATION_SCAN_PROBE_LIMIT,
            row_key_prefix=f"{workspace_id}:",
        )
        if len(result.documents) > _KNOWLEDGE_CONFIGURATION_MUTATION_SCAN_LIMIT:
            raise WorkspaceKnowledgeConfigurationRepositoryError(
                "knowledge_configuration_mutation_scan_limit_exceeded"
            )
        items: list[WorkspaceKnowledgeMutationRecord] = []
        for doc in result.documents:
            row_workspace_id, row_operation, row_hash = _parse_mutation_row_key(doc.row_key)
            if row_workspace_id != workspace_id:
                raise ValueError("knowledge_configuration_record_identity_mismatch")
            mutation = WorkspaceKnowledgeMutationRecord.model_validate(dict(doc.data))
            _assert_record_identity(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                model_tenant_id=mutation.tenant_id,
                model_workspace_id=mutation.workspace_id,
                entity_id=row_operation,
                model_entity_id=mutation.operation.value,
            )
            if mutation.idempotency_key_hash != row_hash:
                raise ValueError("knowledge_configuration_record_identity_mismatch")
            items.append(mutation)
        return sorted(
            items,
            key=lambda item: (
                item.created_at,
                item.operation.value,
                item.idempotency_key_hash,
                item.mutation_id,
            ),
        )

    def find_knowledge_configuration_mutation_by_id(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        mutation_id: str,
    ) -> WorkspaceKnowledgeMutationRecord | None:
        matches = [
            mutation
            for mutation in self.list_knowledge_configuration_mutations(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
            )
            if mutation.mutation_id == mutation_id
        ]
        if not matches:
            return None
        if len(matches) > 1:
            raise WorkspaceKnowledgeConfigurationRepositoryError(
                "knowledge_configuration_mutation_id_not_unique"
            )
        return matches[0]

    def put_knowledge_connection_attachment_version_if_absent(
        self,
        attachment: WorkspaceConnectionAttachment,
    ) -> bool:
        partition_key = _partition(
            attachment.tenant_id,
            _ENTITY_KNOWLEDGE_CONFIGURATION_CONNECTION_ATTACHMENT,
        )
        row_key = _revision_row_key(
            workspace_id=attachment.workspace_id,
            entity_id=attachment.attachment_id,
            revision=attachment.effective_revision,
        )
        return self._put_if_absent(attachment, partition_key=partition_key, row_key=row_key)

    def get_knowledge_connection_attachment_version(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        attachment_id: str,
        effective_revision: int,
    ) -> WorkspaceConnectionAttachment | None:
        partition_key = _partition(
            tenant_id,
            _ENTITY_KNOWLEDGE_CONFIGURATION_CONNECTION_ATTACHMENT,
        )
        row_key = _revision_row_key(
            workspace_id=workspace_id,
            entity_id=attachment_id,
            revision=effective_revision,
        )
        record = self._store.get(partition_key, row_key)
        if record is None:
            return None
        attachment = WorkspaceConnectionAttachment.model_validate(dict(record.data))
        _assert_record_identity(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            model_tenant_id=attachment.tenant_id,
            model_workspace_id=attachment.workspace_id,
            entity_id=attachment_id,
            model_entity_id=attachment.attachment_id,
            revision=effective_revision,
            model_revision=attachment.effective_revision,
        )
        return attachment

    def list_knowledge_connection_attachment_versions(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
    ) -> list[WorkspaceConnectionAttachment]:
        return self._list_revision_versions(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            partition_key=_partition(
                tenant_id,
                _ENTITY_KNOWLEDGE_CONFIGURATION_CONNECTION_ATTACHMENT,
            ),
            model_type=WorkspaceConnectionAttachment,
            entity_id_field="attachment_id",
            sort_key=lambda item: (item.attachment_id, item.effective_revision),
        )

    def delete_knowledge_connection_attachment_version_if_match(
        self,
        attachment: WorkspaceConnectionAttachment,
    ) -> bool:
        partition_key = _partition(
            attachment.tenant_id,
            _ENTITY_KNOWLEDGE_CONFIGURATION_CONNECTION_ATTACHMENT,
        )
        row_key = _revision_row_key(
            workspace_id=attachment.workspace_id,
            entity_id=attachment.attachment_id,
            revision=attachment.effective_revision,
        )
        return self._delete_if_match(attachment, partition_key=partition_key, row_key=row_key)

    def put_knowledge_indexed_source_version_if_absent(
        self,
        binding: WorkspaceIndexedSourceBinding,
    ) -> bool:
        partition_key = _partition(
            binding.tenant_id,
            _ENTITY_KNOWLEDGE_CONFIGURATION_INDEXED_SOURCE,
        )
        row_key = _revision_row_key(
            workspace_id=binding.workspace_id,
            entity_id=binding.indexed_source_binding_id,
            revision=binding.effective_revision,
        )
        return self._put_if_absent(binding, partition_key=partition_key, row_key=row_key)

    def get_knowledge_indexed_source_version(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        indexed_source_binding_id: str,
        effective_revision: int,
    ) -> WorkspaceIndexedSourceBinding | None:
        partition_key = _partition(
            tenant_id,
            _ENTITY_KNOWLEDGE_CONFIGURATION_INDEXED_SOURCE,
        )
        row_key = _revision_row_key(
            workspace_id=workspace_id,
            entity_id=indexed_source_binding_id,
            revision=effective_revision,
        )
        record = self._store.get(partition_key, row_key)
        if record is None:
            return None
        binding = WorkspaceIndexedSourceBinding.model_validate(dict(record.data))
        _assert_record_identity(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            model_tenant_id=binding.tenant_id,
            model_workspace_id=binding.workspace_id,
            entity_id=indexed_source_binding_id,
            model_entity_id=binding.indexed_source_binding_id,
            revision=effective_revision,
            model_revision=binding.effective_revision,
        )
        return binding

    def list_knowledge_indexed_source_versions(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
    ) -> list[WorkspaceIndexedSourceBinding]:
        return self._list_revision_versions(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            partition_key=_partition(
                tenant_id,
                _ENTITY_KNOWLEDGE_CONFIGURATION_INDEXED_SOURCE,
            ),
            model_type=WorkspaceIndexedSourceBinding,
            entity_id_field="indexed_source_binding_id",
            sort_key=lambda item: (item.indexed_source_binding_id, item.effective_revision),
        )

    def delete_knowledge_indexed_source_version_if_match(
        self,
        binding: WorkspaceIndexedSourceBinding,
    ) -> bool:
        partition_key = _partition(
            binding.tenant_id,
            _ENTITY_KNOWLEDGE_CONFIGURATION_INDEXED_SOURCE,
        )
        row_key = _revision_row_key(
            workspace_id=binding.workspace_id,
            entity_id=binding.indexed_source_binding_id,
            revision=binding.effective_revision,
        )
        return self._delete_if_match(binding, partition_key=partition_key, row_key=row_key)

    def put_knowledge_live_access_version_if_absent(
        self,
        binding: WorkspaceLiveAccessBinding,
    ) -> bool:
        partition_key = _partition(
            binding.tenant_id,
            _ENTITY_KNOWLEDGE_CONFIGURATION_LIVE_ACCESS,
        )
        row_key = _revision_row_key(
            workspace_id=binding.workspace_id,
            entity_id=binding.live_access_binding_id,
            revision=binding.effective_revision,
        )
        return self._put_if_absent(binding, partition_key=partition_key, row_key=row_key)

    def get_knowledge_live_access_version(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        live_access_binding_id: str,
        effective_revision: int,
    ) -> WorkspaceLiveAccessBinding | None:
        partition_key = _partition(
            tenant_id,
            _ENTITY_KNOWLEDGE_CONFIGURATION_LIVE_ACCESS,
        )
        row_key = _revision_row_key(
            workspace_id=workspace_id,
            entity_id=live_access_binding_id,
            revision=effective_revision,
        )
        record = self._store.get(partition_key, row_key)
        if record is None:
            return None
        binding = WorkspaceLiveAccessBinding.model_validate(dict(record.data))
        _assert_record_identity(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            model_tenant_id=binding.tenant_id,
            model_workspace_id=binding.workspace_id,
            entity_id=live_access_binding_id,
            model_entity_id=binding.live_access_binding_id,
            revision=effective_revision,
            model_revision=binding.effective_revision,
        )
        return binding

    def list_knowledge_live_access_versions(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
    ) -> list[WorkspaceLiveAccessBinding]:
        return self._list_revision_versions(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            partition_key=_partition(
                tenant_id,
                _ENTITY_KNOWLEDGE_CONFIGURATION_LIVE_ACCESS,
            ),
            model_type=WorkspaceLiveAccessBinding,
            entity_id_field="live_access_binding_id",
            sort_key=lambda item: (item.live_access_binding_id, item.effective_revision),
        )

    def delete_knowledge_live_access_version_if_match(
        self,
        binding: WorkspaceLiveAccessBinding,
    ) -> bool:
        partition_key = _partition(
            binding.tenant_id,
            _ENTITY_KNOWLEDGE_CONFIGURATION_LIVE_ACCESS,
        )
        row_key = _revision_row_key(
            workspace_id=binding.workspace_id,
            entity_id=binding.live_access_binding_id,
            revision=binding.effective_revision,
        )
        return self._delete_if_match(binding, partition_key=partition_key, row_key=row_key)

    def put_knowledge_query_policy_version_if_absent(
        self,
        policy: WorkspaceCommittedQueryPolicy,
    ) -> bool:
        partition_key = _partition(
            policy.tenant_id,
            _ENTITY_KNOWLEDGE_CONFIGURATION_QUERY_POLICY,
        )
        row_key = _revision_row_key(
            workspace_id=policy.workspace_id,
            entity_id=_QUERY_POLICY_ENTITY_ID,
            revision=policy.effective_revision,
        )
        return self._put_if_absent(policy, partition_key=partition_key, row_key=row_key)

    def get_knowledge_query_policy_version(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        effective_revision: int,
    ) -> WorkspaceCommittedQueryPolicy | None:
        partition_key = _partition(
            tenant_id,
            _ENTITY_KNOWLEDGE_CONFIGURATION_QUERY_POLICY,
        )
        row_key = _revision_row_key(
            workspace_id=workspace_id,
            entity_id=_QUERY_POLICY_ENTITY_ID,
            revision=effective_revision,
        )
        record = self._store.get(partition_key, row_key)
        if record is None:
            return None
        try:
            policy = parse_workspace_query_policy(dict(record.data))
        except ValueError as exc:
            raise WorkspaceKnowledgeConfigurationRepositoryError(
                "query_policy_schema_version_unknown"
            ) from exc
        _assert_record_identity(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            model_tenant_id=policy.tenant_id,
            model_workspace_id=policy.workspace_id,
            revision=effective_revision,
            model_revision=policy.effective_revision,
        )
        return policy

    def list_knowledge_query_policy_versions(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
    ) -> list[WorkspaceCommittedQueryPolicy]:
        partition_key = _partition(
            tenant_id,
            _ENTITY_KNOWLEDGE_CONFIGURATION_QUERY_POLICY,
        )
        result = self._store.query(
            partition_key,
            limit=_KNOWLEDGE_CONFIGURATION_REVISION_SCAN_LIMIT + 1,
            row_key_prefix=f"{workspace_id}:",
        )
        if len(result.documents) > _KNOWLEDGE_CONFIGURATION_REVISION_SCAN_LIMIT:
            raise WorkspaceKnowledgeConfigurationRepositoryError(
                "knowledge_configuration_revision_scan_limit_exceeded"
            )
        items: list[WorkspaceCommittedQueryPolicy] = []
        for doc in result.documents:
            try:
                policy = parse_workspace_query_policy(dict(doc.data))
            except ValueError as exc:
                raise WorkspaceKnowledgeConfigurationRepositoryError(
                    "query_policy_schema_version_unknown"
                ) from exc
            row_workspace_id, row_entity_id, row_revision = _parse_revision_row_key(doc.row_key)
            if row_workspace_id != workspace_id:
                raise ValueError("knowledge_configuration_record_identity_mismatch")
            if row_entity_id != _QUERY_POLICY_ENTITY_ID:
                raise ValueError("knowledge_configuration_record_identity_mismatch")
            _assert_record_identity(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                model_tenant_id=policy.tenant_id,
                model_workspace_id=policy.workspace_id,
                entity_id=row_entity_id,
                model_entity_id=_QUERY_POLICY_ENTITY_ID,
                revision=row_revision,
                model_revision=policy.effective_revision,
            )
            items.append(policy)
        return sorted(items, key=lambda item: (item.effective_revision,))

    def delete_knowledge_query_policy_version_if_match(
        self,
        policy: WorkspaceCommittedQueryPolicy,
    ) -> bool:
        partition_key = _partition(
            policy.tenant_id,
            _ENTITY_KNOWLEDGE_CONFIGURATION_QUERY_POLICY,
        )
        row_key = _revision_row_key(
            workspace_id=policy.workspace_id,
            entity_id=_QUERY_POLICY_ENTITY_ID,
            revision=policy.effective_revision,
        )
        return self._delete_if_match(policy, partition_key=partition_key, row_key=row_key)
