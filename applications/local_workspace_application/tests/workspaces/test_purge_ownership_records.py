from datetime import UTC, datetime

import pytest

from local_workspace_application.workspaces.connected_source_models import (
    ConnectedSourceOperationDeliveryAccounting,
    ConnectedSourceSyncEnqueueIntent,
)
from local_workspace_application.workspaces.document_indexing import (
    _WorkspaceDocumentIndexReceipt,
)
from local_workspace_application.workspaces.materialization_visibility import (
    KnowledgeMaterializationOwnershipV1,
)

pytestmark = pytest.mark.unit


def _ownership() -> KnowledgeMaterializationOwnershipV1:
    return KnowledgeMaterializationOwnershipV1.connected(
        tenant_id="tenant-a",
        workspace_id="workspace-a",
        source_id="source-a",
        indexed_source_binding_id="binding-a",
        knowledge_source_binding_ref="knowledge-binding-a",
        delivery_id="d" * 64,
        remote_id="remote-a",
    )


def test_connected_index_receipt_persists_complete_ownership() -> None:
    ownership = _ownership()
    receipt = _WorkspaceDocumentIndexReceipt(
        tenant_id=ownership.tenant_id,
        workspace_id=ownership.workspace_id,
        source_id=ownership.source_id,
        operation_id="operation-a",
        logical_source_path="/remote-a.md",
        safe_file_name="remote-a.md",
        content_hash="sha256:abc",
        document_id="document-a",
        status="in_progress",
        created_at=datetime.now(UTC),
        materialization_scope=ownership.identity_scope,
        materialization_ownership=ownership,
    )
    assert receipt.materialization_ownership == ownership


def test_recovery_records_require_complete_or_explicit_legacy_classification() -> None:
    intent = ConnectedSourceSyncEnqueueIntent(
        tenant_id="tenant-a",
        workspace_id="workspace-a",
        source_id="source-a",
        indexed_source_binding_id="binding-a",
        knowledge_source_binding_ref="knowledge-binding-a",
        operation_id="operation-a",
        enqueue_generation=1,
        updated_at=datetime.now(UTC),
        ownership_classification="COMPLETE_OWNERSHIP",
    )
    assert intent.ownership_classification == "COMPLETE_OWNERSHIP"
    with pytest.raises(ValueError, match="ownership_incomplete"):
        ConnectedSourceSyncEnqueueIntent(
            tenant_id="tenant-a",
            workspace_id="workspace-a",
            source_id="source-a",
            indexed_source_binding_id="binding-a",
            operation_id="operation-b",
            enqueue_generation=1,
            updated_at=datetime.now(UTC),
            ownership_classification="COMPLETE_OWNERSHIP",
        )


def test_delivery_accounting_carries_binding_identity() -> None:
    ownership = _ownership()
    accounting = ConnectedSourceOperationDeliveryAccounting(
        tenant_id=ownership.tenant_id,
        workspace_id=ownership.workspace_id,
        source_id=ownership.source_id,
        indexed_source_binding_id=ownership.indexed_source_binding_id,
        knowledge_source_binding_ref=ownership.knowledge_source_binding_ref,
        operation_id="operation-a",
        delivery_id=ownership.delivery_id,
        documents_indexed=1,
        documents_unchanged=0,
        items_failed=0,
        accounted_at=datetime.now(UTC),
        ownership_classification="COMPLETE_OWNERSHIP",
    )
    assert accounting.workspace_id == ownership.workspace_id
    assert accounting.indexed_source_binding_id == ownership.indexed_source_binding_id
