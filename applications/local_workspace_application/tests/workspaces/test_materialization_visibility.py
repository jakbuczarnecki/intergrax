from datetime import UTC, datetime

import pytest
from local_workspace_application.workspaces.connected_source_models import (
    ConnectedSourceDeliveryReceipt,
    ConnectedSourceDeliverySequenceAssignment,
    ConnectedSourceDeliverySequenceHead,
    ConnectedSourceDeliveryStatus,
    ConnectedSourceSyncSinkError,
)
from local_workspace_application.workspaces.connected_source_delivery import (
    begin_delivery_receipt,
    complete_delivery_receipt,
    delivery_receipt_completed,
)
from local_workspace_application.workspaces.materialization_visibility import (
    KnowledgeMaterializationActivePointerV1,
    KnowledgeMaterializationOwnershipModeV1,
    KnowledgeMaterializationOwnershipV1,
    RepositoryKnowledgeMaterializationVisibility,
)
from local_workspace_application.workspaces.models import (
    WorkspaceDocumentReference,
    WorkspaceSource,
    WorkspaceSourceType,
)
from local_workspace_application.workspaces.repository import ManagedWorkspaceRepository
from local_workspace_application.workspaces.search_evidence import map_search_hits
from pydantic import ValidationError

from intergrax.integrations._shared.in_memory_document_store import (
    InMemoryDocumentStore,
)
from intergrax.integrations.contracts.document_store import DocumentRecord

_NOW = datetime(2026, 8, 5, tzinfo=UTC)
_TENANT = "tenant-a"
_WORKSPACE = "workspace-a"
_SOURCE = "source-a"
_BINDING = "binding-a"
_BINDING_REF = "knowledge-binding-a"
_DELIVERY_1 = "a" * 64
_DELIVERY_2 = "b" * 64


def _repository() -> ManagedWorkspaceRepository:
    repository = ManagedWorkspaceRepository(InMemoryDocumentStore())
    repository.put_source(
        WorkspaceSource(
            source_id=_SOURCE,
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            source_type=WorkspaceSourceType.CONNECTED_SOURCE,
            knowledge_configuration_creation_mutation_id="mutation-a",
            knowledge_configuration_visibility_revision=1,
            created_at=_NOW,
        )
    )
    return repository


def _ownership(delivery_id: str, *, binding_id: str = _BINDING) -> KnowledgeMaterializationOwnershipV1:
    return KnowledgeMaterializationOwnershipV1.connected(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        source_id=_SOURCE,
        indexed_source_binding_id=binding_id,
        knowledge_source_binding_ref=_BINDING_REF,
        delivery_id=delivery_id,
        remote_id="remote-message-1",
    )


def _completed_receipt(
    ownership: KnowledgeMaterializationOwnershipV1,
    *,
    binding_configuration_version: int = 1,
    materialization_sequence: int = 1,
) -> ConnectedSourceDeliveryReceipt:
    assert ownership.indexed_source_binding_id is not None
    assert ownership.knowledge_source_binding_ref is not None
    assert ownership.delivery_id is not None
    return ConnectedSourceDeliveryReceipt(
        tenant_id=ownership.tenant_id,
        workspace_id=ownership.workspace_id,
        source_id=ownership.source_id,
        indexed_source_binding_id=ownership.indexed_source_binding_id,
        knowledge_source_binding_ref=ownership.knowledge_source_binding_ref,
        delivery_id=ownership.delivery_id,
        binding_configuration_version=binding_configuration_version,
        materialization_sequence=materialization_sequence,
        operation_id="operation-a",
        status=ConnectedSourceDeliveryStatus.COMPLETED,
        documents_indexed=1,
        items_failed=0,
        created_at=_NOW,
        completed_at=_NOW,
    )


def _commit(
    repository: ManagedWorkspaceRepository,
    ownership: KnowledgeMaterializationOwnershipV1,
    *,
    document_id: str,
    materialization_revision: int = 1,
) -> None:
    repository.put_connected_source_delivery_receipt(
        _completed_receipt(
            ownership,
            binding_configuration_version=materialization_revision,
            materialization_sequence=materialization_revision,
        )
    )
    repository.put_active_materialization_pointer_if_absent(
        KnowledgeMaterializationActivePointerV1.for_ownership(
            ownership=ownership,
            document_id=document_id,
            materialization_revision=materialization_revision,
            committed_at=_NOW,
        )
    )


def test_ownership_is_strict_and_credential_free() -> None:
    ownership = _ownership(_DELIVERY_1)
    assert ownership.model_dump(mode="json")["delivery_id"] == _DELIVERY_1
    assert "secret" not in repr(ownership.model_dump(mode="json")).lower()
    with pytest.raises(ValidationError, match="connected_materialization_ownership_incomplete"):
        KnowledgeMaterializationOwnershipV1(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            source_id=_SOURCE,
            indexed_source_binding_id=_BINDING,
        )


def test_legacy_is_explicit_and_connected_legacy_visibility_fails_closed() -> None:
    repository = _repository()
    resolver = RepositoryKnowledgeMaterializationVisibility(repository)
    legacy = KnowledgeMaterializationOwnershipV1.legacy(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        source_id="local-source",
    )
    repository.put_source(
        WorkspaceSource(
            source_id="local-source",
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            source_type=WorkspaceSourceType.LOCAL_FOLDER,
            path="docs",
            created_at=_NOW,
        )
    )
    assert resolver.is_visible(ownership=legacy)
    assert not resolver.is_visible(
        ownership=KnowledgeMaterializationOwnershipV1(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            source_id=_SOURCE,
            ownership_mode=KnowledgeMaterializationOwnershipModeV1.LEGACY,
        )
    )


def test_completed_delivery_and_active_pointer_control_supersession() -> None:
    repository = _repository()
    resolver = RepositoryKnowledgeMaterializationVisibility(repository)
    first = _ownership(_DELIVERY_1)
    second = _ownership(_DELIVERY_2)
    _commit(repository, first, document_id="document-1")
    assert resolver.is_visible(ownership=first)
    assert not resolver.is_visible(ownership=second)

    repository.put_connected_source_delivery_receipt(
        _completed_receipt(
            second,
            binding_configuration_version=2,
            materialization_sequence=2,
        )
    )
    assert resolver.is_visible(ownership=first)
    assert not resolver.is_visible(ownership=second)

    wrong_binding = _ownership(_DELIVERY_2, binding_id="binding-other")
    assert not resolver.is_visible(ownership=wrong_binding)


def test_historical_receipt_without_sequence_is_hidden() -> None:
    repository = _repository()
    resolver = RepositoryKnowledgeMaterializationVisibility(repository)
    ownership = _ownership(_DELIVERY_1)
    _commit(repository, ownership, document_id="document-1")
    historical = _completed_receipt(ownership).model_copy(
        update={"materialization_sequence": None}
    )
    repository.put_connected_source_delivery_receipt(historical)
    assert not resolver.is_visible(ownership=ownership)


def test_active_pointer_rejects_naive_committed_at() -> None:
    with pytest.raises(ValidationError, match="timezone_aware"):
        KnowledgeMaterializationActivePointerV1.for_ownership(
            ownership=_ownership(_DELIVERY_1),
            document_id="document-1",
            materialization_revision=1,
            committed_at=_NOW.replace(tzinfo=None),
        )


def test_sequence_models_are_strict_and_utc() -> None:
    head = ConnectedSourceDeliverySequenceHead(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        source_id=_SOURCE,
        indexed_source_binding_id=_BINDING,
        next_sequence=1,
    )
    assert head.model_dump() == {
        "tenant_id": _TENANT,
        "workspace_id": _WORKSPACE,
        "source_id": _SOURCE,
        "indexed_source_binding_id": _BINDING,
        "next_sequence": 1,
    }
    with pytest.raises(ValidationError, match="extra"):
        ConnectedSourceDeliverySequenceHead(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            source_id=_SOURCE,
            indexed_source_binding_id=_BINDING,
            next_sequence=1,
            delivery_sequences={_DELIVERY_1: 1},
        )
    with pytest.raises(ValidationError, match="connected_source_delivery_id_invalid"):
        ConnectedSourceDeliverySequenceAssignment(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            source_id=_SOURCE,
            indexed_source_binding_id=_BINDING,
            delivery_id="not-sha256",
            materialization_sequence=1,
            assigned_at=_NOW,
        )
    with pytest.raises(ValidationError, match="timezone_aware"):
        ConnectedSourceDeliverySequenceAssignment(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            source_id=_SOURCE,
            indexed_source_binding_id=_BINDING,
            delivery_id=_DELIVERY_1,
            materialization_sequence=1,
            assigned_at=_NOW.replace(tzinfo=None),
        )


def test_receipt_sequence_matches_immutable_assignment() -> None:
    repository = _repository()
    ownership = _ownership(_DELIVERY_1)
    assert ownership.indexed_source_binding_id is not None
    assignment_sequence = repository.allocate_connected_source_delivery_sequence(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        source_id=_SOURCE,
        indexed_source_binding_id=ownership.indexed_source_binding_id,
        delivery_id=_DELIVERY_1,
    )
    assignment = repository.get_connected_source_delivery_sequence_assignment(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        source_id=_SOURCE,
        indexed_source_binding_id=ownership.indexed_source_binding_id,
        delivery_id=_DELIVERY_1,
    )
    assert assignment is not None
    assert assignment.materialization_sequence == assignment_sequence

    repository.put_connected_source_delivery_receipt(
        _completed_receipt(
            ownership,
            materialization_sequence=assignment_sequence + 1,
        )
    )
    with pytest.raises(
        ConnectedSourceSyncSinkError,
        match="connected_source_delivery_sequence_conflict",
    ):
        delivery_receipt_completed(
            repository=repository,
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            source_id=_SOURCE,
            delivery_id=_DELIVERY_1,
            indexed_source_binding_id=ownership.indexed_source_binding_id,
            knowledge_source_binding_ref=_BINDING_REF,
            binding_configuration_version=1,
            operation_id="operation-a",
        )

    replay = begin_delivery_receipt(
        repository=repository,
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        source_id=_SOURCE,
        indexed_source_binding_id=ownership.indexed_source_binding_id,
        knowledge_source_binding_ref=_BINDING_REF,
        delivery_id=_DELIVERY_2,
        binding_configuration_version=1,
        operation_id="operation-b",
    )
    replay_assignment = repository.get_connected_source_delivery_sequence_assignment(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        source_id=_SOURCE,
        indexed_source_binding_id=ownership.indexed_source_binding_id,
        delivery_id=_DELIVERY_2,
    )
    assert replay_assignment is not None
    assert replay.materialization_sequence == replay_assignment.materialization_sequence
    completed = complete_delivery_receipt(
        repository=repository,
        receipt=replay,
        documents_indexed=0,
        documents_unchanged=0,
        items_processed=0,
        items_failed=0,
    )
    assert completed.materialization_sequence == replay_assignment.materialization_sequence


class _SearchExecution:
    def __init__(self, evidence: list[dict[str, object]]) -> None:
        self.structured_data = {"search_summary": {"evidence": evidence}}


class _SearchResult:
    def __init__(self, evidence: list[dict[str, object]]) -> None:
        self.execution_result = _SearchExecution(evidence)


def test_actual_search_boundary_filters_prepared_and_malformed_documents() -> None:
    repository = _repository()
    committed = _ownership(_DELIVERY_1)
    _commit(repository, committed, document_id="document-1")
    repository.put_document_ref(
        WorkspaceDocumentReference(
            document_id="document-1",
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            source_id=_SOURCE,
            source_path="connected/source-a/remote-message-1.md",
            file_name="message.md",
            content_hash="sha256:" + "a" * 64,
            indexed_at=_NOW,
            materialization_ownership=committed,
            visibility_authority_ref=_DELIVERY_1,
            visibility_authority_type="delivery_receipt",
        )
    )
    repository.document_store.put(
        DocumentRecord(
            partition_key=f"lkw.managed_workspace:{_TENANT}:document",
            row_key=f"{_WORKSPACE}:malformed-document",
            data={
                "document_id": "malformed-document",
                "tenant_id": _TENANT,
                "workspace_id": _WORKSPACE,
                "source_id": _SOURCE,
                "source_path": "connected/source-a/malformed.md",
                "file_name": "malformed.md",
                "content_hash": "sha256:" + "c" * 64,
                "indexed_at": _NOW.isoformat(),
                "materialization_ownership": {
                    "tenant_id": _TENANT,
                    "workspace_id": _WORKSPACE,
                    "source_id": _SOURCE,
                    "indexed_source_binding_id": _BINDING,
                },
                "visibility_authority_type": "delivery_receipt",
                "visibility_authority_ref": _DELIVERY_1,
            },
        )
    )
    prepared = _ownership(_DELIVERY_2)
    repository.put_connected_source_delivery_receipt(
        ConnectedSourceDeliveryReceipt(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            source_id=_SOURCE,
            indexed_source_binding_id=_BINDING,
            knowledge_source_binding_ref=_BINDING_REF,
            delivery_id=_DELIVERY_2,
            binding_configuration_version=1,
            materialization_sequence=2,
            operation_id="operation-b",
            status=ConnectedSourceDeliveryStatus.IN_PROGRESS,
            created_at=_NOW,
        )
    )
    repository.put_document_ref(
        WorkspaceDocumentReference(
            document_id="document-2",
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            source_id=_SOURCE,
            source_path="connected/source-a/remote-message-1-v2.md",
            file_name="message-v2.md",
            content_hash="sha256:" + "b" * 64,
            indexed_at=_NOW,
            materialization_ownership=prepared,
            visibility_authority_ref=_DELIVERY_2,
            visibility_authority_type="delivery_receipt",
        )
    )
    evidence = [
        {
            "document_id": "document-1",
            "source_id": _SOURCE,
            "workspace_id": _WORKSPACE,
            "source_path": "connected/source-a/remote-message-1.md",
            "file_name": "message.md",
            "score": 0.9,
            "snippet": "committed",
        },
        {
            "document_id": "document-2",
            "source_id": _SOURCE,
            "workspace_id": _WORKSPACE,
            "source_path": "connected/source-a/remote-message-1-v2.md",
            "file_name": "message-v2.md",
            "score": 0.85,
            "snippet": "prepared",
        },
        {
            "document_id": "missing-document",
            "source_id": _SOURCE,
            "workspace_id": _WORKSPACE,
            "source_path": "connected/source-a/missing.md",
            "file_name": "missing.md",
            "score": 0.8,
            "snippet": "prepared or malformed",
        },
        {
            "document_id": "malformed-document",
            "source_id": _SOURCE,
            "workspace_id": _WORKSPACE,
            "source_path": "connected/source-a/malformed.md",
            "file_name": "malformed.md",
            "score": 0.7,
            "snippet": "malformed ownership",
        },
    ]
    hits = map_search_hits(
        repository=repository,
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        task_result=_SearchResult(evidence),
        limit=10,
    )
    assert [hit.document_id for hit in hits] == ["document-1"]
