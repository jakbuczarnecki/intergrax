# © Artur Czarnecki. All rights reserved.

"""Tests for Workspace Connection Attachment domain service and handler."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.runtime.vendor_knowledge.tenant_connections import (
    SafeTenantConnectionV1,
    TenantConnectionAdministrativeStatus,
)
from local_workspace_application.workspaces.knowledge_configuration_handlers import (
    AttachConnectionMutationHandler,
    AttachConnectionMutationIntent,
    connection_attachment_id,
    connection_attachment_request_hash,
    connection_attachment_semantic_identity_hash,
)
from local_workspace_application.workspaces.knowledge_configuration_models import (
    WorkspaceConnectionAttachment,
    WorkspaceConnectionAttachmentStatusV1,
    WorkspaceKnowledgeMutationOperationV1,
    WorkspaceKnowledgeMutationRecord,
    WorkspaceKnowledgeMutationStatusV1,
)
from local_workspace_application.workspaces.knowledge_configuration_mutation_engine import (
    WorkspaceKnowledgeConfigurationMutationEngine,
    WorkspaceKnowledgeConfigurationMutationError,
    WorkspaceKnowledgeMutationExecutionDispositionV1,
    WorkspaceKnowledgeStageInspection,
    WorkspaceKnowledgeStageStateV1,
)
from local_workspace_application.workspaces.knowledge_configuration_service import (
    WorkspaceKnowledgeConfigurationService,
)
from local_workspace_application.workspaces.knowledge_connection_attachment_service import (
    AttachWorkspaceConnectionCommand,
    WorkspaceConnectionAttachmentError,
    WorkspaceConnectionAttachmentService,
)
from local_workspace_application.workspaces.models import Workspace, WorkspaceStatus
from local_workspace_application.workspaces.repository import ManagedWorkspaceRepository

pytestmark = pytest.mark.unit

_NOW = datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC)
_TENANT, _TENANT_B, _WORKSPACE = "tenant-a", "tenant-b", "workspace-1"
_CONNECTION, _CONNECTION_OTHER = "conn.primary", "conn.other"
_SHA256, _SHA256_B = "a" * 64, "b" * 64
_LABEL = "Primary Connection"
_HANDLER = AttachConnectionMutationHandler()
_ATTACHMENT_ID = connection_attachment_id(
    tenant_id=_TENANT, workspace_id=_WORKSPACE, connection_ref=_CONNECTION
)


def _workspace(**overrides: object) -> Workspace:
    payload = {
        "workspace_id": _WORKSPACE,
        "tenant_id": _TENANT,
        "name": "Workspace",
        "status": WorkspaceStatus.ACTIVE,
        "created_at": _NOW,
        "updated_at": _NOW,
    }
    payload.update(overrides)
    return Workspace(**payload)


class _FakeWorkspaceLookup:
    def __init__(self, workspaces: dict[tuple[str, str], Workspace]) -> None:
        self._workspaces = workspaces

    def require_workspace(self, *, tenant_id: str, workspace_id: str) -> Workspace | None:
        workspace = self._workspaces.get((tenant_id, workspace_id))
        if workspace is None or workspace.tenant_id != tenant_id:
            return None
        return workspace


class _FakeConnectionPort:
    def __init__(self, connections: dict[tuple[str, str], SafeTenantConnectionV1]) -> None:
        self._connections = connections

    def get_connection(self, *, tenant_id: str, connection_ref: str) -> SafeTenantConnectionV1 | None:
        return self._connections.get((tenant_id.strip(), connection_ref.strip()))

    def list_connections(self, *, tenant_id: str, limit: int = 100, administrative_status=None):
        return tuple(v for (t, _), v in self._connections.items() if t == tenant_id.strip())


def _safe_connection(**overrides: object) -> SafeTenantConnectionV1:
    payload = {
        "connection_ref": _CONNECTION,
        "tenant_id": _TENANT,
        "provider_id": "provider.slack",
        "integration_kind": IntegrationCategory.CONVERSATION_CHANNEL,
        "safe_display_name": _LABEL,
        "administrative_status": TenantConnectionAdministrativeStatus.ACTIVE,
        "configuration_version": 1,
        "connected_principal_ref": None,
        "created_at": _NOW,
        "updated_at": _NOW,
    }
    payload.update(overrides)
    return SafeTenantConnectionV1(**payload)


def _build_stack(**kwargs: object):
    store = kwargs.get("store") or InMemoryDocumentStore()
    workspaces = kwargs["workspaces"] if "workspaces" in kwargs else {(_TENANT, _WORKSPACE): _workspace()}
    connections = kwargs["connections"] if "connections" in kwargs else {(_TENANT, _CONNECTION): _safe_connection()}
    repo = ManagedWorkspaceRepository(store)
    lookup = _FakeWorkspaceLookup(workspaces)
    config_service = WorkspaceKnowledgeConfigurationService(repo, lookup)
    engine = WorkspaceKnowledgeConfigurationMutationEngine(
        repo,
        lookup,
        config_service,
        {WorkspaceKnowledgeMutationOperationV1.ATTACH_CONNECTION: _HANDLER},
        clock=lambda: _NOW,
        mutation_id_factory=lambda: "mutation-1",
    )
    service = WorkspaceConnectionAttachmentService(
        connection_port=_FakeConnectionPort(connections),
        configuration_service=config_service,
        mutation_engine=engine,
    )
    return service, repo, engine


def _cmd(**overrides: object) -> AttachWorkspaceConnectionCommand:
    payload = {
        "tenant_id": _TENANT,
        "workspace_id": _WORKSPACE,
        "connection_ref": _CONNECTION,
        "expected_revision": 0,
        "idempotency_key_hash": _SHA256,
        "requested_safe_display_label": None,
    }
    payload.update(overrides)
    return AttachWorkspaceConnectionCommand(**payload)


def _mutation_hashes(connection_ref: str = _CONNECTION, label: str = _LABEL) -> dict[str, str]:
    return {
        "semantic_identity_hash": connection_attachment_semantic_identity_hash(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            connection_ref=connection_ref,
        ),
        "normalized_request_hash": connection_attachment_request_hash(
            connection_ref=connection_ref,
            safe_display_label=label,
        ),
    }


def _mutation(**overrides: object) -> WorkspaceKnowledgeMutationRecord:
    payload = {
        "mutation_id": "mutation-1",
        "tenant_id": _TENANT,
        "workspace_id": _WORKSPACE,
        "operation": WorkspaceKnowledgeMutationOperationV1.ATTACH_CONNECTION,
        "idempotency_key_hash": _SHA256,
        "status": WorkspaceKnowledgeMutationStatusV1.RESERVED,
        "created_at": _NOW,
        "updated_at": _NOW,
        "target_revision": 1,
        **_mutation_hashes(),
    }
    payload.update(overrides)
    return WorkspaceKnowledgeMutationRecord(**payload)


def _attachment_row(**overrides: object) -> WorkspaceConnectionAttachment:
    payload = {
        "attachment_id": _ATTACHMENT_ID,
        "tenant_id": _TENANT,
        "workspace_id": _WORKSPACE,
        "connection_ref": _CONNECTION,
        "safe_display_label": _LABEL,
        "status": WorkspaceConnectionAttachmentStatusV1.ATTACHED,
        "mutation_id": "mutation-1",
        "effective_revision": 1,
        "created_at": _NOW,
        "updated_at": _NOW,
    }
    payload.update(overrides)
    return WorkspaceConnectionAttachment(**payload)


def _stage_row(repo: ManagedWorkspaceRepository, **overrides: object) -> WorkspaceConnectionAttachment:
    row = _attachment_row(**overrides)
    repo.put_knowledge_connection_attachment_version_if_absent(row)
    return row


def _assert_no_attach_side_effects(repo: ManagedWorkspaceRepository) -> None:
    assert repo.get_knowledge_configuration_head(tenant_id=_TENANT, workspace_id=_WORKSPACE) is None
    assert not repo.list_knowledge_connection_attachment_versions(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
    )


def _assert_ownership_conflict(
    repo: ManagedWorkspaceRepository,
    mutation: WorkspaceKnowledgeMutationRecord,
    *,
    preserve_row: WorkspaceConnectionAttachment | None = None,
) -> None:
    inspection = _HANDLER.inspect_staged(repository=repo, mutation=mutation)
    assert inspection.state is WorkspaceKnowledgeStageStateV1.OWNERSHIP_CONFLICT
    assert not _HANDLER.cleanup_staged(repository=repo, mutation=mutation, inspection=inspection)
    if preserve_row is not None:
        assert repo.get_knowledge_connection_attachment_version(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            attachment_id=preserve_row.attachment_id,
            effective_revision=preserve_row.effective_revision,
        )


def test_successful_attach_applied() -> None:
    service, repo, _ = _build_stack()
    result = service.attach_connection(_cmd())
    assert result.disposition is WorkspaceKnowledgeMutationExecutionDispositionV1.APPLIED
    assert result.configuration_revision == 1
    assert result.attachment.status is WorkspaceConnectionAttachmentStatusV1.ATTACHED
    assert result.attachment.safe_display_label == _LABEL
    assert "credential_ref" not in result.attachment.model_dump()
    head = repo.get_knowledge_configuration_head(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert head is not None and head.committed_revision == 1


def test_semantic_no_op_and_committed_replay() -> None:
    service, repo, _ = _build_stack()
    applied = service.attach_connection(_cmd())
    replay = service.attach_connection(_cmd(expected_revision=0))
    noop = service.attach_connection(_cmd(idempotency_key_hash=_SHA256_B, expected_revision=1))
    assert applied.disposition is WorkspaceKnowledgeMutationExecutionDispositionV1.APPLIED
    assert replay.disposition is WorkspaceKnowledgeMutationExecutionDispositionV1.COMMITTED_REPLAY
    assert noop.disposition is WorkspaceKnowledgeMutationExecutionDispositionV1.EXISTING_RESULT
    assert len(repo.list_knowledge_connection_attachment_versions(
        tenant_id=_TENANT, workspace_id=_WORKSPACE,
    )) == 1


def test_idempotency_conflict_before_revision_conflict() -> None:
    service, _, _ = _build_stack()
    service.attach_connection(_cmd())
    with pytest.raises(WorkspaceKnowledgeConfigurationMutationError) as exc:
        service.attach_connection(_cmd(requested_safe_display_label="Alias", expected_revision=0))
    assert exc.value.error_code == "configuration_idempotency_conflict"


@pytest.mark.parametrize(
    ("connections", "error_code"),
    [
        ({}, "connection_not_found"),
        ({(_TENANT_B, _CONNECTION): _safe_connection(tenant_id=_TENANT_B)}, "connection_not_found"),
        ({(_TENANT, _CONNECTION): _safe_connection(connection_ref=_CONNECTION_OTHER)}, "connection_not_found"),
        ({(_TENANT, _CONNECTION): _safe_connection(
            administrative_status=TenantConnectionAdministrativeStatus.DISABLED
        )}, "connection_unavailable"),
        ({(_TENANT, _CONNECTION): _safe_connection(
            administrative_status=TenantConnectionAdministrativeStatus.REVOKED
        )}, "connection_unavailable"),
    ],
)
def test_connection_boundary(connections, error_code) -> None:
    service, repo, _ = _build_stack(connections=connections)
    with pytest.raises(WorkspaceConnectionAttachmentError) as exc:
        service.attach_connection(_cmd())
    assert exc.value.error_code == error_code
    if error_code == "connection_not_found":
        _assert_no_attach_side_effects(repo)


@pytest.mark.parametrize(
    ("requested", "error"),
    [
        (None, None),
        ("  Alias  ", None),
        ("   ", "safe_display_label_invalid"),
        ("x" * 257, "safe_display_label_invalid"),
        ("Authorization: secret", "safe_display_label_invalid"),
        ("Bearer abc123", "safe_display_label_invalid"),
        ("api-key: value", "safe_display_label_invalid"),
        ("https://user:pass@example.com/path", "safe_display_label_invalid"),
        ("Support: https://user:pass@example.com", "safe_display_label_invalid"),
        ("Docs at https://example.com/path?access_token=secret", "safe_display_label_invalid"),
        ("Open https://example.com?API_KEY=value for details", "safe_display_label_invalid"),
        ("Documentation: https://example.com/help", None),
    ],
)
def test_safe_display_label(requested: str | None, error: str | None) -> None:
    service, repo, _ = _build_stack()
    if error is None:
        result = service.attach_connection(_cmd(requested_safe_display_label=requested))
        expected = requested.strip() if requested is not None else _LABEL
        assert result.attachment.safe_display_label == expected
        return
    with pytest.raises(WorkspaceConnectionAttachmentError) as exc:
        service.attach_connection(_cmd(requested_safe_display_label=requested))
    assert exc.value.error_code == error
    _assert_no_attach_side_effects(repo)


def test_unsafe_server_derived_default_label() -> None:
    service, repo, _ = _build_stack(
        connections={(_TENANT, _CONNECTION): _safe_connection(
            safe_display_name="Support: https://user:pass@example.com"
        )}
    )
    with pytest.raises(WorkspaceConnectionAttachmentError) as exc:
        service.attach_connection(_cmd())
    assert exc.value.error_code == "safe_display_label_invalid"
    _assert_no_attach_side_effects(repo)


def test_handler_stage_inspect_cleanup_and_foreign_ownership() -> None:
    repo = ManagedWorkspaceRepository(InMemoryDocumentStore())
    mutation = _mutation()
    intent = AttachConnectionMutationIntent(
        attachment_id=_ATTACHMENT_ID,
        connection_ref=_CONNECTION,
        safe_display_label=_LABEL,
    )
    _HANDLER.stage(repository=repo, mutation=mutation, target_revision=1, intent=intent, now=_NOW)
    inspection = _HANDLER.inspect_staged(repository=repo, mutation=mutation)
    assert inspection.state is WorkspaceKnowledgeStageStateV1.COMPLETE_VALID
    assert _HANDLER.cleanup_staged(repository=repo, mutation=mutation, inspection=inspection)
    repo.put_knowledge_connection_attachment_version_if_absent(
        _attachment_row(mutation_id="foreign-mutation")
    )
    assert _HANDLER.inspect_staged(repository=repo, mutation=mutation).state is (
        WorkspaceKnowledgeStageStateV1.ABSENT
    )


def test_handler_cleanup_failure_and_preserves_committed() -> None:
    repo = ManagedWorkspaceRepository(InMemoryDocumentStore())
    committed = _attachment_row(mutation_id="committed-mutation", effective_revision=1)
    staged = committed.model_copy(update={"mutation_id": "mutation-1", "effective_revision": 2})
    repo.put_knowledge_connection_attachment_version_if_absent(committed)
    repo.put_knowledge_connection_attachment_version_if_absent(staged)
    mutation = _mutation(target_revision=2)
    inspection = _HANDLER.inspect_staged(repository=repo, mutation=mutation)
    assert _HANDLER.cleanup_staged(repository=repo, mutation=mutation, inspection=inspection)
    assert repo.get_knowledge_connection_attachment_version(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        attachment_id=committed.attachment_id,
        effective_revision=1,
    )
    repo2 = ManagedWorkspaceRepository(InMemoryDocumentStore())
    repo2.put_knowledge_connection_attachment_version_if_absent(_attachment_row())
    repo2.delete_knowledge_connection_attachment_version_if_match = lambda *_: False  # type: ignore[method-assign]
    m = _mutation()
    assert not _HANDLER.cleanup_staged(
        repository=repo2,
        mutation=m,
        inspection=_HANDLER.inspect_staged(repository=repo2, mutation=m),
    )


def test_handler_cleanup_ownership_conflict_is_false() -> None:
    repo = ManagedWorkspaceRepository(InMemoryDocumentStore())
    row = _stage_row(repo)
    mutation = _mutation()
    conflict = WorkspaceKnowledgeStageInspection(state=WorkspaceKnowledgeStageStateV1.OWNERSHIP_CONFLICT)
    assert not _HANDLER.cleanup_staged(repository=repo, mutation=mutation, inspection=conflict)
    assert repo.get_knowledge_connection_attachment_version(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        attachment_id=row.attachment_id,
        effective_revision=row.effective_revision,
    )


@pytest.mark.parametrize(
    ("row_overrides",),
    [
        ({"connection_ref": _CONNECTION_OTHER, "attachment_id": connection_attachment_id(
            tenant_id=_TENANT, workspace_id=_WORKSPACE, connection_ref=_CONNECTION_OTHER
        )},),
        ({"attachment_id": "wca:wrongattachmentidentity00000000"},),
        ({"safe_display_label": "Different Label"},),
    ],
)
def test_handler_staged_identity_conflicts(row_overrides: dict[str, object]) -> None:
    repo = ManagedWorkspaceRepository(InMemoryDocumentStore())
    row = _stage_row(repo, **row_overrides)
    _assert_ownership_conflict(repo, _mutation(), preserve_row=row)


@pytest.mark.parametrize(
    ("result_entity_type", "result_entity_id"),
    [
        ("indexed_source_binding", _ATTACHMENT_ID),
        ("connection_attachment", "wca:wrongattachmentidentity00000000"),
    ],
)
def test_handler_prepared_result_reference_mismatch(
    result_entity_type: str,
    result_entity_id: str,
) -> None:
    repo = ManagedWorkspaceRepository(InMemoryDocumentStore())
    _stage_row(repo)
    mutation = _mutation(
        status=WorkspaceKnowledgeMutationStatusV1.PREPARED,
        result_entity_type=result_entity_type,
        result_entity_id=result_entity_id,
    )
    assert _HANDLER.inspect_staged(repository=repo, mutation=mutation).state is (
        WorkspaceKnowledgeStageStateV1.OWNERSHIP_CONFLICT
    )
