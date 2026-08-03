# © Artur Czarnecki. All rights reserved.

"""Tests for Workspace Connection Detachment domain service and handler."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.conversation_channel.slack.integration import (
    SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
)
from intergrax.integrations.providers.conversation_channel.slack.knowledge_read import (
    SLACK_CONVERSATION_SOURCE_KIND,
)
from intergrax.runtime.vendor_knowledge.bindings import (
    KnowledgeSourceBinding,
    KnowledgeSourceBindingStatus,
    KnowledgeSourceScope,
)
from intergrax.runtime.vendor_knowledge.tenant_connections import (
    SafeTenantConnectionV1,
    TenantConnectionAdministrativeStatus,
)
from local_workspace_application.workspaces.knowledge_configuration_handlers import (
    AttachConnectionMutationHandler,
    connection_attachment_id,
    connection_attachment_semantic_identity_hash,
)
from local_workspace_application.workspaces.knowledge_configuration_models import (
    LiveAccessBindingStatusV1,
    WorkspaceConnectionAttachmentStatusV1,
    WorkspaceIndexedSourceBindingStatusV1,
    WorkspaceKnowledgeMutationOperationV1,
    WorkspaceKnowledgeMutationRecord,
    WorkspaceKnowledgeMutationStatusV1,
)
from local_workspace_application.workspaces.knowledge_configuration_mutation_engine import (
    WorkspaceKnowledgeConfigurationMutationEngine,
    WorkspaceKnowledgeMutationExecutionDispositionV1,
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
from local_workspace_application.workspaces.knowledge_connection_detachment_handler import (
    DetachConnectionMutationHandler,
    DetachConnectionMutationIntent,
    detach_connection_request_hash,
    detach_connection_stage_manifest_hash,
)
from local_workspace_application.workspaces.knowledge_connection_detachment_service import (
    DetachWorkspaceConnectionCommand,
    WorkspaceConnectionDetachmentService,
)
from local_workspace_application.workspaces.models import (
    Workspace,
    WorkspaceSource,
    WorkspaceSourceStatus,
    WorkspaceSourceType,
    WorkspaceStatus,
)
from local_workspace_application.workspaces.repository import ManagedWorkspaceRepository

pytestmark = pytest.mark.unit

_NOW = datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC)
_TENANT, _WORKSPACE = "tenant-a", "workspace-1"
_CONNECTION, _CONNECTION_OTHER = "conn.primary", "conn.other"
_SHA256, _SHA256_B = "a" * 64, "b" * 64
_LABEL = "Primary Connection"
_ATTACH_HANDLER = AttachConnectionMutationHandler()
_DETACH_HANDLER = DetachConnectionMutationHandler()
_ATTACHMENT_ID = connection_attachment_id(
    tenant_id=_TENANT, workspace_id=_WORKSPACE, connection_ref=_CONNECTION
)


class _FakeWorkspaceLookup:
    def __init__(self, workspaces: dict[tuple[str, str], Workspace]) -> None:
        self._workspaces = workspaces

    def require_workspace(self, *, tenant_id: str, workspace_id: str) -> Workspace | None:
        workspace = self._workspaces.get((tenant_id, workspace_id))
        if workspace is None or workspace.tenant_id != tenant_id:
            return None
        return workspace


class _FakeConnectionPort:
    def get_connection(self, *, tenant_id: str, connection_ref: str) -> SafeTenantConnectionV1 | None:
        if tenant_id == _TENANT and connection_ref == _CONNECTION:
            return SafeTenantConnectionV1(
                connection_ref=_CONNECTION,
                tenant_id=_TENANT,
                provider_id="provider.slack",
                integration_kind=IntegrationCategory.CONVERSATION_CHANNEL,
                safe_display_name=_LABEL,
                administrative_status=TenantConnectionAdministrativeStatus.ACTIVE,
                configuration_version=1,
                connected_principal_ref=None,
                created_at=_NOW,
                updated_at=_NOW,
            )
        return None

    def list_connections(self, *, tenant_id: str, limit: int = 100, administrative_status=None):
        conn = self.get_connection(tenant_id=tenant_id, connection_ref=_CONNECTION)
        return (conn,) if conn is not None else ()


class _TenantBindingPort:
    def __init__(self, bindings: dict[str, KnowledgeSourceBinding] | None = None, *, fail: bool = False) -> None:
        self._bindings = bindings or {}
        self._fail = fail

    def get_binding(self, *, tenant_id: str, binding_id: str) -> KnowledgeSourceBinding | None:
        if self._fail:
            raise RuntimeError("lookup_failed")
        return self._bindings.get(binding_id)


def _tenant_binding(
    *,
    binding_id: str = "ksb-1",
    connection_ref: str = _CONNECTION,
    tenant_id: str = _TENANT,
) -> KnowledgeSourceBinding:
    return KnowledgeSourceBinding(
        binding_id=binding_id,
        tenant_id=tenant_id,
        provider_id=SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
        integration_kind=IntegrationCategory.CONVERSATION_CHANNEL,
        source_kind=SLACK_CONVERSATION_SOURCE_KIND,
        connection_ref=connection_ref,
        safe_display_name="Binding",
        scope=KnowledgeSourceScope(
            remote_scope_id="scope",
            remote_scope_type="slack_conversation",
            safe_display_name="Binding",
            parameters={},
        ),
        status=KnowledgeSourceBindingStatus.ACTIVE,
        configuration_version=1,
    )


def _workspace() -> Workspace:
    return Workspace(
        workspace_id=_WORKSPACE,
        tenant_id=_TENANT,
        name="Workspace",
        status=WorkspaceStatus.ACTIVE,
        created_at=_NOW,
        updated_at=_NOW,
    )


def _build_stack(
  bindings: _TenantBindingPort | None = None,
  mutation_ids: list[str] | None = None,
):
    store = InMemoryDocumentStore()
    repo = ManagedWorkspaceRepository(store)
    repo.put_workspace(_workspace())
    lookup = _FakeWorkspaceLookup({(_TENANT, _WORKSPACE): _workspace()})
    config_service = WorkspaceKnowledgeConfigurationService(repo, lookup)
    ids = mutation_ids or ["mutation-1", "mutation-2", "mutation-3"]
    idx = {"i": 0}

    def _next_id() -> str:
        value = ids[idx["i"]]
        idx["i"] = min(idx["i"] + 1, len(ids) - 1)
        return value

    engine = WorkspaceKnowledgeConfigurationMutationEngine(
        repo,
        lookup,
        config_service,
        {
            WorkspaceKnowledgeMutationOperationV1.ATTACH_CONNECTION: _ATTACH_HANDLER,
            WorkspaceKnowledgeMutationOperationV1.DETACH_CONNECTION: _DETACH_HANDLER,
        },
        clock=lambda: _NOW,
        mutation_id_factory=_next_id,
    )
    attach_service = WorkspaceConnectionAttachmentService(
        connection_port=_FakeConnectionPort(),
        configuration_service=config_service,
        mutation_engine=engine,
    )
    detach_service = WorkspaceConnectionDetachmentService(
        configuration_service=config_service,
        mutation_engine=engine,
        tenant_binding_port=bindings or _TenantBindingPort(),
    )
    return attach_service, detach_service, repo, engine, config_service


def _attach(attach_service: WorkspaceConnectionAttachmentService, *, revision: int = 0) -> int:
    result = attach_service.attach_connection(
        AttachWorkspaceConnectionCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            connection_ref=_CONNECTION,
            expected_revision=revision,
            idempotency_key_hash=_SHA256,
        )
    )
    return result.configuration_revision


def _detach_cmd(**overrides: object) -> DetachWorkspaceConnectionCommand:
    payload = {
        "tenant_id": _TENANT,
        "workspace_id": _WORKSPACE,
        "connection_ref": _CONNECTION,
        "expected_revision": 1,
        "idempotency_key_hash": _SHA256_B,
    }
    payload.update(overrides)
    return DetachWorkspaceConnectionCommand(**payload)


def _indexed_row(
    binding_id: str,
    *,
    status: WorkspaceIndexedSourceBindingStatusV1,
    ksb_ref: str = "ksb-1",
    revision: int = 1,
) -> dict:
    from local_workspace_application.workspaces.knowledge_configuration_models import (
        WorkspaceIndexedSourceBinding,
    )

    return WorkspaceIndexedSourceBinding(
        indexed_source_binding_id=binding_id,
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        knowledge_source_binding_ref=ksb_ref,
        source_id=f"source-{binding_id}",
        status=status,
        mutation_id="seed-mutation",
        effective_revision=revision,
        semantic_identity_hash=_SHA256,
        created_at=_NOW,
        updated_at=_NOW,
    )


def _live_row(
    binding_id: str,
    *,
    status: LiveAccessBindingStatusV1,
    connection_ref: str = _CONNECTION,
    revision: int = 1,
):
    from local_workspace_application.workspaces.knowledge_configuration_models import (
        WorkspaceLiveAccessBinding,
    )

    return WorkspaceLiveAccessBinding(
        live_access_binding_id=binding_id,
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        connection_ref=connection_ref,
        allowed_capability_ids=("cap.read",),
        derived_provider_id="provider-1",
        derived_integration_kind=IntegrationCategory.WIKI_KNOWLEDGE,
        derived_safe_display_label="Wiki",
        status=status,
        mutation_id="seed-mutation",
        effective_revision=revision,
        semantic_identity_hash=_SHA256,
        created_at=_NOW,
        updated_at=_NOW,
    )


def test_simple_detach() -> None:
    attach_service, detach_service, repo, _, _ = _build_stack()
    revision = _attach(attach_service)
    config_before = attach_service._configuration_service.get_configuration(
        tenant_id=_TENANT, workspace_id=_WORKSPACE
    )
    assert config_before is not None
    created_at = config_before.connection_attachments[0].created_at
    result = detach_service.detach_connection(_detach_cmd(expected_revision=revision))
    assert result.disposition is WorkspaceKnowledgeMutationExecutionDispositionV1.APPLIED
    assert result.attachment.status is WorkspaceConnectionAttachmentStatusV1.DETACHED
    assert result.attachment.attachment_id == _ATTACHMENT_ID
    assert result.attachment.safe_display_label == _LABEL
    assert result.attachment.created_at == created_at
    assert result.configuration_revision == revision + 1


def test_detach_unavailable_attachment() -> None:
    attach_service, detach_service, repo, _, _ = _build_stack()
    revision = _attach(attach_service)
    config = attach_service._configuration_service.get_configuration(
        tenant_id=_TENANT, workspace_id=_WORKSPACE
    )
    prev = config.connection_attachments[0]
    from local_workspace_application.workspaces.knowledge_configuration_models import (
        WorkspaceConnectionAttachment,
    )

    repo.put_knowledge_connection_attachment_version_if_absent(
        WorkspaceConnectionAttachment(
            attachment_id=prev.attachment_id,
            tenant_id=prev.tenant_id,
            workspace_id=prev.workspace_id,
            connection_ref=prev.connection_ref,
            safe_display_label=prev.safe_display_label,
            status=WorkspaceConnectionAttachmentStatusV1.UNAVAILABLE,
            mutation_id="mutation-unavailable",
            effective_revision=revision + 1,
            created_at=prev.created_at,
            updated_at=_NOW,
        )
    )
    head = repo.get_knowledge_configuration_head(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert head is not None
    repo.replace_knowledge_configuration_head_if_match(
        expected=head,
        replacement=head.model_copy(
            update={"committed_revision": revision + 1, "updated_at": _NOW}
        ),
    )
    result = detach_service.detach_connection(_detach_cmd(expected_revision=revision + 1))
    assert result.attachment.status is WorkspaceConnectionAttachmentStatusV1.DETACHED


def test_existing_detached_no_op() -> None:
    attach_service, detach_service, _, _, _ = _build_stack()
    revision = _attach(attach_service)
    first = detach_service.detach_connection(_detach_cmd(expected_revision=revision))
    second = detach_service.detach_connection(
        _detach_cmd(expected_revision=first.configuration_revision, idempotency_key_hash="c" * 64)
    )
    assert second.disposition is WorkspaceKnowledgeMutationExecutionDispositionV1.EXISTING_RESULT
    assert second.configuration_revision == first.configuration_revision


def test_committed_replay_after_cascade() -> None:
    attach_service, detach_service, _, _, _ = _build_stack()
    revision = _attach(attach_service)
    first = detach_service.detach_connection(_detach_cmd(expected_revision=revision))
    replay = detach_service.detach_connection(_detach_cmd(expected_revision=revision))
    assert first.disposition is WorkspaceKnowledgeMutationExecutionDispositionV1.APPLIED
    assert replay.disposition is WorkspaceKnowledgeMutationExecutionDispositionV1.COMMITTED_REPLAY
    assert replay.configuration_revision == first.configuration_revision


def test_missing_attachment() -> None:
    _, detach_service, repo, engine, _ = _build_stack()
    with pytest.raises(WorkspaceConnectionAttachmentError) as exc:
        detach_service.detach_connection(_detach_cmd(expected_revision=0))
    assert exc.value.error_code == "connection_attachment_not_found"
    assert not repo.list_knowledge_configuration_mutations(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert repo.get_knowledge_configuration_head(tenant_id=_TENANT, workspace_id=_WORKSPACE) is None


@pytest.mark.parametrize(
    ("port", "error"),
    [
        (_TenantBindingPort(bindings={}), "connection_detach_dependency_resolution_failed"),
        (_TenantBindingPort(fail=True), "connection_detach_dependency_resolution_failed"),
        (
            _TenantBindingPort(
                bindings={"ksb-1": _tenant_binding(tenant_id="other-tenant")}
            ),
            "connection_detach_dependency_resolution_failed",
        ),
        (
            _TenantBindingPort(bindings={"ksb-1": _tenant_binding(binding_id="other-id")}),
            "connection_detach_dependency_resolution_failed",
        ),
        (
            _TenantBindingPort(
                bindings={
                    "ksb-1": type(
                        "Binding",
                        (),
                        {
                            "tenant_id": _TENANT,
                            "binding_id": "ksb-1",
                            "connection_ref": "  ",
                        },
                    )()
                }
            ),
            "connection_detach_dependency_resolution_failed",
        ),
    ],
)
def test_tenant_binding_resolution_failure(port, error) -> None:
    attach_service, detach_service, repo, engine, _ = _build_stack(bindings=port)
    revision = _attach(attach_service)
    repo.put_knowledge_indexed_source_version_if_absent(
        _indexed_row("idx-active", status=WorkspaceIndexedSourceBindingStatusV1.ACTIVE)
    )
    with pytest.raises(WorkspaceConnectionAttachmentError) as exc:
        detach_service.detach_connection(_detach_cmd(expected_revision=revision))
    assert exc.value.error_code == error
    assert not any(
        m.operation is WorkspaceKnowledgeMutationOperationV1.DETACH_CONNECTION
        for m in repo.list_knowledge_configuration_mutations(
            tenant_id=_TENANT, workspace_id=_WORKSPACE
        )
    )


def test_indexed_and_live_cascade() -> None:
    bindings = _TenantBindingPort(
        {
            "ksb-primary": _tenant_binding(binding_id="ksb-primary"),
            "ksb-other": _tenant_binding(binding_id="ksb-other", connection_ref=_CONNECTION_OTHER),
        }
    )
    attach_service, detach_service, repo, _, config_service = _build_stack(bindings=bindings)
    revision = _attach(attach_service)
    rows = [
        _indexed_row("idx-active", status=WorkspaceIndexedSourceBindingStatusV1.ACTIVE, ksb_ref="ksb-primary"),
        _indexed_row("idx-error", status=WorkspaceIndexedSourceBindingStatusV1.ERROR, ksb_ref="ksb-primary"),
        _indexed_row("idx-disabled", status=WorkspaceIndexedSourceBindingStatusV1.DISABLED, ksb_ref="ksb-primary"),
        _indexed_row("idx-unavailable", status=WorkspaceIndexedSourceBindingStatusV1.UNAVAILABLE, ksb_ref="ksb-primary"),
        _indexed_row("idx-other", status=WorkspaceIndexedSourceBindingStatusV1.ACTIVE, ksb_ref="ksb-other"),
        _live_row("live-active", status=LiveAccessBindingStatusV1.ACTIVE),
        _live_row("live-disabled", status=LiveAccessBindingStatusV1.DISABLED),
        _live_row("live-unavailable", status=LiveAccessBindingStatusV1.UNAVAILABLE),
        _live_row("live-revoked", status=LiveAccessBindingStatusV1.REVOKED),
        _live_row("live-other", status=LiveAccessBindingStatusV1.ACTIVE, connection_ref=_CONNECTION_OTHER),
    ]
    for row in rows:
        if hasattr(row, "indexed_source_binding_id"):
            repo.put_knowledge_indexed_source_version_if_absent(row)
        else:
            repo.put_knowledge_live_access_version_if_absent(row)
    result = detach_service.detach_connection(_detach_cmd(expected_revision=revision))
    config = config_service.get_configuration(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert config is not None
    indexed = {b.indexed_source_binding_id: b.status for b in config.indexed_sources}
    live = {b.live_access_binding_id: b.status for b in config.live_access_bindings}
    assert indexed["idx-active"] is WorkspaceIndexedSourceBindingStatusV1.UNAVAILABLE
    assert indexed["idx-error"] is WorkspaceIndexedSourceBindingStatusV1.UNAVAILABLE
    assert indexed["idx-disabled"] is WorkspaceIndexedSourceBindingStatusV1.DISABLED
    assert indexed["idx-unavailable"] is WorkspaceIndexedSourceBindingStatusV1.UNAVAILABLE
    assert indexed["idx-other"] is WorkspaceIndexedSourceBindingStatusV1.ACTIVE
    assert live["live-active"] is LiveAccessBindingStatusV1.UNAVAILABLE
    assert live["live-disabled"] is LiveAccessBindingStatusV1.DISABLED
    assert live["live-unavailable"] is LiveAccessBindingStatusV1.UNAVAILABLE
    assert live["live-revoked"] is LiveAccessBindingStatusV1.REVOKED
    assert live["live-other"] is LiveAccessBindingStatusV1.ACTIVE
    assert result.configuration_revision == revision + 1


def test_non_destructive_detach() -> None:
    bindings = _TenantBindingPort({"ksb-primary": _tenant_binding(binding_id="ksb-primary")})
    attach_service, detach_service, repo, _, config_service = _build_stack(bindings=bindings)
    revision = _attach(attach_service)
    source = WorkspaceSource(
        source_id="source-idx-active",
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        source_type=WorkspaceSourceType.CONNECTED_SOURCE,
        status=WorkspaceSourceStatus.REGISTERED,
        created_at=_NOW,
        knowledge_configuration_creation_mutation_id="seed-mutation",
        knowledge_configuration_visibility_revision=revision,
    )
    repo.put_source_if_absent(source)
    repo.put_knowledge_indexed_source_version_if_absent(
        _indexed_row("idx-active", status=WorkspaceIndexedSourceBindingStatusV1.ACTIVE, ksb_ref="ksb-primary")
    )
    config_before = config_service.get_configuration(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert config_before is not None
    policy_before = config_before.query_policy
    detach_service.detach_connection(_detach_cmd(expected_revision=revision))
    config = config_service.get_configuration(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert config is not None
    assert config.query_policy == policy_before
    assert repo.get_source(tenant_id=_TENANT, workspace_id=_WORKSPACE, source_id=source.source_id) is not None


def test_stable_stage_manifest_hash() -> None:
    manifest_a = detach_connection_stage_manifest_hash(
        attachment_id=_ATTACHMENT_ID,
        connection_ref=_CONNECTION,
        indexed_source_binding_ids=("idx-b", "idx-a"),
        live_access_binding_ids=("live-z", "live-a"),
    )
    manifest_b = detach_connection_stage_manifest_hash(
        attachment_id=_ATTACHMENT_ID,
        connection_ref=_CONNECTION,
        indexed_source_binding_ids=("idx-a", "idx-b"),
        live_access_binding_ids=("live-a", "live-z"),
    )
    assert manifest_a == manifest_b


def test_complete_staged_recovery() -> None:
    attach_service, detach_service, repo, engine, _ = _build_stack(
        mutation_ids=["mutation-attach", "mutation-detach"]
    )
    revision = _attach(attach_service)
    intent = DetachConnectionMutationIntent(
        attachment_id=_ATTACHMENT_ID,
        connection_ref=_CONNECTION,
        indexed_source_binding_ids=(),
        live_access_binding_ids=(),
    )
    manifest = detach_connection_stage_manifest_hash(
        attachment_id=intent.attachment_id,
        connection_ref=intent.connection_ref,
        indexed_source_binding_ids=intent.indexed_source_binding_ids,
        live_access_binding_ids=intent.live_access_binding_ids,
    )
    semantic_hash = connection_attachment_semantic_identity_hash(
        tenant_id=_TENANT, workspace_id=_WORKSPACE, connection_ref=_CONNECTION
    )
    mutation = WorkspaceKnowledgeMutationRecord(
        mutation_id="mutation-detach",
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        operation=WorkspaceKnowledgeMutationOperationV1.DETACH_CONNECTION,
        idempotency_key_hash=_SHA256_B,
        normalized_request_hash=detach_connection_request_hash(
            tenant_id=_TENANT, workspace_id=_WORKSPACE, connection_ref=_CONNECTION
        ),
        semantic_identity_hash=semantic_hash,
        stage_manifest_hash=manifest,
        target_revision=revision + 1,
        status=WorkspaceKnowledgeMutationStatusV1.PREPARED,
        result_entity_type="connection_attachment",
        result_entity_id=_ATTACHMENT_ID,
        created_at=_NOW,
        updated_at=_NOW,
    )
    repo.put_knowledge_configuration_mutation_if_absent(mutation)
    head = repo.get_knowledge_configuration_head(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert head is not None
    repo.replace_knowledge_configuration_head_if_match(
        expected=head,
        replacement=head.model_copy(
            update={
                "pending_revision": revision + 1,
                "pending_mutation_id": "mutation-detach",
                "updated_at": _NOW,
            }
        ),
    )
    _DETACH_HANDLER.stage(
        repository=repo, mutation=mutation, target_revision=revision + 1, intent=intent, now=_NOW
    )
    inspection = _DETACH_HANDLER.inspect_staged(repository=repo, mutation=mutation)
    assert inspection.state is WorkspaceKnowledgeStageStateV1.COMPLETE_VALID
    recovery = engine.recover_workspace_knowledge_mutation(
        tenant_id=_TENANT, workspace_id=_WORKSPACE
    )
    assert recovery.mutation is not None
    assert recovery.mutation.status is WorkspaceKnowledgeMutationStatusV1.COMMITTED
    head = repo.get_knowledge_configuration_head(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert head is not None and head.committed_revision == revision + 1


def test_partial_stage_cleanup() -> None:
    attach_service, _, repo, engine, _ = _build_stack(mutation_ids=["mutation-attach", "mutation-detach"])
    revision = _attach(attach_service)
    intent = DetachConnectionMutationIntent(
        attachment_id=_ATTACHMENT_ID,
        connection_ref=_CONNECTION,
        indexed_source_binding_ids=("idx-missing",),
        live_access_binding_ids=(),
    )
    manifest = detach_connection_stage_manifest_hash(
        attachment_id=intent.attachment_id,
        connection_ref=intent.connection_ref,
        indexed_source_binding_ids=intent.indexed_source_binding_ids,
        live_access_binding_ids=intent.live_access_binding_ids,
    )
    semantic_hash = connection_attachment_semantic_identity_hash(
        tenant_id=_TENANT, workspace_id=_WORKSPACE, connection_ref=_CONNECTION
    )
    mutation = WorkspaceKnowledgeMutationRecord(
        mutation_id="mutation-detach",
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        operation=WorkspaceKnowledgeMutationOperationV1.DETACH_CONNECTION,
        idempotency_key_hash=_SHA256_B,
        normalized_request_hash=detach_connection_request_hash(
            tenant_id=_TENANT, workspace_id=_WORKSPACE, connection_ref=_CONNECTION
        ),
        semantic_identity_hash=semantic_hash,
        stage_manifest_hash=manifest,
        target_revision=revision + 1,
        status=WorkspaceKnowledgeMutationStatusV1.PREPARED,
        created_at=_NOW,
        updated_at=_NOW,
    )
    repo.put_knowledge_configuration_mutation_if_absent(mutation)
    head = repo.get_knowledge_configuration_head(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert head is not None
    repo.replace_knowledge_configuration_head_if_match(
        expected=head,
        replacement=head.model_copy(
            update={
                "pending_revision": revision + 1,
                "pending_mutation_id": "mutation-detach",
                "updated_at": _NOW,
            }
        ),
    )
    from local_workspace_application.workspaces.knowledge_configuration_models import (
        WorkspaceConnectionAttachment,
    )

    config = attach_service._configuration_service.get_configuration(
        tenant_id=_TENANT, workspace_id=_WORKSPACE
    )
    prev = config.connection_attachments[0]
    repo.put_knowledge_connection_attachment_version_if_absent(
        WorkspaceConnectionAttachment(
            attachment_id=prev.attachment_id,
            tenant_id=prev.tenant_id,
            workspace_id=prev.workspace_id,
            connection_ref=prev.connection_ref,
            safe_display_label=prev.safe_display_label,
            status=WorkspaceConnectionAttachmentStatusV1.DETACHED,
            mutation_id="mutation-detach",
            effective_revision=revision + 1,
            created_at=prev.created_at,
            updated_at=_NOW,
        )
    )
    inspection = _DETACH_HANDLER.inspect_staged(repository=repo, mutation=mutation)
    assert inspection.state is WorkspaceKnowledgeStageStateV1.INCOMPLETE_OWNED
    assert _DETACH_HANDLER.cleanup_staged(repository=repo, mutation=mutation, inspection=inspection)
    assert not any(
        row.mutation_id == "mutation-detach" and row.effective_revision == revision + 1
        for row in repo.list_knowledge_connection_attachment_versions(
            tenant_id=_TENANT, workspace_id=_WORKSPACE
        )
    )


def test_ownership_conflict_blocks_cleanup() -> None:
    attach_service, _, repo, _, _ = _build_stack()
    revision = _attach(attach_service)
    mutation = WorkspaceKnowledgeMutationRecord(
        mutation_id="mutation-detach",
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        operation=WorkspaceKnowledgeMutationOperationV1.DETACH_CONNECTION,
        idempotency_key_hash=_SHA256_B,
        normalized_request_hash="d" * 64,
        semantic_identity_hash="e" * 64,
        stage_manifest_hash="f" * 64,
        target_revision=revision + 1,
        status=WorkspaceKnowledgeMutationStatusV1.PREPARED,
        created_at=_NOW,
        updated_at=_NOW,
    )
    from local_workspace_application.workspaces.knowledge_configuration_models import (
        WorkspaceConnectionAttachment,
    )

    repo.put_knowledge_connection_attachment_version_if_absent(
        WorkspaceConnectionAttachment(
            attachment_id=_ATTACHMENT_ID,
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            connection_ref=_CONNECTION,
            safe_display_label="Tampered",
            status=WorkspaceConnectionAttachmentStatusV1.DETACHED,
            mutation_id="mutation-detach",
            effective_revision=revision + 1,
            created_at=_NOW,
            updated_at=_NOW,
        )
    )
    inspection = _DETACH_HANDLER.inspect_staged(repository=repo, mutation=mutation)
    assert inspection.state is WorkspaceKnowledgeStageStateV1.OWNERSHIP_CONFLICT
    assert not _DETACH_HANDLER.cleanup_staged(repository=repo, mutation=mutation, inspection=inspection)


def test_attachment_reconciliation() -> None:
    bindings = _TenantBindingPort({"ksb-primary": _tenant_binding(binding_id="ksb-primary")})
    attach_service, detach_service, repo, _, _ = _build_stack(bindings=bindings)
    revision = _attach(attach_service)
    detach_service.detach_connection(_detach_cmd(expected_revision=revision))
    repo.put_knowledge_indexed_source_version_if_absent(
        _indexed_row("idx-new", status=WorkspaceIndexedSourceBindingStatusV1.ACTIVE, ksb_ref="ksb-primary")
    )
    head = repo.get_knowledge_configuration_head(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    current_revision = head.committed_revision if head else revision + 1
    result = detach_service.detach_connection(
        _detach_cmd(expected_revision=current_revision, idempotency_key_hash="c" * 64)
    )
    assert result.attachment.status is WorkspaceConnectionAttachmentStatusV1.DETACHED
    config = attach_service._configuration_service.get_configuration(
        tenant_id=_TENANT, workspace_id=_WORKSPACE
    )
    assert config is not None
    indexed = {b.indexed_source_binding_id: b.status for b in config.indexed_sources}
    assert indexed["idx-new"] is WorkspaceIndexedSourceBindingStatusV1.UNAVAILABLE
    assert result.configuration_revision == current_revision + 1
