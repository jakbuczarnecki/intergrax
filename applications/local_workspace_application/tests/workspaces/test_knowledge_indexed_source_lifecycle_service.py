# © Artur Czarnecki. All rights reserved.

"""Tests for WorkspaceIndexedSourceLifecycleService."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.conversation_channel.slack.integration import SLACK_CONVERSATION_CHANNEL_PROVIDER_ID
from intergrax.integrations.providers.conversation_channel.slack.knowledge_read import SLACK_CONVERSATION_SOURCE_KIND
from intergrax.runtime.vendor_knowledge.bindings import KnowledgeSourceBinding, KnowledgeSourceBindingStatus, KnowledgeSourceScope
from intergrax.runtime.vendor_knowledge.tenant_connections import SafeTenantConnectionV1, TenantConnectionAdministrativeStatus
from local_workspace_application.workspaces.connected_source_ids import connected_source_id, indexed_source_binding_id
from local_workspace_application.workspaces.knowledge_configuration_handlers import (
    AttachConnectionMutationHandler,
    CreateIndexedSourceMutationHandler,
    DisableIndexedSourceMutationHandler,
)
from local_workspace_application.workspaces.knowledge_configuration_models import (
    IndexedSourceAudienceEligibilityV1,
    IndexedSourceSyncModeV1,
    WorkspaceIndexedSourceBindingStatusV1,
    WorkspaceKnowledgeMutationOperationV1,
)
from local_workspace_application.workspaces.knowledge_configuration_mutation_engine import (
    WorkspaceKnowledgeConfigurationMutationEngine,
    WorkspaceKnowledgeMutationExecutionDispositionV1,
)
from local_workspace_application.workspaces.knowledge_configuration_service import WorkspaceKnowledgeConfigurationService
from local_workspace_application.workspaces.knowledge_connection_attachment_service import (
    AttachWorkspaceConnectionCommand,
    WorkspaceConnectionAttachmentService,
)
from local_workspace_application.workspaces.knowledge_indexed_source_lifecycle_service import (
    ActivateWorkspaceIndexedSourceCommand,
    DisableWorkspaceIndexedSourceCommand,
    WorkspaceIndexedSourceLifecycleError,
    WorkspaceIndexedSourceLifecycleService,
)
from local_workspace_application.workspaces.models import Workspace, WorkspaceSourceType, WorkspaceStatus
from local_workspace_application.workspaces.repository import ManagedWorkspaceRepository
from local_workspace_application.workspaces.service import ManagedWorkspaceService

pytestmark = pytest.mark.unit

_NOW = datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC)
_TENANT, _WORKSPACE = "tenant-a", "workspace-1"
_CONNECTION = "conn.slack"
_KSB_REF = "ksb-1"
_SHA256_A, _SHA256_B, _SHA256_C, _SHA256_D, _SHA256_E = (
    "a" * 64,
    "b" * 64,
    "c" * 64,
    "d" * 64,
    "e" * 64,
)
_BINDING_ID = indexed_source_binding_id(_TENANT, _WORKSPACE, _KSB_REF)
_SOURCE_ID = connected_source_id(_TENANT, _WORKSPACE, _KSB_REF)
_SYNC = IndexedSourceSyncModeV1.FULL
_AUDIENCE = IndexedSourceAudienceEligibilityV1.PERSONAL_ONLY


class _FakeConnectionPort:
    def get_connection(self, *, tenant_id: str, connection_ref: str) -> SafeTenantConnectionV1 | None:
        if tenant_id == _TENANT and connection_ref == _CONNECTION:
            return SafeTenantConnectionV1(
                connection_ref=_CONNECTION,
                tenant_id=_TENANT,
                provider_id="provider.slack",
                integration_kind=IntegrationCategory.CONVERSATION_CHANNEL,
                safe_display_name="Slack",
                administrative_status=TenantConnectionAdministrativeStatus.ACTIVE,
                configuration_version=1,
                connected_principal_ref=None,
                created_at=_NOW,
                updated_at=_NOW,
            )
        return None

    def list_connections(self, *, tenant_id: str, limit: int = 100, administrative_status=None):
        conn = self.get_connection(tenant_id=tenant_id, connection_ref=_CONNECTION)
        return (conn,) if conn else ()


class _TenantBindingPort:
    def __init__(self, *, fail: bool = False) -> None:
        self._fail = fail
        self.call_count = 0

    def get_binding(self, *, tenant_id: str, binding_id: str):
        self.call_count += 1
        if self._fail:
            raise RuntimeError("lookup_failed")
        if tenant_id != _TENANT or binding_id != _KSB_REF:
            return None
        return KnowledgeSourceBinding(
            binding_id=_KSB_REF,
            tenant_id=_TENANT,
            provider_id=SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
            integration_kind=IntegrationCategory.CONVERSATION_CHANNEL,
            source_kind=SLACK_CONVERSATION_SOURCE_KIND,
            connection_ref=_CONNECTION,
            safe_display_name="Slack Binding",
            scope=KnowledgeSourceScope(
                remote_scope_id="scope",
                remote_scope_type="slack_conversation",
                safe_display_name="Slack Binding",
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


def _build_stack(*, port: _TenantBindingPort | None = None):
    store = InMemoryDocumentStore()
    repo = ManagedWorkspaceRepository(store)
    repo.put_workspace(_workspace())
    lookup = ManagedWorkspaceService(repo)
    config = WorkspaceKnowledgeConfigurationService(repo, lookup)
    ids = [f"mutation-{i}" for i in range(1, 9)]
    idx = {"i": 0}

    def _next_id() -> str:
        value = ids[idx["i"]]
        idx["i"] = min(idx["i"] + 1, len(ids) - 1)
        return value

    engine = WorkspaceKnowledgeConfigurationMutationEngine(
        repo,
        lookup,
        config,
        {
            WorkspaceKnowledgeMutationOperationV1.ATTACH_CONNECTION: AttachConnectionMutationHandler(),
            WorkspaceKnowledgeMutationOperationV1.CREATE_INDEXED_SOURCE: CreateIndexedSourceMutationHandler(),
            WorkspaceKnowledgeMutationOperationV1.DISABLE_INDEXED_SOURCE: DisableIndexedSourceMutationHandler(),
        },
        clock=lambda: _NOW,
        mutation_id_factory=_next_id,
    )
    attach = WorkspaceConnectionAttachmentService(
        connection_port=_FakeConnectionPort(),
        configuration_service=config,
        mutation_engine=engine,
    )
    binding_port = port or _TenantBindingPort()
    lifecycle = WorkspaceIndexedSourceLifecycleService(
        repository=repo,
        configuration_service=config,
        mutation_engine=engine,
        tenant_binding_port=binding_port,
    )
    return attach, lifecycle, repo, binding_port


def _attach(attach_svc, *, rev: int = 0) -> int:
    return attach_svc.attach_connection(
        AttachWorkspaceConnectionCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            connection_ref=_CONNECTION,
            expected_revision=rev,
            idempotency_key_hash=_SHA256_A,
        )
    ).configuration_revision


def _activate_cmd(**overrides: object) -> ActivateWorkspaceIndexedSourceCommand:
    payload = {
        "tenant_id": _TENANT,
        "workspace_id": _WORKSPACE,
        "knowledge_source_binding_ref": _KSB_REF,
        "expected_revision": 1,
        "idempotency_key_hash": _SHA256_A,
        "sync_mode": _SYNC,
        "audience_eligibility": _AUDIENCE,
    }
    payload.update(overrides)
    return ActivateWorkspaceIndexedSourceCommand(**payload)


def _disable_cmd(**overrides: object) -> DisableWorkspaceIndexedSourceCommand:
    payload = {
        "tenant_id": _TENANT,
        "workspace_id": _WORKSPACE,
        "indexed_source_binding_id": _BINDING_ID,
        "expected_revision": 2,
        "idempotency_key_hash": _SHA256_C,
    }
    payload.update(overrides)
    return DisableWorkspaceIndexedSourceCommand(**payload)


def _seed_active(lifecycle, attach_svc) -> int:
    rev = _attach(attach_svc)
    lifecycle.activate_indexed_source(_activate_cmd(expected_revision=rev))
    return rev + 1


def test_initial_create_applied_active_source() -> None:
    attach, lifecycle, repo, _ = _build_stack()
    rev = _attach(attach)
    result = lifecycle.activate_indexed_source(_activate_cmd(expected_revision=rev))
    assert result.disposition is WorkspaceKnowledgeMutationExecutionDispositionV1.APPLIED
    assert result.created_new_source is True
    assert result.binding.status is WorkspaceIndexedSourceBindingStatusV1.ACTIVE
    assert result.binding.indexed_source_binding_id == _BINDING_ID
    assert result.binding.source_id == _SOURCE_ID
    source = repo.get_source(tenant_id=_TENANT, workspace_id=_WORKSPACE, source_id=_SOURCE_ID)
    assert source is not None
    assert source.source_type is WorkspaceSourceType.CONNECTED_SOURCE


def test_active_noop_existing_result() -> None:
    attach, lifecycle, _, _ = _build_stack()
    rev = _seed_active(lifecycle, attach)
    result = lifecycle.activate_indexed_source(
        _activate_cmd(expected_revision=rev, idempotency_key_hash=_SHA256_B)
    )
    assert result.disposition is WorkspaceKnowledgeMutationExecutionDispositionV1.EXISTING_RESULT
    assert result.created_new_source is False
    assert result.binding.status is WorkspaceIndexedSourceBindingStatusV1.ACTIVE


def test_exact_replay_committed_replay() -> None:
    attach, lifecycle, _, _ = _build_stack()
    rev = _seed_active(lifecycle, attach)
    replay = lifecycle.activate_indexed_source(_activate_cmd(expected_revision=rev))
    assert replay.disposition is WorkspaceKnowledgeMutationExecutionDispositionV1.COMMITTED_REPLAY
    assert replay.created_new_source is False
    assert replay.binding.indexed_source_binding_id == _BINDING_ID


def test_disable_active_to_disabled_and_noop() -> None:
    attach, lifecycle, _, _ = _build_stack()
    rev = _seed_active(lifecycle, attach)
    disabled = lifecycle.disable_indexed_source(_disable_cmd(expected_revision=rev))
    assert disabled.disposition is WorkspaceKnowledgeMutationExecutionDispositionV1.APPLIED
    assert disabled.binding.status is WorkspaceIndexedSourceBindingStatusV1.DISABLED
    noop = lifecycle.disable_indexed_source(
        _disable_cmd(expected_revision=disabled.configuration_revision, idempotency_key_hash=_SHA256_D)
    )
    assert noop.disposition is WorkspaceKnowledgeMutationExecutionDispositionV1.EXISTING_RESULT
    assert noop.binding.status is WorkspaceIndexedSourceBindingStatusV1.DISABLED


def test_reactivation_after_disable_same_ids() -> None:
    attach, lifecycle, repo, _ = _build_stack()
    rev = _seed_active(lifecycle, attach)
    disabled = lifecycle.disable_indexed_source(_disable_cmd(expected_revision=rev))
    reactivated = lifecycle.activate_indexed_source(
        _activate_cmd(
            expected_revision=disabled.configuration_revision,
            idempotency_key_hash=_SHA256_E,
        )
    )
    assert reactivated.disposition is WorkspaceKnowledgeMutationExecutionDispositionV1.APPLIED
    assert reactivated.created_new_source is False
    assert reactivated.binding.indexed_source_binding_id == _BINDING_ID
    assert reactivated.binding.source_id == _SOURCE_ID
    assert reactivated.binding.status is WorkspaceIndexedSourceBindingStatusV1.ACTIVE
    assert repo.get_source(tenant_id=_TENANT, workspace_id=_WORKSPACE, source_id=_SOURCE_ID) is not None


def test_activation_replay_after_disable_skips_tenant_port() -> None:
    attach, lifecycle, _, binding_port = _build_stack()
    rev = _seed_active(lifecycle, attach)
    lifecycle.disable_indexed_source(_disable_cmd(expected_revision=rev))
    binding_port._fail = True
    calls_before = binding_port.call_count
    replay = lifecycle.activate_indexed_source(_activate_cmd(expected_revision=rev))
    assert replay.disposition is WorkspaceKnowledgeMutationExecutionDispositionV1.COMMITTED_REPLAY
    assert replay.binding.status is WorkspaceIndexedSourceBindingStatusV1.ACTIVE
    assert binding_port.call_count == calls_before


def test_disable_replay_after_reactivation() -> None:
    attach, lifecycle, _, _ = _build_stack()
    rev = _seed_active(lifecycle, attach)
    disabled = lifecycle.disable_indexed_source(_disable_cmd(expected_revision=rev))
    lifecycle.activate_indexed_source(
        _activate_cmd(
            expected_revision=disabled.configuration_revision,
            idempotency_key_hash=_SHA256_E,
        )
    )
    replay = lifecycle.disable_indexed_source(_disable_cmd(expected_revision=rev))
    assert replay.disposition is WorkspaceKnowledgeMutationExecutionDispositionV1.COMMITTED_REPLAY
    assert replay.binding.status is WorkspaceIndexedSourceBindingStatusV1.DISABLED


def test_workspace_not_found_on_activate() -> None:
    _, lifecycle, _, _ = _build_stack()
    with pytest.raises(WorkspaceIndexedSourceLifecycleError, match="workspace_not_found"):
        lifecycle.activate_indexed_source(
            _activate_cmd(workspace_id="missing-workspace", expected_revision=0)
        )


def test_workspace_not_found_on_disable() -> None:
    _, lifecycle, _, _ = _build_stack()
    with pytest.raises(WorkspaceIndexedSourceLifecycleError, match="workspace_not_found"):
        lifecycle.disable_indexed_source(
            _disable_cmd(workspace_id="missing-workspace", expected_revision=0)
        )


def test_indexed_source_not_found_on_disable() -> None:
    attach, lifecycle, _, _ = _build_stack()
    rev = _attach(attach)
    with pytest.raises(WorkspaceIndexedSourceLifecycleError, match="indexed_source_not_found"):
        lifecycle.disable_indexed_source(
            _disable_cmd(
                indexed_source_binding_id="idx:missing",
                expected_revision=rev,
            )
        )
