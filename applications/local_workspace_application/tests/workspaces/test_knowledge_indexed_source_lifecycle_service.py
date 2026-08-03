# © Artur Czarnecki. All rights reserved.

"""Tests for WorkspaceIndexedSourceLifecycleService."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.integrations.contracts.document_store import DocumentRecord
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.conversation_channel.slack.integration import SLACK_CONVERSATION_CHANNEL_PROVIDER_ID
from intergrax.integrations.providers.conversation_channel.slack.knowledge_read import SLACK_CONVERSATION_SOURCE_KIND
from intergrax.runtime.vendor_knowledge.bindings import KnowledgeSourceBinding, KnowledgeSourceBindingStatus, KnowledgeSourceScope
from intergrax.runtime.vendor_knowledge.tenant_connections import SafeTenantConnectionV1, TenantConnectionAdministrativeStatus
from local_workspace_application.workspaces.connected_source_ids import (
    connected_source_id,
    connected_source_id_from_semantic_hash,
    indexed_source_binding_id,
    workspace_indexed_source_semantic_hash,
)
from local_workspace_application.workspaces.connected_source_source_projection import (
    ConnectedSourceOriginValidationError,
    validate_connected_source_durable_origin,
)
from local_workspace_application.workspaces.knowledge_configuration_handlers import (
    AttachConnectionMutationHandler,
    CreateIndexedSourceMutationHandler,
    CreateIndexedSourceMutationIntent,
    DisableIndexedSourceMutationHandler,
    DisableIndexedSourceMutationIntent,
)
from local_workspace_application.workspaces.knowledge_configuration_hashing import (
    normalize_create_indexed_source_request_hash,
    normalize_disable_indexed_source_request_hash,
    semantic_identity_hash_for_create_indexed_source,
    semantic_identity_hash_for_disable_indexed_source,
)
from local_workspace_application.workspaces.knowledge_configuration_models import (
    IndexedSourceAudienceEligibilityV1,
    IndexedSourceSyncModeV1,
    WorkspaceIndexedSourceBinding,
    WorkspaceIndexedSourceBindingStatusV1,
    WorkspaceKnowledgeMutationOperationV1,
    WorkspaceKnowledgeMutationRecord,
    WorkspaceKnowledgeMutationStatusV1,
)
from local_workspace_application.workspaces.knowledge_configuration_mutation_engine import (
    WorkspaceKnowledgeConfigurationMutationEngine,
    WorkspaceKnowledgeConfigurationMutationError,
    WorkspaceKnowledgeMutationExecutionDispositionV1,
    WorkspaceKnowledgeMutationRecoveryDispositionV1,
    WorkspaceKnowledgeStageStateV1,
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
from local_workspace_application.workspaces.models import (
    Workspace,
    WorkspaceSource,
    WorkspaceSourceStatus,
    WorkspaceSourceType,
    WorkspaceStatus,
)
from local_workspace_application.workspaces.repository import ManagedWorkspaceRepository
from local_workspace_application.workspaces.service import ManagedWorkspaceService

pytestmark = pytest.mark.unit

_NOW = datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC)
_TENANT, _WORKSPACE = "tenant-a", "workspace-1"
_CONNECTION = "conn.slack"
_CONNECTION_OTHER = "conn.slack.other"
_CONNECTION_THIRD = "conn.slack.third"
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


_CREATE_HANDLER = CreateIndexedSourceMutationHandler()
_DISABLE_HANDLER = DisableIndexedSourceMutationHandler()


class _FakeConnectionPort:
    def get_connection(self, *, tenant_id: str, connection_ref: str) -> SafeTenantConnectionV1 | None:
        refs = {_CONNECTION, _CONNECTION_OTHER, _CONNECTION_THIRD}
        if tenant_id == _TENANT and connection_ref in refs:
            return SafeTenantConnectionV1(
                connection_ref=connection_ref,
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
        return tuple(
            self.get_connection(tenant_id=tenant_id, connection_ref=ref)
            for ref in (_CONNECTION, _CONNECTION_OTHER, _CONNECTION_THIRD)
        )


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


def _build_stack(*, port: _TenantBindingPort | None = None, mutation_ids: list[str] | None = None):
    store = InMemoryDocumentStore()
    repo = ManagedWorkspaceRepository(store)
    repo.put_workspace(_workspace())
    lookup = ManagedWorkspaceService(repo)
    config = WorkspaceKnowledgeConfigurationService(repo, lookup)
    ids = mutation_ids or [f"mutation-{i}" for i in range(1, 9)]
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
    return attach, lifecycle, repo, binding_port, engine


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
    attach, lifecycle, repo, _, _ = _build_stack()
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
    attach, lifecycle, _, _, _ = _build_stack()
    rev = _seed_active(lifecycle, attach)
    result = lifecycle.activate_indexed_source(
        _activate_cmd(expected_revision=rev, idempotency_key_hash=_SHA256_B)
    )
    assert result.disposition is WorkspaceKnowledgeMutationExecutionDispositionV1.EXISTING_RESULT
    assert result.created_new_source is False
    assert result.binding.status is WorkspaceIndexedSourceBindingStatusV1.ACTIVE


def test_exact_replay_committed_replay() -> None:
    attach, lifecycle, _, _, _ = _build_stack()
    rev = _seed_active(lifecycle, attach)
    replay = lifecycle.activate_indexed_source(_activate_cmd(expected_revision=rev))
    assert replay.disposition is WorkspaceKnowledgeMutationExecutionDispositionV1.COMMITTED_REPLAY
    assert replay.created_new_source is False
    assert replay.binding.indexed_source_binding_id == _BINDING_ID


def test_disable_active_to_disabled_and_noop() -> None:
    attach, lifecycle, _, _, _ = _build_stack()
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
    attach, lifecycle, repo, _, _ = _build_stack()
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
    attach, lifecycle, _, binding_port, _ = _build_stack()
    rev = _seed_active(lifecycle, attach)
    lifecycle.disable_indexed_source(_disable_cmd(expected_revision=rev))
    binding_port._fail = True
    calls_before = binding_port.call_count
    replay = lifecycle.activate_indexed_source(_activate_cmd(expected_revision=rev))
    assert replay.disposition is WorkspaceKnowledgeMutationExecutionDispositionV1.COMMITTED_REPLAY
    assert replay.binding.status is WorkspaceIndexedSourceBindingStatusV1.ACTIVE
    assert binding_port.call_count == calls_before


def test_disable_replay_after_reactivation() -> None:
    attach, lifecycle, _, _, _ = _build_stack()
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
    _, lifecycle, _, _, _ = _build_stack()
    with pytest.raises(WorkspaceIndexedSourceLifecycleError, match="workspace_not_found"):
        lifecycle.activate_indexed_source(
            _activate_cmd(workspace_id="missing-workspace", expected_revision=0)
        )


def test_workspace_not_found_on_disable() -> None:
    _, lifecycle, _, _, _ = _build_stack()
    with pytest.raises(WorkspaceIndexedSourceLifecycleError, match="workspace_not_found"):
        lifecycle.disable_indexed_source(
            _disable_cmd(workspace_id="missing-workspace", expected_revision=0)
        )


def _attach_other(attach_svc, *, rev: int, idempotency_key_hash: str = _SHA256_B) -> int:
    return attach_svc.attach_connection(
        AttachWorkspaceConnectionCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            connection_ref=_CONNECTION_OTHER,
            expected_revision=rev,
            idempotency_key_hash=idempotency_key_hash,
        )
    ).configuration_revision


def _attach_third(attach_svc, *, rev: int, idempotency_key_hash: str) -> int:
    return attach_svc.attach_connection(
        AttachWorkspaceConnectionCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            connection_ref=_CONNECTION_THIRD,
            expected_revision=rev,
            idempotency_key_hash=idempotency_key_hash,
        )
    ).configuration_revision


def _pending_head(repo, *, revision: int, mutation_id: str) -> None:
    head = repo.get_knowledge_configuration_head(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert head is not None
    repo.replace_knowledge_configuration_head_if_match(
        expected=head,
        replacement=head.model_copy(
            update={
                "pending_revision": revision,
                "pending_mutation_id": mutation_id,
                "updated_at": _NOW,
            }
        ),
    )


def _create_intent(**overrides: object) -> CreateIndexedSourceMutationIntent:
    payload = {
        "knowledge_source_binding_ref": _KSB_REF,
        "sync_mode": _SYNC,
        "audience_eligibility": _AUDIENCE,
        "cached_safe_display_label": "Slack Binding",
    }
    payload.update(overrides)
    return CreateIndexedSourceMutationIntent(**payload)


def _create_mutation(
    repo,
    *,
    revision: int,
    mutation_id: str = "mutation-create",
    idem: str = _SHA256_A,
    intent: CreateIndexedSourceMutationIntent | None = None,
) -> tuple[WorkspaceKnowledgeMutationRecord, CreateIndexedSourceMutationIntent]:
    intent = intent or _create_intent()
    binding_ref = intent.knowledge_source_binding_ref.strip()
    req = normalize_create_indexed_source_request_hash(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        knowledge_source_binding_ref=binding_ref,
        sync_mode=intent.sync_mode,
        audience_eligibility=intent.audience_eligibility,
    )
    sem = semantic_identity_hash_for_create_indexed_source(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        knowledge_source_binding_ref=binding_ref,
    )
    mutation = WorkspaceKnowledgeMutationRecord(
        mutation_id=mutation_id,
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        operation=WorkspaceKnowledgeMutationOperationV1.CREATE_INDEXED_SOURCE,
        idempotency_key_hash=idem,
        normalized_request_hash=req,
        semantic_identity_hash=sem,
        target_revision=revision,
        status=WorkspaceKnowledgeMutationStatusV1.PREPARED,
        result_entity_type="indexed_source_binding",
        result_entity_id=indexed_source_binding_id(_TENANT, _WORKSPACE, binding_ref),
        created_at=_NOW,
        updated_at=_NOW,
    )
    repo.put_knowledge_configuration_mutation_if_absent(mutation)
    _pending_head(repo, revision=revision, mutation_id=mutation_id)
    return mutation, intent


def _connected_source_row(
    *,
    source_id: str = _SOURCE_ID,
    mutation_id: str,
    revision: int,
    status: WorkspaceSourceStatus = WorkspaceSourceStatus.REGISTERED,
) -> WorkspaceSource:
    return WorkspaceSource(
        source_id=source_id,
        workspace_id=_WORKSPACE,
        tenant_id=_TENANT,
        source_type=WorkspaceSourceType.CONNECTED_SOURCE,
        path="",
        recursive=False,
        status=status,
        created_at=_NOW,
        knowledge_configuration_creation_mutation_id=mutation_id,
        knowledge_configuration_visibility_revision=revision,
    )


def _binding_row(
    *,
    binding_id: str = _BINDING_ID,
    source_id: str = _SOURCE_ID,
    mutation_id: str,
    revision: int,
    status: WorkspaceIndexedSourceBindingStatusV1 = WorkspaceIndexedSourceBindingStatusV1.ACTIVE,
) -> WorkspaceIndexedSourceBinding:
    semantic = workspace_indexed_source_semantic_hash(_TENANT, _WORKSPACE, _KSB_REF)
    return WorkspaceIndexedSourceBinding(
        indexed_source_binding_id=binding_id,
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        knowledge_source_binding_ref=_KSB_REF,
        source_id=source_id,
        sync_mode=_SYNC,
        status=status,
        audience_eligibility=_AUDIENCE,
        mutation_id=mutation_id,
        effective_revision=revision,
        semantic_identity_hash=semantic,
        created_at=_NOW,
        updated_at=_NOW,
        cached_safe_display_label="Slack Binding",
    )


def test_revision_gap_disable_and_reactivation() -> None:
    attach, lifecycle, repo, _, _ = _build_stack()
    create_revision = _attach(attach)
    lifecycle.activate_indexed_source(_activate_cmd(expected_revision=create_revision))
    first_unrelated_revision = _attach_other(attach, rev=create_revision + 1)
    source_before = repo.get_source(tenant_id=_TENANT, workspace_id=_WORKSPACE, source_id=_SOURCE_ID)
    disabled = lifecycle.disable_indexed_source(
        _disable_cmd(expected_revision=first_unrelated_revision, idempotency_key_hash=_SHA256_C)
    )
    disable_revision = disabled.configuration_revision
    assert disabled.disposition is WorkspaceKnowledgeMutationExecutionDispositionV1.APPLIED
    assert disabled.binding.status is WorkspaceIndexedSourceBindingStatusV1.DISABLED
    assert disabled.binding.effective_revision == disable_revision
    assert repo.get_source(tenant_id=_TENANT, workspace_id=_WORKSPACE, source_id=_SOURCE_ID) == source_before
    second_unrelated_revision = _attach_third(
        attach, rev=disable_revision, idempotency_key_hash="f" * 64,
    )
    reactivated = lifecycle.activate_indexed_source(
        _activate_cmd(
            expected_revision=second_unrelated_revision,
            idempotency_key_hash=_SHA256_E,
        )
    )
    reactivate_revision = reactivated.configuration_revision
    assert reactivated.disposition is WorkspaceKnowledgeMutationExecutionDispositionV1.APPLIED
    assert reactivated.binding.indexed_source_binding_id == _BINDING_ID
    assert reactivated.binding.source_id == _SOURCE_ID
    assert reactivated.binding.status is WorkspaceIndexedSourceBindingStatusV1.ACTIVE
    assert reactivated.binding.effective_revision == reactivate_revision
    initial_commit_revision = create_revision + 1
    assert (
        first_unrelated_revision,
        disable_revision,
        second_unrelated_revision,
        reactivate_revision,
    ) == tuple(initial_commit_revision + offset for offset in (1, 2, 3, 4))
    assert repo.get_source(tenant_id=_TENANT, workspace_id=_WORKSPACE, source_id=_SOURCE_ID) == source_before


def test_disable_replay_skips_configuration_load() -> None:
    attach, lifecycle, repo, binding_port, _ = _build_stack()
    rev = _seed_active(lifecycle, attach)
    lifecycle.disable_indexed_source(_disable_cmd(expected_revision=rev, idempotency_key_hash=_SHA256_C))
    lifecycle.activate_indexed_source(_activate_cmd(expected_revision=rev + 1, idempotency_key_hash=_SHA256_E))
    active_before = repo.list_knowledge_indexed_source_versions(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    config_calls = 0

    class _FailConfig:
        def get_configuration(self, **kwargs):
            nonlocal config_calls
            config_calls += 1
            raise RuntimeError("config_forbidden")

    lifecycle._configuration_service = _FailConfig()  # type: ignore[assignment]
    binding_port._fail = True
    port_calls_before = binding_port.call_count
    replay = lifecycle.disable_indexed_source(_disable_cmd(expected_revision=rev, idempotency_key_hash=_SHA256_C))
    assert replay.disposition is WorkspaceKnowledgeMutationExecutionDispositionV1.COMMITTED_REPLAY
    assert replay.binding.status is WorkspaceIndexedSourceBindingStatusV1.DISABLED
    assert replay.configuration_revision == rev + 1 and config_calls == 0
    assert binding_port.call_count == port_calls_before
    assert repo.list_knowledge_indexed_source_versions(tenant_id=_TENANT, workspace_id=_WORKSPACE) == active_before


def test_future_owned_binding_row_is_ownership_conflict() -> None:
    attach, _, repo, _, engine = _build_stack(mutation_ids=["mutation-attach", "mutation-create"])
    rev = _attach(attach)
    mutation, intent = _create_mutation(repo, revision=rev + 1, mutation_id="mutation-create")
    repo.put_source_if_absent(_connected_source_row(mutation_id=mutation.mutation_id, revision=rev + 1))
    repo.put_knowledge_indexed_source_version_if_absent(
        _binding_row(mutation_id=mutation.mutation_id, revision=rev + 1)
    )
    repo.put_knowledge_indexed_source_version_if_absent(
        _binding_row(mutation_id=mutation.mutation_id, revision=rev + 2)
    )
    inspection = _CREATE_HANDLER.inspect_staged(repository=repo, mutation=mutation)
    assert inspection.state is WorkspaceKnowledgeStageStateV1.OWNERSHIP_CONFLICT
    with pytest.raises(WorkspaceKnowledgeConfigurationMutationError, match="configuration_recovery_required"):
        engine.recover_workspace_knowledge_mutation(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    head = repo.get_knowledge_configuration_head(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert head is not None
    assert head.committed_revision == rev
    assert head.pending_mutation_id == "mutation-create"
    assert len([
        b for b in repo.list_knowledge_indexed_source_versions(
            tenant_id=_TENANT, workspace_id=_WORKSPACE,
        )
        if b.mutation_id == mutation.mutation_id and b.effective_revision == rev + 2
    ]) == 1


def test_source_only_partial_recovery_aborts() -> None:
    attach, _, repo, _, engine = _build_stack(mutation_ids=["mutation-attach", "mutation-create"])
    rev = _attach(attach)
    mutation, _ = _create_mutation(repo, revision=rev + 1, mutation_id="mutation-create")
    repo.put_source_if_absent(_connected_source_row(mutation_id=mutation.mutation_id, revision=rev + 1))
    inspection = _CREATE_HANDLER.inspect_staged(repository=repo, mutation=mutation)
    assert inspection.state is WorkspaceKnowledgeStageStateV1.INCOMPLETE_OWNED
    recovery = engine.recover_workspace_knowledge_mutation(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert recovery.disposition is WorkspaceKnowledgeMutationRecoveryDispositionV1.ABORTED
    head = repo.get_knowledge_configuration_head(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert head is not None
    assert head.committed_revision == rev
    assert head.pending_mutation_id is None
    assert repo.get_source(tenant_id=_TENANT, workspace_id=_WORKSPACE, source_id=_SOURCE_ID) is None


def test_binding_only_partial_recovery_aborts() -> None:
    attach, _, repo, _, engine = _build_stack(mutation_ids=["mutation-attach", "mutation-create"])
    rev = _attach(attach)
    other_source = _connected_source_row(
        source_id="src:connected:other",
        mutation_id="other-mutation",
        revision=rev,
    )
    repo.put_source_if_absent(other_source)
    mutation, _ = _create_mutation(repo, revision=rev + 1, mutation_id="mutation-create")
    repo.put_knowledge_indexed_source_version_if_absent(
        _binding_row(mutation_id=mutation.mutation_id, revision=rev + 1)
    )
    inspection = _CREATE_HANDLER.inspect_staged(repository=repo, mutation=mutation)
    assert inspection.state is WorkspaceKnowledgeStageStateV1.INCOMPLETE_OWNED
    recovery = engine.recover_workspace_knowledge_mutation(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert recovery.disposition is WorkspaceKnowledgeMutationRecoveryDispositionV1.ABORTED
    assert repo.get_source(tenant_id=_TENANT, workspace_id=_WORKSPACE, source_id="src:connected:other") is not None


def test_corrupt_source_only_partial_stays_recovery_required() -> None:
    attach, _, repo, _, engine = _build_stack(mutation_ids=["mutation-attach", "mutation-create"])
    rev = _attach(attach)
    mutation, _ = _create_mutation(repo, revision=rev + 1, mutation_id="mutation-create")
    wrong_id = f"src:connected:{'1' * 32}"
    corrupt = _connected_source_row(
        source_id=wrong_id,
        mutation_id=mutation.mutation_id,
        revision=rev + 1,
    )
    repo.put_source_if_absent(corrupt)
    inspection = _CREATE_HANDLER.inspect_staged(repository=repo, mutation=mutation)
    assert inspection.state is WorkspaceKnowledgeStageStateV1.OWNERSHIP_CONFLICT
    with pytest.raises(WorkspaceKnowledgeConfigurationMutationError, match="configuration_recovery_required"):
        engine.recover_workspace_knowledge_mutation(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert repo.get_source(tenant_id=_TENANT, workspace_id=_WORKSPACE, source_id=wrong_id) is not None


def test_wrong_revision_source_stays_recovery_required() -> None:
    attach, _, repo, _, engine = _build_stack(mutation_ids=["mutation-attach", "mutation-create"])
    rev = _attach(attach)
    mutation, _ = _create_mutation(repo, revision=rev + 1, mutation_id="mutation-create")
    repo.put_source_if_absent(
        _connected_source_row(mutation_id=mutation.mutation_id, revision=rev + 2)
    )
    inspection = _CREATE_HANDLER.inspect_staged(repository=repo, mutation=mutation)
    assert inspection.state is WorkspaceKnowledgeStageStateV1.OWNERSHIP_CONFLICT
    with pytest.raises(WorkspaceKnowledgeConfigurationMutationError, match="configuration_recovery_required"):
        engine.recover_workspace_knowledge_mutation(tenant_id=_TENANT, workspace_id=_WORKSPACE)


_CORRUPT_INITIAL_CASES = [
    ({"semantic_identity_hash": "d" * 64}, None), ({"status": WorkspaceIndexedSourceBindingStatusV1.DISABLED}, None),
    ({"tenant_id": "tenant-other"}, None), ({"workspace_id": "workspace-other"}, None),
    ({"source_id": f"src:connected:{'2' * 32}"}, None), ({"created_at": datetime(2024, 1, 1, tzinfo=UTC)}, None),
    ({"updated_at": datetime(2024, 1, 2, tzinfo=UTC)}, None), (None, {"source_id": f"src:connected:{'3' * 32}"}),
    (None, {"tenant_id": "tenant-other"}), (None, {"workspace_id": "workspace-other"}),
    (None, {"status": WorkspaceSourceStatus.READY}), (None, {"status": WorkspaceSourceStatus.ERROR}),
    (None, {"last_sync_at": _NOW}), (None, {"source_type": WorkspaceSourceType.WEB_RESOURCE}),
    (None, {"created_at": datetime(2024, 1, 1, tzinfo=UTC)}),
]


def _patch_store_field(repo, *, partition: str, row_key: str, updates: dict) -> None:
    record = repo.document_store.get(partition, row_key)
    assert record is not None
    data = dict(record.data)
    data.update({k: v.value if hasattr(v, "value") else v for k, v in updates.items()})
    repo.document_store.put(DocumentRecord(partition_key=partition, row_key=row_key, data=data))


def _disable_mutation(repo, *, revision: int, mutation_id: str = "mutation-disable") -> WorkspaceKnowledgeMutationRecord:
    mutation = WorkspaceKnowledgeMutationRecord(
        mutation_id=mutation_id, tenant_id=_TENANT, workspace_id=_WORKSPACE,
        operation=WorkspaceKnowledgeMutationOperationV1.DISABLE_INDEXED_SOURCE,
        idempotency_key_hash=_SHA256_C,
        normalized_request_hash=normalize_disable_indexed_source_request_hash(
            tenant_id=_TENANT, workspace_id=_WORKSPACE, indexed_source_binding_id=_BINDING_ID,
        ),
        semantic_identity_hash=semantic_identity_hash_for_disable_indexed_source(
            tenant_id=_TENANT, workspace_id=_WORKSPACE, knowledge_source_binding_ref=_KSB_REF,
        ),
        target_revision=revision, status=WorkspaceKnowledgeMutationStatusV1.PREPARED,
        result_entity_type="indexed_source_binding", result_entity_id=_BINDING_ID,
        created_at=_NOW, updated_at=_NOW,
    )
    repo.put_knowledge_configuration_mutation_if_absent(mutation)
    _pending_head(repo, revision=revision, mutation_id=mutation_id)
    return mutation


def _replace_owned_binding(repo, mutation, **updates) -> None:
    binding = next(
        b for b in repo.list_knowledge_indexed_source_versions(tenant_id=_TENANT, workspace_id=_WORKSPACE)
        if b.mutation_id == mutation.mutation_id
    )
    row_key = f"{_WORKSPACE}:{binding.indexed_source_binding_id}:rev:{binding.effective_revision:020d}"
    partition = f"lkw.managed_workspace:{_TENANT}:knowledge_configuration_indexed_source"
    if {"tenant_id", "workspace_id"} & updates.keys():
        _patch_store_field(repo, partition=partition, row_key=row_key, updates=updates)
        return
    repo.delete_knowledge_indexed_source_version_if_match(binding)
    repo.put_knowledge_indexed_source_version_if_absent(binding.model_copy(update=updates))


def _replace_owned_source(repo, mutation, **updates) -> None:
    source = next(
        s for s in repo.list_sources(tenant_id=_TENANT, workspace_id=_WORKSPACE)
        if s.knowledge_configuration_creation_mutation_id == mutation.mutation_id
    )
    row_key = f"{_WORKSPACE}:{source.source_id}"
    if {"tenant_id", "workspace_id", "source_type"} & updates.keys():
        _patch_store_field(
            repo, partition=f"lkw.managed_workspace:{_TENANT}:source", row_key=row_key, updates=updates,
        )
        return
    repo.delete_source_if_match(source)
    repo.put_source_if_absent(source.model_copy(update=updates))


def _assert_corrupt_initial_recovery_blocked(repo, engine, mutation, *, committed_revision: int) -> None:
    assert _CREATE_HANDLER.inspect_staged(repository=repo, mutation=mutation).state is (
        WorkspaceKnowledgeStageStateV1.OWNERSHIP_CONFLICT
    )
    with pytest.raises(WorkspaceKnowledgeConfigurationMutationError, match="configuration_recovery_required"):
        engine.recover_workspace_knowledge_mutation(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    head = repo.get_knowledge_configuration_head(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert head is not None and head.committed_revision == committed_revision
    assert head.pending_mutation_id == mutation.mutation_id and next(
        m for m in repo.list_knowledge_configuration_mutations(tenant_id=_TENANT, workspace_id=_WORKSPACE)
        if m.mutation_id == mutation.mutation_id
    ).status is WorkspaceKnowledgeMutationStatusV1.RECOVERY_REQUIRED


@pytest.mark.parametrize(("binding_updates", "source_updates"), _CORRUPT_INITIAL_CASES)
def test_corrupt_initial_complete_stage_blocks_recovery(binding_updates, source_updates) -> None:
    attach, _, repo, _, engine = _build_stack(mutation_ids=["mutation-attach", "mutation-create"])
    rev = _attach(attach)
    mutation, intent = _create_mutation(repo, revision=rev + 1, mutation_id="mutation-create")
    _CREATE_HANDLER.stage(
        repository=repo, mutation=mutation, target_revision=mutation.target_revision, intent=intent, now=_NOW,
    )
    if binding_updates:
        _replace_owned_binding(repo, mutation, **binding_updates)
    if source_updates:
        _replace_owned_source(repo, mutation, **source_updates)
    _assert_corrupt_initial_recovery_blocked(repo, engine, mutation, committed_revision=rev)


def test_complete_reactivation_recovery_commits() -> None:
    attach, lifecycle, repo, _, engine = _build_stack(
        mutation_ids=["mutation-attach", "mutation-create", "mutation-disable", "mutation-reactivate"],
    )
    rev = _seed_active(lifecycle, attach)
    reactivate_rev = lifecycle.disable_indexed_source(_disable_cmd(expected_revision=rev)).configuration_revision + 1
    source_before = repo.get_source(tenant_id=_TENANT, workspace_id=_WORKSPACE, source_id=_SOURCE_ID)
    mutation, intent = _create_mutation(
        repo, revision=reactivate_rev, mutation_id="mutation-reactivate", idem=_SHA256_E,
    )
    _CREATE_HANDLER.stage(
        repository=repo, mutation=mutation, target_revision=reactivate_rev, intent=intent, now=_NOW,
    )
    assert _CREATE_HANDLER.inspect_staged(repository=repo, mutation=mutation).state is (
        WorkspaceKnowledgeStageStateV1.COMPLETE_VALID
    )
    assert not any(
        s.knowledge_configuration_creation_mutation_id == mutation.mutation_id
        for s in repo.list_sources(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    )
    assert engine.recover_workspace_knowledge_mutation(
        tenant_id=_TENANT, workspace_id=_WORKSPACE,
    ).disposition is WorkspaceKnowledgeMutationRecoveryDispositionV1.COMMITTED
    binding = next(
        b for b in repo.list_knowledge_indexed_source_versions(tenant_id=_TENANT, workspace_id=_WORKSPACE)
        if b.effective_revision == reactivate_rev
    )
    head = repo.get_knowledge_configuration_head(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert binding.status is WorkspaceIndexedSourceBindingStatusV1.ACTIVE
    assert binding.indexed_source_binding_id == _BINDING_ID and binding.source_id == _SOURCE_ID
    assert repo.get_source(tenant_id=_TENANT, workspace_id=_WORKSPACE, source_id=_SOURCE_ID) == source_before
    assert head is not None and head.committed_revision == reactivate_rev and head.pending_mutation_id is None


def test_complete_disable_recovery_commits() -> None:
    attach, lifecycle, repo, _, engine = _build_stack(
        mutation_ids=["mutation-attach", "mutation-create", "mutation-disable"],
    )
    rev = _seed_active(lifecycle, attach)
    source_before = repo.get_source(tenant_id=_TENANT, workspace_id=_WORKSPACE, source_id=_SOURCE_ID)
    disable_rev = rev + 1
    mutation = _disable_mutation(repo, revision=disable_rev, mutation_id="mutation-disable")
    _DISABLE_HANDLER.stage(
        repository=repo, mutation=mutation, target_revision=disable_rev,
        intent=DisableIndexedSourceMutationIntent(
            indexed_source_binding_id=_BINDING_ID, knowledge_source_binding_ref=_KSB_REF,
        ),
        now=_NOW,
    )
    assert _DISABLE_HANDLER.inspect_staged(repository=repo, mutation=mutation).state is (
        WorkspaceKnowledgeStageStateV1.COMPLETE_VALID
    )
    assert engine.recover_workspace_knowledge_mutation(
        tenant_id=_TENANT, workspace_id=_WORKSPACE,
    ).disposition is WorkspaceKnowledgeMutationRecoveryDispositionV1.COMMITTED
    binding = next(
        b for b in repo.list_knowledge_indexed_source_versions(tenant_id=_TENANT, workspace_id=_WORKSPACE)
        if b.effective_revision == disable_rev
    )
    head = repo.get_knowledge_configuration_head(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert binding.status is WorkspaceIndexedSourceBindingStatusV1.DISABLED
    assert binding.indexed_source_binding_id == _BINDING_ID and binding.source_id == _SOURCE_ID
    assert repo.get_source(tenant_id=_TENANT, workspace_id=_WORKSPACE, source_id=_SOURCE_ID) == source_before
    assert head.committed_revision == disable_rev and head.pending_mutation_id is None


def test_complete_initial_recovery_commits() -> None:
    attach, _, repo, _, engine = _build_stack(mutation_ids=["mutation-attach", "mutation-create"])
    rev = _attach(attach)
    mutation, intent = _create_mutation(repo, revision=rev + 1, mutation_id="mutation-create")
    _CREATE_HANDLER.stage(
        repository=repo, mutation=mutation, target_revision=rev + 1, intent=intent, now=_NOW,
    )
    assert _CREATE_HANDLER.inspect_staged(repository=repo, mutation=mutation).state is (
        WorkspaceKnowledgeStageStateV1.COMPLETE_VALID
    )
    assert engine.recover_workspace_knowledge_mutation(
        tenant_id=_TENANT, workspace_id=_WORKSPACE,
    ).disposition is WorkspaceKnowledgeMutationRecoveryDispositionV1.COMMITTED
    head = repo.get_knowledge_configuration_head(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    binding = next(
        b for b in repo.list_knowledge_indexed_source_versions(tenant_id=_TENANT, workspace_id=_WORKSPACE)
        if b.mutation_id == mutation.mutation_id
    )
    source = repo.get_source(tenant_id=_TENANT, workspace_id=_WORKSPACE, source_id=_SOURCE_ID)
    assert head is not None and head.committed_revision == rev + 1 and head.pending_mutation_id is None
    assert binding.status is WorkspaceIndexedSourceBindingStatusV1.ACTIVE
    assert binding.semantic_identity_hash == mutation.semantic_identity_hash
    assert binding.indexed_source_binding_id == _BINDING_ID and binding.source_id == _SOURCE_ID
    assert source is not None and source.status is WorkspaceSourceStatus.REGISTERED and source.last_sync_at is None


def test_source_origin_validator_accepts_and_rejects() -> None:
    attach, lifecycle, repo, _, _ = _build_stack()
    rev = _attach(attach)
    lifecycle.activate_indexed_source(_activate_cmd(expected_revision=rev))
    binding = repo.list_knowledge_indexed_source_versions(
        tenant_id=_TENANT, workspace_id=_WORKSPACE,
    )[0]
    validate_connected_source_durable_origin(
        repository=repo,
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        source_id=_SOURCE_ID,
        binding=binding,
        committed_configuration_revision=rev + 1,
    )
    lifecycle.disable_indexed_source(_disable_cmd(expected_revision=rev + 1))
    reactivated = lifecycle.activate_indexed_source(
        _activate_cmd(expected_revision=rev + 2, idempotency_key_hash=_SHA256_E)
    )
    validate_connected_source_durable_origin(
        repository=repo,
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        source_id=_SOURCE_ID,
        binding=reactivated.binding,
        committed_configuration_revision=rev + 3,
    )
    with pytest.raises(ConnectedSourceOriginValidationError, match="source_id_mismatch"):
        validate_connected_source_durable_origin(
            repository=repo,
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            source_id="src:connected:wrong",
            binding=reactivated.binding,
            committed_configuration_revision=rev + 3,
        )
    with pytest.raises(ConnectedSourceOriginValidationError, match="binding_revision_after_committed"):
        validate_connected_source_durable_origin(
            repository=repo,
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            source_id=_SOURCE_ID,
            binding=reactivated.binding,
            committed_configuration_revision=0,
        )


def test_indexed_source_id_from_semantic_hash_matches_canonical() -> None:
    semantic = workspace_indexed_source_semantic_hash(_TENANT, _WORKSPACE, _KSB_REF)
    assert indexed_source_binding_id(_TENANT, _WORKSPACE, _KSB_REF) == (
        f"idx:{semantic[:32]}"
    )
    assert connected_source_id(_TENANT, _WORKSPACE, _KSB_REF) == (
        f"src:connected:{semantic[:32]}"
    )
    assert connected_source_id_from_semantic_hash(semantic) == connected_source_id(
        _TENANT, _WORKSPACE, _KSB_REF
    )


def test_indexed_source_not_found_on_disable() -> None:
    attach, lifecycle, _, _, _ = _build_stack()
    rev = _attach(attach)
    with pytest.raises(WorkspaceIndexedSourceLifecycleError, match="indexed_source_not_found"):
        lifecycle.disable_indexed_source(
            _disable_cmd(
                indexed_source_binding_id="idx:missing",
                expected_revision=rev,
            )
        )
