# © Artur Czarnecki. All rights reserved.

"""Tests for WorkspaceLiveAccessBindingService and live access hashing."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.runtime.vendor_knowledge.remote_resource_discovery import (
    RemoteResourceAvailabilityV1,
    RemoteResourceDescriptorV1,
)
from intergrax.runtime.vendor_knowledge.tenant_connection_capabilities import (
    CapabilityEffectV1,
    LiveCapabilityDescriptorV1,
)
from intergrax.runtime.vendor_knowledge.tenant_connections import (
    SafeTenantConnectionV1,
    TenantConnectionAdministrativeStatus,
)
from local_workspace_application.workspaces.knowledge_configuration_handlers import (
    AttachConnectionMutationHandler,
)
from local_workspace_application.workspaces.knowledge_configuration_hashing import (
    live_access_binding_id_from_semantic_hash,
    normalize_create_live_access_binding_request_hash,
    normalize_disable_live_access_binding_request_hash,
    normalize_live_access_capability_set,
    semantic_identity_hash_for_live_access_binding,
)
from local_workspace_application.workspaces.knowledge_configuration_models import (
    KnowledgeAudienceEligibilityV1,
    LiveAccessBindingStatusV1,
    WorkspaceKnowledgeMutationOperationV1,
    WorkspaceKnowledgeMutationRecord,
    WorkspaceKnowledgeMutationStatusV1,
    WorkspaceLiveAccessBinding,
)
from local_workspace_application.workspaces.knowledge_configuration_mutation_engine import (
    WorkspaceKnowledgeConfigurationMutationEngine,
    WorkspaceKnowledgeConfigurationMutationError,
    WorkspaceKnowledgeMutationExecutionDispositionV1,
    WorkspaceKnowledgeMutationRecoveryDispositionV1,
    WorkspaceKnowledgeStageStateV1,
)
from local_workspace_application.workspaces.knowledge_configuration_service import (
    WorkspaceKnowledgeConfigurationService,
)
from local_workspace_application.workspaces.knowledge_connection_attachment_service import (
    AttachWorkspaceConnectionCommand,
    WorkspaceConnectionAttachmentService,
)
from local_workspace_application.workspaces.knowledge_live_access_handlers import (
    CreateLiveAccessBindingMutationHandler,
    CreateLiveAccessBindingMutationIntent,
    DisableLiveAccessBindingMutationHandler,
)
from local_workspace_application.workspaces.knowledge_live_access_service import (
    CreateWorkspaceLiveAccessBindingCommand,
    DisableWorkspaceLiveAccessBindingCommand,
    WorkspaceLiveAccessBindingError,
    WorkspaceLiveAccessBindingService,
)
from local_workspace_application.workspaces.models import Workspace, WorkspaceStatus
from local_workspace_application.workspaces.repository import ManagedWorkspaceRepository
from local_workspace_application.workspaces.service import ManagedWorkspaceService

pytestmark = pytest.mark.unit

_NOW = datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC)
_TENANT, _WORKSPACE = "tenant-a", "workspace-1"
_CONNECTION = "conn.slack"
_RESOURCE = "resource-1"
_CAP_READ = "cap.read"
_CAP_RESOURCE = "cap.resource"
_PROVIDER = "provider.slack"
_SHA256_A, _SHA256_B, _SHA256_C, _SHA256_D, _SHA256_E = (
    "a" * 64,
    "b" * 64,
    "c" * 64,
    "d" * 64,
    "e" * 64,
)
_AUDIENCE = KnowledgeAudienceEligibilityV1.PERSONAL_ONLY
_SEMANTIC = semantic_identity_hash_for_live_access_binding(
    tenant_id=_TENANT,
    workspace_id=_WORKSPACE,
    connection_ref=_CONNECTION,
    normalized_remote_resource_id=None,
    normalized_capability_set=(_CAP_READ,),
)
_BINDING_ID = live_access_binding_id_from_semantic_hash(_SEMANTIC)

_CREATE_HANDLER = CreateLiveAccessBindingMutationHandler()
_DISABLE_HANDLER = DisableLiveAccessBindingMutationHandler()


def _workspace() -> Workspace:
    return Workspace(
        workspace_id=_WORKSPACE,
        tenant_id=_TENANT,
        name="Workspace",
        status=WorkspaceStatus.ACTIVE,
        created_at=_NOW,
        updated_at=_NOW,
    )


def _safe_connection(**overrides: object) -> SafeTenantConnectionV1:
    payload = {
        "connection_ref": _CONNECTION,
        "tenant_id": _TENANT,
        "provider_id": _PROVIDER,
        "integration_kind": IntegrationCategory.CONVERSATION_CHANNEL,
        "safe_display_name": "Slack",
        "administrative_status": TenantConnectionAdministrativeStatus.ACTIVE,
        "configuration_version": 1,
        "connected_principal_ref": None,
        "created_at": _NOW,
        "updated_at": _NOW,
    }
    payload.update(overrides)
    return SafeTenantConnectionV1(**payload)


def _descriptor(**overrides: object) -> LiveCapabilityDescriptorV1:
    payload = {
        "capability_id": _CAP_READ,
        "provider_id": _PROVIDER,
        "integration_kind": IntegrationCategory.CONVERSATION_CHANNEL,
        "effect": CapabilityEffectV1.READ,
        "read_only": True,
        "resource_scope_required": False,
        "request_schema_ref": "schema://req",
        "result_schema_ref": "schema://res",
        "available": True,
    }
    payload.update(overrides)
    return LiveCapabilityDescriptorV1(**payload)


def _resource_descriptor(**overrides: object) -> RemoteResourceDescriptorV1:
    payload = {
        "remote_resource_id": _RESOURCE,
        "resource_type": "slack_conversation",
        "safe_display_label": "General",
        "availability": RemoteResourceAvailabilityV1.AVAILABLE,
        "supported_capability_ids": (_CAP_RESOURCE,),
        "connection_ref": _CONNECTION,
        "provider_id": _PROVIDER,
        "integration_kind": IntegrationCategory.CONVERSATION_CHANNEL,
        "source_kind": "slack_conversation",
        "discovered_at": _NOW,
        "snapshot_version": "snap-1",
    }
    payload.update(overrides)
    return RemoteResourceDescriptorV1(**payload)


class _FakeConnectionPort:
    def get_connection(self, *, tenant_id: str, connection_ref: str) -> SafeTenantConnectionV1 | None:
        if tenant_id == _TENANT and connection_ref == _CONNECTION:
            return _safe_connection()
        return None

    def list_connections(self, *, tenant_id: str, limit: int = 100, administrative_status=None):
        conn = self.get_connection(tenant_id=tenant_id, connection_ref=_CONNECTION)
        return (conn,) if conn is not None else ()


class _FakeCatalog:
    def __init__(self, descriptors: tuple[LiveCapabilityDescriptorV1, ...]) -> None:
        self._descriptors = descriptors
        self.call_count = 0

    def list_capabilities(self, *, tenant_id: str, connection_ref: str, remote_resource_id: str | None):
        self.call_count += 1
        return self._descriptors


class _FakeResourceLookup:
    def __init__(self, resource: RemoteResourceDescriptorV1 | None) -> None:
        self._resource = resource
        self.call_count = 0

    async def get_remote_resource(self, *, tenant_id: str, connection_ref: str, remote_resource_id: str):
        self.call_count += 1
        if self._resource is None or remote_resource_id != self._resource.remote_resource_id:
            return None
        return self._resource


def _build_stack(
    *,
    catalog: _FakeCatalog | None = None,
    lookup: _FakeResourceLookup | None = None,
    mutation_ids: list[str] | None = None,
):
    store = InMemoryDocumentStore()
    repo = ManagedWorkspaceRepository(store)
    repo.put_workspace(_workspace())
    lookup_service = ManagedWorkspaceService(repo)
    config = WorkspaceKnowledgeConfigurationService(repo, lookup_service)
    ids = mutation_ids or [f"mutation-{i}" for i in range(1, 12)]
    idx = {"i": 0}

    def _next_id() -> str:
        value = ids[idx["i"]]
        idx["i"] = min(idx["i"] + 1, len(ids) - 1)
        return value

    handlers = {
        WorkspaceKnowledgeMutationOperationV1.ATTACH_CONNECTION: AttachConnectionMutationHandler(),
        WorkspaceKnowledgeMutationOperationV1.CREATE_LIVE_ACCESS_BINDING: CreateLiveAccessBindingMutationHandler(),
        WorkspaceKnowledgeMutationOperationV1.DISABLE_LIVE_ACCESS_BINDING: DisableLiveAccessBindingMutationHandler(),
    }
    engine = WorkspaceKnowledgeConfigurationMutationEngine(
        repo,
        lookup_service,
        config,
        handlers,
        clock=lambda: _NOW,
        mutation_id_factory=_next_id,
    )
    attach = WorkspaceConnectionAttachmentService(
        connection_port=_FakeConnectionPort(),
        configuration_service=config,
        mutation_engine=engine,
    )
    catalog_impl = catalog or _FakeCatalog((_descriptor(),))
    lookup_impl = lookup
    service = WorkspaceLiveAccessBindingService(
        repository=repo,
        configuration_service=config,
        mutation_engine=engine,
        tenant_connection_port=_FakeConnectionPort(),
        capability_catalog=catalog_impl,
        remote_resource_lookup_port=lookup_impl,
    )
    return attach, service, repo, catalog_impl, lookup_impl, engine


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


def _create_cmd(**overrides: object) -> CreateWorkspaceLiveAccessBindingCommand:
    payload = {
        "tenant_id": _TENANT,
        "workspace_id": _WORKSPACE,
        "connection_ref": _CONNECTION,
        "remote_resource_id": None,
        "allowed_capability_ids": (_CAP_READ,),
        "expected_revision": 1,
        "idempotency_key_hash": _SHA256_A,
        "audience_eligibility": _AUDIENCE,
    }
    payload.update(overrides)
    return CreateWorkspaceLiveAccessBindingCommand(**payload)


def _disable_cmd(**overrides: object) -> DisableWorkspaceLiveAccessBindingCommand:
    payload = {
        "tenant_id": _TENANT,
        "workspace_id": _WORKSPACE,
        "live_access_binding_id": _BINDING_ID,
        "expected_revision": 2,
        "idempotency_key_hash": _SHA256_C,
    }
    payload.update(overrides)
    return DisableWorkspaceLiveAccessBindingCommand(**payload)


@pytest.mark.asyncio
async def test_capability_order_does_not_affect_semantic_identity() -> None:
    first = semantic_identity_hash_for_live_access_binding(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        connection_ref=_CONNECTION,
        normalized_remote_resource_id=None,
        normalized_capability_set=normalize_live_access_capability_set(("cap.b", "cap.a")),
    )
    second = semantic_identity_hash_for_live_access_binding(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        connection_ref=_CONNECTION,
        normalized_remote_resource_id=None,
        normalized_capability_set=normalize_live_access_capability_set(("cap.a", "cap.b")),
    )
    assert first == second


def test_duplicate_capability_ids_removed_and_blank_rejected() -> None:
    assert normalize_live_access_capability_set((" cap.a ", "cap.a", "cap.b")) == ("cap.a", "cap.b")
    with pytest.raises(ValueError, match="blank_capability_id"):
        normalize_live_access_capability_set((" ",))


def test_trimmed_requests_share_binding_id() -> None:
    semantic = semantic_identity_hash_for_live_access_binding(
        tenant_id=f" {_TENANT} ",
        workspace_id=f" {_WORKSPACE} ",
        connection_ref=f" {_CONNECTION} ",
        normalized_remote_resource_id=None,
        normalized_capability_set=(_CAP_READ,),
    )
    assert live_access_binding_id_from_semantic_hash(semantic) == _BINDING_ID


def test_different_capability_or_resource_sets_change_id() -> None:
    base = semantic_identity_hash_for_live_access_binding(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        connection_ref=_CONNECTION,
        normalized_remote_resource_id=None,
        normalized_capability_set=(_CAP_READ,),
    )
    other_cap = semantic_identity_hash_for_live_access_binding(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        connection_ref=_CONNECTION,
        normalized_remote_resource_id=None,
        normalized_capability_set=("cap.other",),
    )
    other_resource = semantic_identity_hash_for_live_access_binding(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        connection_ref=_CONNECTION,
        normalized_remote_resource_id=_RESOURCE,
        normalized_capability_set=(_CAP_READ,),
    )
    assert base != other_cap != other_resource


def test_idempotency_key_not_in_binding_id() -> None:
    first = normalize_create_live_access_binding_request_hash(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        connection_ref=_CONNECTION,
        remote_resource_id=None,
        allowed_capability_ids=(_CAP_READ,),
        audience_eligibility=_AUDIENCE,
    )
    second = normalize_create_live_access_binding_request_hash(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        connection_ref=_CONNECTION,
        remote_resource_id=None,
        allowed_capability_ids=(_CAP_READ,),
        audience_eligibility=KnowledgeAudienceEligibilityV1.SHARED_ALLOWED,
    )
    assert first != second
    assert live_access_binding_id_from_semantic_hash(_SEMANTIC) == _BINDING_ID


@pytest.mark.asyncio
async def test_initial_create_applied() -> None:
    attach, service, repo, _, _, _ = _build_stack()
    rev = _attach(attach)
    result = await service.create_live_access_binding(_create_cmd(expected_revision=rev))
    assert result.disposition is WorkspaceKnowledgeMutationExecutionDispositionV1.APPLIED
    assert result.created_new_binding is True
    assert result.binding.status is LiveAccessBindingStatusV1.ACTIVE
    assert result.binding.live_access_binding_id == _BINDING_ID
    assert repo.list_sources(tenant_id=_TENANT, workspace_id=_WORKSPACE) == []
    assert repo.list_knowledge_indexed_source_versions(tenant_id=_TENANT, workspace_id=_WORKSPACE) == []


@pytest.mark.asyncio
async def test_semantic_noop_and_replay() -> None:
    attach, service, _, catalog, _, _ = _build_stack()
    rev = _attach(attach)
    await service.create_live_access_binding(_create_cmd(expected_revision=rev))
    noop = await service.create_live_access_binding(
        _create_cmd(expected_revision=rev + 1, idempotency_key_hash=_SHA256_B)
    )
    assert noop.disposition is WorkspaceKnowledgeMutationExecutionDispositionV1.EXISTING_RESULT
    calls_before_replay = catalog.call_count
    replay = await service.create_live_access_binding(_create_cmd(expected_revision=rev + 1))
    assert replay.disposition is WorkspaceKnowledgeMutationExecutionDispositionV1.COMMITTED_REPLAY
    assert catalog.call_count == calls_before_replay


@pytest.mark.asyncio
async def test_disable_and_noop() -> None:
    attach, service, _, _, _, _ = _build_stack()
    rev = _attach(attach)
    await service.create_live_access_binding(_create_cmd(expected_revision=rev))
    disabled = service.disable_live_access_binding(_disable_cmd(expected_revision=rev + 1))
    assert disabled.disposition is WorkspaceKnowledgeMutationExecutionDispositionV1.APPLIED
    assert disabled.binding.status is LiveAccessBindingStatusV1.DISABLED
    noop = service.disable_live_access_binding(
        _disable_cmd(expected_revision=disabled.configuration_revision, idempotency_key_hash=_SHA256_D)
    )
    assert noop.disposition is WorkspaceKnowledgeMutationExecutionDispositionV1.EXISTING_RESULT


@pytest.mark.asyncio
async def test_reactivation_same_id() -> None:
    attach, service, _, _, _, _ = _build_stack()
    rev = _attach(attach)
    await service.create_live_access_binding(_create_cmd(expected_revision=rev))
    disabled = service.disable_live_access_binding(_disable_cmd(expected_revision=rev + 1))
    reactivated = await service.create_live_access_binding(
        _create_cmd(expected_revision=disabled.configuration_revision, idempotency_key_hash=_SHA256_E)
    )
    assert reactivated.disposition is WorkspaceKnowledgeMutationExecutionDispositionV1.APPLIED
    assert reactivated.binding.live_access_binding_id == _BINDING_ID
    assert reactivated.binding.status is LiveAccessBindingStatusV1.ACTIVE


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("overrides", "error", "capability_id"),
    [
        ({"effect": CapabilityEffectV1.WRITE}, "capability_not_read_only", _CAP_READ),
        ({"effect": CapabilityEffectV1.EXECUTE}, "capability_not_read_only", _CAP_READ),
        ({"effect": CapabilityEffectV1.ADMIN}, "capability_not_read_only", _CAP_READ),
        ({"read_only": False}, "capability_not_read_only", _CAP_READ),
        ({"available": False}, "capability_not_read_only", _CAP_READ),
        ({"capability_id": "cap.read.write"}, "capability_not_read_only", "cap.read.write"),
    ],
)
async def test_capability_fence(overrides: dict, error: str, capability_id: str) -> None:
    attach, service, _, _, _, _ = _build_stack(catalog=_FakeCatalog((_descriptor(**overrides),)))
    rev = _attach(attach)
    with pytest.raises(WorkspaceLiveAccessBindingError, match=error):
        await service.create_live_access_binding(
            _create_cmd(expected_revision=rev, allowed_capability_ids=(capability_id,))
        )


@pytest.mark.asyncio
async def test_unknown_capability_rejected() -> None:
    attach, service, _, _, _, _ = _build_stack()
    rev = _attach(attach)
    with pytest.raises(WorkspaceLiveAccessBindingError, match="capability_not_found"):
        await service.create_live_access_binding(
            _create_cmd(expected_revision=rev, allowed_capability_ids=("missing",))
        )


@pytest.mark.asyncio
async def test_catalog_provider_mismatch_rejected() -> None:
    attach, service, _, _, _, _ = _build_stack(
        catalog=_FakeCatalog((_descriptor(provider_id="other"),)),
    )
    rev = _attach(attach)
    with pytest.raises(WorkspaceLiveAccessBindingError, match="capability_catalog_invalid"):
        await service.create_live_access_binding(_create_cmd(expected_revision=rev))


@pytest.mark.asyncio
async def test_resource_required_without_id() -> None:
    catalog = _FakeCatalog((_descriptor(capability_id=_CAP_RESOURCE, resource_scope_required=True),))
    attach, service, _, _, _, _ = _build_stack(catalog=catalog)
    rev = _attach(attach)
    with pytest.raises(WorkspaceLiveAccessBindingError, match="remote_resource_required"):
        await service.create_live_access_binding(
            _create_cmd(expected_revision=rev, allowed_capability_ids=(_CAP_RESOURCE,))
        )


@pytest.mark.asyncio
async def test_valid_resource_binding() -> None:
    catalog = _FakeCatalog(
        (
            _descriptor(
                capability_id=_CAP_RESOURCE,
                resource_scope_required=True,
                supported_resource_types=("slack_conversation",),
            ),
        )
    )
    lookup = _FakeResourceLookup(_resource_descriptor())
    attach, service, _, _, _, _ = _build_stack(catalog=catalog, lookup=lookup)
    rev = _attach(attach)
    result = await service.create_live_access_binding(
        _create_cmd(
            expected_revision=rev,
            remote_resource_id=_RESOURCE,
            allowed_capability_ids=(_CAP_RESOURCE,),
        )
    )
    assert result.binding.derived_resource_type == "slack_conversation"
    assert result.binding.derived_safe_display_label == "General"


@pytest.mark.asyncio
async def test_resource_lookup_unavailable() -> None:
    catalog = _FakeCatalog((_descriptor(capability_id=_CAP_RESOURCE, resource_scope_required=True),))
    attach, service, _, _, _, _ = _build_stack(catalog=catalog, lookup=None)
    rev = _attach(attach)
    with pytest.raises(WorkspaceLiveAccessBindingError, match="remote_resource_lookup_unavailable"):
        await service.create_live_access_binding(
            _create_cmd(
                expected_revision=rev,
                remote_resource_id=_RESOURCE,
                allowed_capability_ids=(_CAP_RESOURCE,),
            )
        )


@pytest.mark.asyncio
async def test_disable_replay_skips_ports() -> None:
    attach, service, repo, catalog, lookup, _ = _build_stack()
    rev = _attach(attach)
    await service.create_live_access_binding(_create_cmd(expected_revision=rev))
    service.disable_live_access_binding(_disable_cmd(expected_revision=rev + 1))
    await service.create_live_access_binding(
        _create_cmd(expected_revision=rev + 2, idempotency_key_hash=_SHA256_E)
    )
    catalog.call_count = 0
    if lookup is not None:
        lookup.call_count = 0
    replay = service.disable_live_access_binding(_disable_cmd(expected_revision=rev + 1))
    assert replay.disposition is WorkspaceKnowledgeMutationExecutionDispositionV1.COMMITTED_REPLAY
    assert catalog.call_count == 0


def _live_row(*, mutation_id: str, revision: int, status: LiveAccessBindingStatusV1) -> WorkspaceLiveAccessBinding:
    return WorkspaceLiveAccessBinding(
        live_access_binding_id=_BINDING_ID,
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        connection_ref=_CONNECTION,
        remote_resource_id=None,
        allowed_capability_ids=(_CAP_READ,),
        derived_provider_id=_PROVIDER,
        derived_integration_kind=IntegrationCategory.CONVERSATION_CHANNEL,
        derived_resource_type=None,
        derived_safe_display_label="Slack",
        status=status,
        audience_eligibility=_AUDIENCE,
        mutation_id=mutation_id,
        effective_revision=revision,
        semantic_identity_hash=_SEMANTIC,
        created_at=_NOW,
        updated_at=_NOW,
    )


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


def _create_mutation(repo, *, revision: int, mutation_id: str = "mutation-create") -> WorkspaceKnowledgeMutationRecord:
    request_hash = normalize_create_live_access_binding_request_hash(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        connection_ref=_CONNECTION,
        remote_resource_id=None,
        allowed_capability_ids=(_CAP_READ,),
        audience_eligibility=_AUDIENCE,
    )
    mutation = WorkspaceKnowledgeMutationRecord(
        mutation_id=mutation_id,
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        operation=WorkspaceKnowledgeMutationOperationV1.CREATE_LIVE_ACCESS_BINDING,
        idempotency_key_hash=_SHA256_A,
        normalized_request_hash=request_hash,
        semantic_identity_hash=_SEMANTIC,
        target_revision=revision,
        status=WorkspaceKnowledgeMutationStatusV1.PREPARED,
        result_entity_type="live_access_binding",
        result_entity_id=_BINDING_ID,
        created_at=_NOW,
        updated_at=_NOW,
    )
    repo.put_knowledge_configuration_mutation_if_absent(mutation)
    _pending_head(repo, revision=revision, mutation_id=mutation_id)
    return mutation


@pytest.mark.asyncio
async def test_complete_prepared_create_recovery_commits() -> None:
    attach, _, repo, _, _, engine = _build_stack(mutation_ids=["mutation-attach", "mutation-create"])
    rev = _attach(attach)
    mutation = _create_mutation(repo, revision=rev + 1)
    intent = CreateLiveAccessBindingMutationIntent(
        connection_ref=_CONNECTION,
        remote_resource_id=None,
        allowed_capability_ids=(_CAP_READ,),
        audience_eligibility=_AUDIENCE,
        derived_provider_id=_PROVIDER,
        derived_integration_kind=IntegrationCategory.CONVERSATION_CHANNEL,
        derived_resource_type=None,
        derived_safe_display_label="Slack",
    )
    _CREATE_HANDLER.stage(
        repository=repo,
        mutation=mutation,
        target_revision=mutation.target_revision,
        intent=intent,
        now=_NOW,
    )
    recovery = engine.recover_workspace_knowledge_mutation(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert recovery.disposition is WorkspaceKnowledgeMutationRecoveryDispositionV1.COMMITTED
    head = repo.get_knowledge_configuration_head(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert head is not None and head.committed_revision == rev + 1


def test_duplicate_owned_rows_conflict() -> None:
    attach, _, repo, _, _, engine = _build_stack(mutation_ids=["mutation-attach", "mutation-create"])
    rev = _attach(attach)
    mutation = _create_mutation(repo, revision=rev + 1)
    repo.put_knowledge_live_access_version_if_absent(
        _live_row(mutation_id=mutation.mutation_id, revision=rev + 1, status=LiveAccessBindingStatusV1.ACTIVE)
    )
    repo.put_knowledge_live_access_version_if_absent(
        _live_row(mutation_id=mutation.mutation_id, revision=rev + 2, status=LiveAccessBindingStatusV1.ACTIVE)
    )
    assert _CREATE_HANDLER.inspect_staged(repository=repo, mutation=mutation).state is (
        WorkspaceKnowledgeStageStateV1.OWNERSHIP_CONFLICT
    )
    with pytest.raises(WorkspaceKnowledgeConfigurationMutationError, match="configuration_recovery_required"):
        engine.recover_workspace_knowledge_mutation(tenant_id=_TENANT, workspace_id=_WORKSPACE)


def test_cleanup_deletes_exact_owned_row() -> None:
    attach, _, repo, _, _, engine = _build_stack(mutation_ids=["mutation-attach", "mutation-create"])
    rev = _attach(attach)
    mutation = _create_mutation(repo, revision=rev + 1)
    staged = _live_row(mutation_id=mutation.mutation_id, revision=rev + 1, status=LiveAccessBindingStatusV1.ACTIVE)
    repo.put_knowledge_live_access_version_if_absent(staged)
    inspection = _CREATE_HANDLER.inspect_staged(repository=repo, mutation=mutation)
    assert inspection.state is WorkspaceKnowledgeStageStateV1.COMPLETE_VALID
    assert _CREATE_HANDLER.cleanup_staged(repository=repo, mutation=mutation, inspection=inspection) is True
    assert repo.list_knowledge_live_access_versions(tenant_id=_TENANT, workspace_id=_WORKSPACE) == []
    recovery = engine.recover_workspace_knowledge_mutation(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert recovery.disposition is WorkspaceKnowledgeMutationRecoveryDispositionV1.ABORTED
