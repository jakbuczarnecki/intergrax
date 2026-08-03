# © Artur Czarnecki. All rights reserved.

"""Bounded acceptance proof: durable connection indexed-live reuse without duplication."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.runtime.vendor_knowledge.binding_document_store import (
    DocumentStoreKnowledgeSourceBindingRepository,
)
from intergrax.runtime.vendor_knowledge.bindings import (
    KnowledgeSourceBinding,
    KnowledgeSourceBindingService,
    KnowledgeSourceBindingStatus,
    KnowledgeSourceScope,
)
from intergrax.runtime.vendor_knowledge.connections import (
    ConnectionAwareVendorResolver,
    KnowledgeConnectionRegistry,
)
from intergrax.runtime.vendor_knowledge.errors import (
    VendorKnowledgeError,
    VendorKnowledgeErrorCode,
)
from intergrax.runtime.vendor_knowledge.facade import VendorKnowledgeFacadeService
from intergrax.runtime.vendor_knowledge.models import KnowledgeSourceRef, KnowledgeSourceScope as FacadeScope
from intergrax.runtime.vendor_knowledge.registry import KnowledgeAdapterRegistry
from intergrax.runtime.vendor_knowledge.tenant_connection_capabilities import (
    CapabilityEffectV1,
    LiveCapabilityDescriptorV1,
    RepositoryTenantConnectionPort,
    TenantLiveCapabilityCatalog,
)
from intergrax.runtime.vendor_knowledge.tenant_connection_document_store import (
    DocumentStoreTenantConnectionRepository,
)
from intergrax.runtime.vendor_knowledge.tenant_connection_rehydration import (
    TenantConnectionRehydrationStatus,
    TenantConnectionRehydrator,
)
from intergrax.runtime.vendor_knowledge.tenant_connections import (
    SafeTenantConnectionV1,
    TenantConnection,
    TenantConnectionAdministrativeStatus,
    TenantConnectionService,
)
from local_workspace_application.serving.knowledge_live_access_routes import (
    mount_knowledge_live_access_routes,
)
from local_workspace_application.workspaces.connected_source_ids import (
    connected_source_id,
    indexed_source_binding_id,
)
from local_workspace_application.workspaces.knowledge_configuration_handlers import (
    AttachConnectionMutationHandler,
    CreateIndexedSourceMutationHandler,
    DisableIndexedSourceMutationHandler,
)
from local_workspace_application.workspaces.knowledge_configuration_hashing import (
    live_access_binding_id_from_semantic_hash,
    semantic_identity_hash_for_live_access_binding,
)
from local_workspace_application.workspaces.knowledge_configuration_models import (
    IndexedSourceAudienceEligibilityV1,
    IndexedSourceSyncModeV1,
    LiveAccessBindingStatusV1,
    WorkspaceConnectionAttachmentStatusV1,
    WorkspaceIndexedSourceBindingStatusV1,
    WorkspaceKnowledgeMutationOperationV1,
    WorkspaceLiveAccessBinding,
)
from local_workspace_application.workspaces.knowledge_configuration_mutation_engine import (
    WorkspaceKnowledgeConfigurationMutationEngine,
)
from local_workspace_application.workspaces.knowledge_configuration_service import (
    WorkspaceKnowledgeConfigurationService,
)
from local_workspace_application.workspaces.knowledge_connection_attachment_service import (
    AttachWorkspaceConnectionCommand,
    WorkspaceConnectionAttachmentService,
)
from local_workspace_application.workspaces.knowledge_indexed_source_lifecycle_service import (
    ActivateWorkspaceIndexedSourceCommand,
    DisableWorkspaceIndexedSourceCommand,
    WorkspaceIndexedSourceLifecycleService,
)
from local_workspace_application.workspaces.knowledge_live_access_handlers import (
    CreateLiveAccessBindingMutationHandler,
    DisableLiveAccessBindingMutationHandler,
)
from local_workspace_application.workspaces.knowledge_live_access_service import (
    CreateWorkspaceLiveAccessBindingCommand,
    DisableWorkspaceLiveAccessBindingCommand,
    WorkspaceLiveAccessBindingService,
)
from local_workspace_application.workspaces.models import Workspace, WorkspaceSourceType, WorkspaceStatus
from local_workspace_application.workspaces.repository import ManagedWorkspaceRepository
from local_workspace_application.workspaces.service import ManagedWorkspaceService
from tests.unit.runtime.vendor_knowledge._fakes import FakeAdapter, FakeIntegration

pytestmark = pytest.mark.integration

_TENANT = "tenant-proof"
_WORKSPACE = "workspace-proof"
_WORKSPACE_OTHER = "workspace-other"
_CONNECTION = "conn-proof-1"
_BINDING_REF = "bind-proof-1"
_CREDENTIAL_REF = "secrets/tenant-proof/conn-proof-1"
_SECRET = "proof-runtime-secret"
_PROVIDER = "example"
_CAP_READ = "proof.read"
_NOW = datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC)
_SHA256_A = "a" * 64
_SHA256_B = "b" * 64
_SHA256_C = "c" * 64
_SHA256_D = "d" * 64
_SYNC = IndexedSourceSyncModeV1.FULL
_AUDIENCE = IndexedSourceAudienceEligibilityV1.PERSONAL_ONLY
_INDEXED_BINDING_ID = indexed_source_binding_id(_TENANT, _WORKSPACE, _BINDING_REF)
_SOURCE_ID = connected_source_id(_TENANT, _WORKSPACE, _BINDING_REF)
_LIVE_SEMANTIC = semantic_identity_hash_for_live_access_binding(
    tenant_id=_TENANT,
    workspace_id=_WORKSPACE,
    connection_ref=_CONNECTION,
    normalized_remote_resource_id=None,
    normalized_capability_set=(_CAP_READ,),
)
_LIVE_BINDING_ID = live_access_binding_id_from_semantic_hash(_LIVE_SEMANTIC)
_FORBIDDEN_LEAK_KEYS = frozenset(
    {
        "credential_ref",
        "validated_secret_free_config",
        "access_token",
        "refresh_token",
        "client_secret",
        "password",
        "api_key",
        "authorization",
    }
)


class _RecordingSecretsStore:
    def __init__(self, *, secret: str | None = _SECRET) -> None:
        self.secret = secret
        self.calls: list[str] = []

    def get_secret(self, path: str, *, version: str | None = None) -> str:
        self.calls.append(path)
        if self.secret is None:
            raise KeyError("missing")
        return self.secret

    def put_secret(self, path: str, value: str) -> None:
        return None

    def delete_secret(self, path: str) -> None:
        return None


class _CountingFactory:
    def __init__(self, *, integration: object | None = None) -> None:
        self.calls: list[dict[str, object]] = []
        self._integration = integration

    def create_integration(
        self,
        *,
        tenant_id: str,
        connection_ref: str,
        provider_id: str,
        integration_kind: IntegrationCategory,
        credential_ref: str,
        credential: str,
        secret_free_config,
    ) -> object:
        self.calls.append(
            {
                "tenant_id": tenant_id,
                "connection_ref": connection_ref,
                "provider_id": provider_id,
                "integration_kind": integration_kind,
                "credential_ref": credential_ref,
                "credential": credential,
                "secret_free_config": secret_free_config,
            }
        )
        if self._integration is not None:
            return self._integration
        return FakeIntegration(
            provider_id=provider_id,
            integration_kind=integration_kind.value,
        )


class _FailingFallbackResolver:
    def __init__(self) -> None:
        self.calls: list[KnowledgeSourceRef] = []

    def resolve(self, *, source: KnowledgeSourceRef) -> object:
        self.calls.append(source)
        raise AssertionError("fallback resolver must not be invoked when connection_ref is set")


class _RepositoryTenantBindingPort:
    def __init__(self, repository: DocumentStoreKnowledgeSourceBindingRepository) -> None:
        self._repository = repository

    def get_binding(self, *, tenant_id: str, binding_id: str) -> KnowledgeSourceBinding | None:
        binding = self._repository.get(tenant_id=tenant_id.strip(), binding_id=binding_id.strip())
        if binding is None or binding.tenant_id != tenant_id.strip():
            return None
        return binding


class _LiveResolverProbe:
    def __init__(self, resolver: ConnectionAwareVendorResolver) -> None:
        self._resolver = resolver

    def resolve_integration(self, *, binding: WorkspaceLiveAccessBinding) -> object:
        source = KnowledgeSourceRef(
            tenant_id=binding.tenant_id,
            provider_id=binding.derived_provider_id,
            integration_kind=binding.derived_integration_kind,
            source_kind="live_proof",
            connection_ref=binding.connection_ref,
            scope=FacadeScope(
                remote_scope_id="live-proof-scope",
                remote_scope_type="live_proof",
                safe_display_name="Live proof scope",
                parameters={},
            ),
        )
        return self._resolver.resolve(source=source)


@dataclass
class ProofContext:
    store: InMemoryDocumentStore
    connection_repo: DocumentStoreTenantConnectionRepository
    binding_repo: DocumentStoreKnowledgeSourceBindingRepository
    workspace_repo: ManagedWorkspaceRepository
    registry: KnowledgeConnectionRegistry
    secrets: _RecordingSecretsStore
    factory: _CountingFactory
    fallback: _FailingFallbackResolver
    connection_resolver: ConnectionAwareVendorResolver
    binding_service: KnowledgeSourceBindingService
    facade: VendorKnowledgeFacadeService
    adapter: FakeAdapter
    config_service: WorkspaceKnowledgeConfigurationService
    attach_service: WorkspaceConnectionAttachmentService
    indexed_lifecycle: WorkspaceIndexedSourceLifecycleService
    live_service: WorkspaceLiveAccessBindingService
    capability_catalog: TenantLiveCapabilityCatalog
    reconstructed_integration: object
    rehydration_results: tuple[Any, ...] = field(default_factory=tuple)


def _tenant_connection(**overrides: object) -> TenantConnection:
    payload = {
        "connection_ref": _CONNECTION,
        "tenant_id": _TENANT,
        "provider_id": _PROVIDER,
        "integration_kind": IntegrationCategory.ISSUE_TRACKER,
        "safe_display_name": "Proof connection",
        "administrative_status": TenantConnectionAdministrativeStatus.ACTIVE,
        "credential_ref": _CREDENTIAL_REF,
        "validated_secret_free_config": {"base_url": "https://provider.example.test"},
        "configuration_version": 1,
        "created_at": _NOW,
        "updated_at": _NOW,
        "connected_principal_ref": None,
    }
    payload.update(overrides)
    return TenantConnection(**payload)


def _tenant_binding(**overrides: object) -> KnowledgeSourceBinding:
    payload = {
        "binding_id": _BINDING_REF,
        "tenant_id": _TENANT,
        "provider_id": _PROVIDER,
        "integration_kind": IntegrationCategory.ISSUE_TRACKER,
        "source_kind": "issues",
        "connection_ref": _CONNECTION,
        "credential_ref": None,
        "safe_display_name": "Proof project",
        "scope": KnowledgeSourceScope(
            remote_scope_id="project-proof",
            remote_scope_type="project",
            safe_display_name="Proof project",
            parameters={},
        ),
        "status": KnowledgeSourceBindingStatus.ACTIVE,
        "configuration_version": 1,
    }
    payload.update(overrides)
    return KnowledgeSourceBinding(**payload)


def _workspace(workspace_id: str) -> Workspace:
    return Workspace(
        workspace_id=workspace_id,
        tenant_id=_TENANT,
        name=f"Workspace {workspace_id}",
        status=WorkspaceStatus.ACTIVE,
        created_at=_NOW,
        updated_at=_NOW,
    )


def _persist_durable_pre_restart(store: InMemoryDocumentStore) -> None:
    connection_repo = DocumentStoreTenantConnectionRepository(store)
    binding_repo = DocumentStoreKnowledgeSourceBindingRepository(store)
    TenantConnectionService(tenant_id=_TENANT, repository=connection_repo).create(_tenant_connection())
    binding_repo.create(_tenant_binding())


def _assert_registry_unregistered(registry: KnowledgeConnectionRegistry) -> None:
    with pytest.raises(VendorKnowledgeError) as exc_info:
        registry.resolve(
            tenant_id=_TENANT,
            connection_ref=_CONNECTION,
            provider_id=_PROVIDER,
            integration_kind=IntegrationCategory.ISSUE_TRACKER,
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.INTEGRATION_NOT_FOUND


def _assert_single_registry_registration(registry: KnowledgeConnectionRegistry, integration: object) -> None:
    resolved = registry.resolve(
        tenant_id=_TENANT,
        connection_ref=_CONNECTION,
        provider_id=_PROVIDER,
        integration_kind=IntegrationCategory.ISSUE_TRACKER,
    )
    assert resolved is integration
    with pytest.raises(ValueError, match="already registered"):
        registry.register(
            tenant_id=_TENANT,
            connection_ref=_CONNECTION,
            provider_id=_PROVIDER,
            integration_kind=IntegrationCategory.ISSUE_TRACKER,
            integration=FakeIntegration(),
        )


def _build_proof_context(
    store: InMemoryDocumentStore | None = None,
    *,
    persist: bool = True,
    workspaces: tuple[str, ...] = (_WORKSPACE,),
    secrets: _RecordingSecretsStore | None = None,
    integration: object | None = None,
) -> ProofContext:
    shared_store = store or InMemoryDocumentStore()
    if persist:
        _persist_durable_pre_restart(shared_store)

    connection_repo = DocumentStoreTenantConnectionRepository(shared_store)
    binding_repo = DocumentStoreKnowledgeSourceBindingRepository(shared_store)
    workspace_repo = ManagedWorkspaceRepository(shared_store)
    for workspace_id in workspaces:
        workspace_repo.put_workspace(_workspace(workspace_id))

    registry = KnowledgeConnectionRegistry()
    secrets_store = secrets or _RecordingSecretsStore()
    reconstructed = integration or FakeIntegration(
        provider_id=_PROVIDER,
        integration_kind=IntegrationCategory.ISSUE_TRACKER.value,
    )
    factory = _CountingFactory(integration=reconstructed)
    rehydrator = TenantConnectionRehydrator(
        repository=connection_repo,
        secrets_store=secrets_store,
        integration_factory=factory,
        connection_registry=registry,
    )

    _assert_registry_unregistered(registry)
    rehydration_results = rehydrator.rehydrate_tenant(tenant_id=_TENANT)
    assert len(rehydration_results) == 1
    assert rehydration_results[0].status is TenantConnectionRehydrationStatus.REGISTERED
    assert rehydration_results[0].error_code is None
    assert len(secrets_store.calls) == 1
    assert secrets_store.calls[0] == _CREDENTIAL_REF
    assert len(factory.calls) == 1
    _assert_single_registry_registration(registry, reconstructed)

    durable = connection_repo.get(tenant_id=_TENANT, connection_ref=_CONNECTION)
    assert durable is not None
    assert durable.connection_ref == _CONNECTION
    assert durable.configuration_version == 1

    fallback = _FailingFallbackResolver()
    connection_resolver = ConnectionAwareVendorResolver(
        tenant_id=_TENANT,
        connection_registry=registry,
        fallback_resolver=fallback,
    )
    adapter = FakeAdapter(
        provider_id=_PROVIDER,
        integration_kind=IntegrationCategory.ISSUE_TRACKER,
        source_kind="issues",
    )
    adapter_registry = KnowledgeAdapterRegistry()
    adapter_registry.register(adapter)
    binding_service = KnowledgeSourceBindingService(
        tenant_id=_TENANT,
        repository=binding_repo,
        integration_resolver=connection_resolver,
        adapter_registry=adapter_registry,
    )
    facade = VendorKnowledgeFacadeService(
        tenant_id=_TENANT,
        resolver=connection_resolver,
        adapter_registry=adapter_registry,
    )

    lookup = ManagedWorkspaceService(workspace_repo)
    config_service = WorkspaceKnowledgeConfigurationService(workspace_repo, lookup)
    mutation_ids = [f"mutation-{index}" for index in range(1, 20)]
    cursor = {"index": 0}

    def _next_mutation_id() -> str:
        value = mutation_ids[cursor["index"]]
        cursor["index"] = min(cursor["index"] + 1, len(mutation_ids) - 1)
        return value

    handlers = {
        WorkspaceKnowledgeMutationOperationV1.ATTACH_CONNECTION: AttachConnectionMutationHandler(),
        WorkspaceKnowledgeMutationOperationV1.CREATE_INDEXED_SOURCE: CreateIndexedSourceMutationHandler(),
        WorkspaceKnowledgeMutationOperationV1.DISABLE_INDEXED_SOURCE: DisableIndexedSourceMutationHandler(),
        WorkspaceKnowledgeMutationOperationV1.CREATE_LIVE_ACCESS_BINDING: CreateLiveAccessBindingMutationHandler(),
        WorkspaceKnowledgeMutationOperationV1.DISABLE_LIVE_ACCESS_BINDING: DisableLiveAccessBindingMutationHandler(),
    }
    mutation_engine = WorkspaceKnowledgeConfigurationMutationEngine(
        workspace_repo,
        lookup,
        config_service,
        handlers,
        clock=lambda: _NOW,
        mutation_id_factory=_next_mutation_id,
    )
    connection_port = RepositoryTenantConnectionPort(connection_repo)
    attach_service = WorkspaceConnectionAttachmentService(
        connection_port=connection_port,
        configuration_service=config_service,
        mutation_engine=mutation_engine,
    )
    tenant_binding_port = _RepositoryTenantBindingPort(binding_repo)
    indexed_lifecycle = WorkspaceIndexedSourceLifecycleService(
        repository=workspace_repo,
        configuration_service=config_service,
        mutation_engine=mutation_engine,
        tenant_binding_port=tenant_binding_port,
    )
    capability_catalog = TenantLiveCapabilityCatalog(connection_port=connection_port)
    capability_catalog.register(
        LiveCapabilityDescriptorV1(
            capability_id=_CAP_READ,
            provider_id=_PROVIDER,
            integration_kind=IntegrationCategory.ISSUE_TRACKER,
            effect=CapabilityEffectV1.READ,
            read_only=True,
            resource_scope_required=False,
            request_schema_ref="schema://proof/read/request",
            result_schema_ref="schema://proof/read/result",
            available=True,
        )
    )
    live_service = WorkspaceLiveAccessBindingService(
        repository=workspace_repo,
        configuration_service=config_service,
        mutation_engine=mutation_engine,
        tenant_connection_port=connection_port,
        capability_catalog=capability_catalog,
        remote_resource_lookup_port=None,
    )

    return ProofContext(
        store=shared_store,
        connection_repo=connection_repo,
        binding_repo=binding_repo,
        workspace_repo=workspace_repo,
        registry=registry,
        secrets=secrets_store,
        factory=factory,
        fallback=fallback,
        connection_resolver=connection_resolver,
        binding_service=binding_service,
        facade=facade,
        adapter=adapter,
        config_service=config_service,
        attach_service=attach_service,
        indexed_lifecycle=indexed_lifecycle,
        live_service=live_service,
        capability_catalog=capability_catalog,
        reconstructed_integration=reconstructed,
        rehydration_results=rehydration_results,
    )


async def _seed_active_bindings(ctx: ProofContext) -> int:
    revision = ctx.attach_service.attach_connection(
        AttachWorkspaceConnectionCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            connection_ref=_CONNECTION,
            expected_revision=0,
            idempotency_key_hash=_SHA256_A,
        )
    ).configuration_revision
    ctx.indexed_lifecycle.activate_indexed_source(
        ActivateWorkspaceIndexedSourceCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            knowledge_source_binding_ref=_BINDING_REF,
            expected_revision=revision,
            idempotency_key_hash=_SHA256_B,
            sync_mode=_SYNC,
            audience_eligibility=_AUDIENCE,
        )
    )
    revision += 1
    await ctx.live_service.create_live_access_binding(
        CreateWorkspaceLiveAccessBindingCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            connection_ref=_CONNECTION,
            remote_resource_id=None,
            allowed_capability_ids=(_CAP_READ,),
            expected_revision=revision,
            idempotency_key_hash=_SHA256_C,
            audience_eligibility=_AUDIENCE,
        )
    )
    return revision + 1


async def _run_indexed_probe(ctx: ProofContext) -> object:
    source = ctx.binding_service.resolve_source(_BINDING_REF)
    await ctx.facade.inspect_source(source=source)
    return ctx.adapter.inspect_calls[-1]["integration"]


def _run_live_probe(ctx: ProofContext) -> object:
    configuration = ctx.config_service.get_configuration(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert configuration is not None
    active = [
        binding
        for binding in configuration.live_access_bindings
        if binding.status is LiveAccessBindingStatusV1.ACTIVE
    ]
    assert len(active) == 1
    return _LiveResolverProbe(ctx.connection_resolver).resolve_integration(binding=active[0])


def _assert_cardinality(ctx: ProofContext, *, configuration_revision: int) -> None:
    assert len(ctx.connection_repo.list(tenant_id=_TENANT)) == 1
    assert len(ctx.binding_repo.list(tenant_id=_TENANT)) == 1
    configuration = ctx.config_service.get_configuration(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert configuration is not None
    assert configuration.configuration_revision == configuration_revision
    attached = [
        item
        for item in configuration.connection_attachments
        if item.status is WorkspaceConnectionAttachmentStatusV1.ATTACHED
    ]
    indexed = [
        item
        for item in configuration.indexed_sources
        if item.status is WorkspaceIndexedSourceBindingStatusV1.ACTIVE
    ]
    live = [
        item
        for item in configuration.live_access_bindings
        if item.status is LiveAccessBindingStatusV1.ACTIVE
    ]
    assert len(attached) == 1
    assert len(indexed) == 1
    assert len(live) == 1
    assert attached[0].connection_ref == _CONNECTION
    assert indexed[0].knowledge_source_binding_ref == _BINDING_REF
    assert live[0].connection_ref == _CONNECTION
    assert len(ctx.factory.calls) == 1
    _assert_single_registry_registration(ctx.registry, ctx.reconstructed_integration)


def _serialized_contains_forbidden(payload: str, *, forbidden: frozenset[str]) -> list[str]:
    return sorted(key for key in forbidden if f'"{key}"' in payload or f"'{key}'" in payload)


def _assert_leak_scan(ctx: ProofContext) -> None:
    configuration = ctx.config_service.get_configuration(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert configuration is not None
    source = ctx.workspace_repo.get_source(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        source_id=_SOURCE_ID,
    )
    assert source is not None
    records: list[tuple[str, dict[str, object]]] = [
        ("configuration", configuration.model_dump(mode="json")),
        ("workspace_source", source.model_dump(mode="json")),
    ]
    for attachment in configuration.connection_attachments:
        records.append((f"attachment:{attachment.attachment_id}", attachment.model_dump(mode="json")))
    for binding in configuration.indexed_sources:
        records.append((f"indexed:{binding.indexed_source_binding_id}", binding.model_dump(mode="json")))
    for binding in configuration.live_access_bindings:
        records.append((f"live:{binding.live_access_binding_id}", binding.model_dump(mode="json")))

    serialized = json.dumps(records, sort_keys=True)
    assert _SECRET not in serialized
    for label, payload in records:
        dumped = json.dumps(payload, sort_keys=True)
        assert _SECRET not in dumped, label
        leaked = _serialized_contains_forbidden(dumped, forbidden=_FORBIDDEN_LEAK_KEYS)
        assert leaked == [], f"{label} leaked forbidden keys: {leaked}"

    attachment = configuration.connection_attachments[0]
    attachment_dump = attachment.model_dump()
    for forbidden in (
        "provider_id",
        "integration_kind",
        "source_kind",
        "remote_scope_id",
        "remote_scope_type",
        "credential_ref",
    ):
        assert forbidden not in attachment_dump

    indexed = configuration.indexed_sources[0]
    indexed_dump = indexed.model_dump()
    for forbidden in (
        "connection_ref",
        "provider_id",
        "integration_kind",
        "source_kind",
        "remote_scope_id",
        "remote_scope_type",
        "credential_ref",
    ):
        assert forbidden not in indexed_dump

    live = configuration.live_access_bindings[0]
    live_dump = live.model_dump()
    assert live_dump["derived_provider_id"] == _PROVIDER
    assert live_dump["derived_integration_kind"] == IntegrationCategory.ISSUE_TRACKER.value
    for forbidden in ("credential_ref", "validated_secret_free_config", "source_kind"):
        assert forbidden not in live_dump


async def test_durable_connection_is_rehydrated_once_and_reused_by_indexed_and_live_paths() -> None:
    ctx = _build_proof_context()
    revision = await _seed_active_bindings(ctx)
    _assert_cardinality(ctx, configuration_revision=revision)

    configuration = ctx.config_service.get_configuration(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert configuration is not None
    indexed = [
        binding
        for binding in configuration.indexed_sources
        if binding.status is WorkspaceIndexedSourceBindingStatusV1.ACTIVE
    ]
    assert len(indexed) == 1
    assert indexed[0].knowledge_source_binding_ref == _BINDING_REF
    source = ctx.workspace_repo.get_source(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        source_id=_SOURCE_ID,
    )
    assert source is not None
    assert source.source_type is WorkspaceSourceType.CONNECTED_SOURCE

    indexed_integration = await _run_indexed_probe(ctx)
    live_integration = _run_live_probe(ctx)
    assert indexed_integration is ctx.reconstructed_integration
    assert live_integration is ctx.reconstructed_integration
    assert indexed_integration is live_integration

    await _run_indexed_probe(ctx)
    _run_live_probe(ctx)
    assert len(ctx.factory.calls) == 1
    assert len(ctx.fallback.calls) == 0
    _assert_single_registry_registration(ctx.registry, ctx.reconstructed_integration)
    _assert_leak_scan(ctx)


async def test_disabling_indexed_source_does_not_disable_live_access() -> None:
    ctx = _build_proof_context()
    revision = await _seed_active_bindings(ctx)
    disabled = ctx.indexed_lifecycle.disable_indexed_source(
        DisableWorkspaceIndexedSourceCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            indexed_source_binding_id=_INDEXED_BINDING_ID,
            expected_revision=revision,
            idempotency_key_hash=_SHA256_D,
        )
    )
    configuration = ctx.config_service.get_configuration(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert configuration is not None
    assert disabled.binding.status is WorkspaceIndexedSourceBindingStatusV1.DISABLED
    active_live = [
        binding
        for binding in configuration.live_access_bindings
        if binding.status is LiveAccessBindingStatusV1.ACTIVE
    ]
    assert len(active_live) == 1

    live_integration = _run_live_probe(ctx)
    assert live_integration is ctx.reconstructed_integration
    assert len(ctx.factory.calls) == 1

    durable_connection = ctx.connection_repo.get(tenant_id=_TENANT, connection_ref=_CONNECTION)
    assert durable_connection is not None
    assert durable_connection.administrative_status is TenantConnectionAdministrativeStatus.ACTIVE
    durable_binding = ctx.binding_repo.get(tenant_id=_TENANT, binding_id=_BINDING_REF)
    assert durable_binding is not None
    assert durable_binding.status is KnowledgeSourceBindingStatus.ACTIVE


async def test_disabling_live_access_does_not_disable_indexed_source() -> None:
    ctx = _build_proof_context()
    revision = await _seed_active_bindings(ctx)
    disabled = ctx.live_service.disable_live_access_binding(
        DisableWorkspaceLiveAccessBindingCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            live_access_binding_id=_LIVE_BINDING_ID,
            expected_revision=revision,
            idempotency_key_hash=_SHA256_D,
        )
    )
    configuration = ctx.config_service.get_configuration(tenant_id=_TENANT, workspace_id=_WORKSPACE)
    assert configuration is not None
    assert disabled.binding.status is LiveAccessBindingStatusV1.DISABLED
    active_indexed = [
        binding
        for binding in configuration.indexed_sources
        if binding.status is WorkspaceIndexedSourceBindingStatusV1.ACTIVE
    ]
    assert len(active_indexed) == 1

    indexed_integration = await _run_indexed_probe(ctx)
    assert indexed_integration is ctx.reconstructed_integration
    assert len(ctx.factory.calls) == 1


def test_cross_workspace_live_binding_attempt_returns_404() -> None:
    ctx = _build_proof_context(workspaces=(_WORKSPACE, _WORKSPACE_OTHER))
    revision = ctx.attach_service.attach_connection(
        AttachWorkspaceConnectionCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            connection_ref=_CONNECTION,
            expected_revision=0,
            idempotency_key_hash=_SHA256_A,
        )
    ).configuration_revision

    app = FastAPI()
    mount_knowledge_live_access_routes(
        app,
        live_access_service=ctx.live_service,
        repository=ctx.workspace_repo,
    )
    client = TestClient(app)
    response = client.post(
        "/v1/local_workspace/workspaces/workspace-other/knowledge/live-access-bindings",
        headers={
            "X-Tenant-Id": _TENANT,
            "If-Match": f"WKC/{revision}",
            "Idempotency-Key": "cross-workspace-proof",
        },
        json={"connection_ref": _CONNECTION, "allowed_capability_ids": [_CAP_READ]},
    )
    assert response.status_code == 404
    assert response.json()["detail"] == "connection_not_attached"
    assert _SECRET not in response.text
    assert _CREDENTIAL_REF not in response.text
    assert _PROVIDER not in response.json()

    other_configuration = ctx.config_service.get_configuration(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE_OTHER,
    )
    assert other_configuration is not None
    assert other_configuration.live_access_bindings == ()
    proof_configuration = ctx.config_service.get_configuration(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
    )
    assert proof_configuration is not None
    assert proof_configuration.live_access_bindings == ()


def test_missing_secret_keeps_durable_connection_and_skips_runtime_registration() -> None:
    store = InMemoryDocumentStore()
    missing_ref = "secrets/tenant-proof/missing"
    connection_repo = DocumentStoreTenantConnectionRepository(store)
    TenantConnectionService(tenant_id=_TENANT, repository=connection_repo).create(
        _tenant_connection(connection_ref="conn-missing", credential_ref=missing_ref)
    )

    restarted_repo = DocumentStoreTenantConnectionRepository(store)
    registry = KnowledgeConnectionRegistry()
    secrets = _RecordingSecretsStore(secret=None)
    factory = _CountingFactory()
    rehydrator = TenantConnectionRehydrator(
        repository=restarted_repo,
        secrets_store=secrets,
        integration_factory=factory,
        connection_registry=registry,
    )
    _assert_registry_unregistered(registry)
    results = rehydrator.rehydrate_tenant(tenant_id=_TENANT)
    assert len(results) == 1
    assert results[0].status is TenantConnectionRehydrationStatus.UNAVAILABLE
    assert results[0].error_code == "tenant_connection_secret_unavailable"
    assert factory.calls == []
    _assert_registry_unregistered(registry)

    durable = restarted_repo.get(tenant_id=_TENANT, connection_ref="conn-missing")
    assert durable is not None
    assert durable.connection_ref == "conn-missing"
    assert durable.administrative_status is TenantConnectionAdministrativeStatus.ACTIVE
    assert durable.configuration_version == 1
    safe = results[0].connection
    assert isinstance(safe, SafeTenantConnectionV1)
    assert safe.connection_ref == "conn-missing"
    assert "credential_ref" not in safe.model_dump()
