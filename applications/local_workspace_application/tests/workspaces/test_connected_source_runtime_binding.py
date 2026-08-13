# © Artur Czarnecki. All rights reserved.

"""Regression tests for connected-source runtime connection reconciliation."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.conversation_channel.slack.backend import (
    SlackConversationChannelBackend,
)
from intergrax.integrations.providers.conversation_channel.slack.config import (
    SlackConversationChannelIntegrationConfig,
)
from intergrax.integrations.providers.conversation_channel.slack.integration import (
    SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
)
from intergrax.integrations.providers.conversation_channel.slack.knowledge_read import (
    SLACK_CONVERSATION_SOURCE_KIND,
    SlackConversationExactMessageResult,
    SlackConversationInventoryPage,
    SlackConversationKind,
    SlackConversationMessage,
    SlackConversationMessagePage,
    SlackConversationSummary,
    SlackConversationKind as ReadSlackConversationKind,
    compute_slack_conversation_message_revision,
)
from intergrax.integrations.providers.conversation_channel.slack.mapping import (
    parse_slack_ts,
)
from intergrax.runtime.vendor_knowledge.adapters.slack_conversation import (
    SLACK_CONVERSATION_SCOPE_TYPE,
    encode_slack_conversation_scope_id,
)
from intergrax.runtime.vendor_knowledge.binding_document_store import (
    DocumentStoreKnowledgeSourceBindingRepository,
)
from intergrax.runtime.vendor_knowledge.bindings import (
    KnowledgeSourceBinding,
    KnowledgeSourceBindingStatus,
)
from intergrax.runtime.vendor_knowledge.connections import KnowledgeConnectionRegistry
from intergrax.runtime.vendor_knowledge.errors import VendorKnowledgeError, VendorKnowledgeErrorCode
from intergrax.runtime.vendor_knowledge.models import KnowledgeSourceScope
from intergrax.integrations.providers.conversation_channel.slack.tenant_connection_factory import (
    SlackTenantConnectionIntegrationFactory,
)
from intergrax.runtime.vendor_knowledge.tenant_connection_factory_registry import (
    TenantConnectionIntegrationFactoryRegistry,
)
from intergrax.runtime.vendor_knowledge.tenant_connection_document_store import (
    DocumentStoreTenantConnectionRepository,
)
from intergrax.runtime.vendor_knowledge.tenant_connection_rehydration import (
    TenantConnectionRehydrator,
    TenantConnectionRuntimeRegistryReconciler,
)
from intergrax.runtime.vendor_knowledge.tenant_connections import (
    TenantConnection,
    TenantConnectionAdministrativeStatus,
)
from intergrax.integrations.providers.conversation_channel.slack.integration import (
    SlackConversationChannelIntegration,
)
from intergrax.runtime.vendor_knowledge.sync_publication_fence import (
    DocumentStoreKnowledgeSyncPublicationFenceRepository,
    KnowledgeSyncPublicationFenceV1,
)
from local_workspace_application.host.settings import LocalWorkspaceBackendSettings
from local_workspace_application.workspaces.connected_source_wiring import (
    build_connected_source_wiring,
)
from local_workspace_application.workspaces.document_indexing import (
    WorkspaceDocumentIndexingResult,
    WorkspaceDocumentIndexingService,
)
from local_workspace_application.workspaces.knowledge_configuration_handlers import (
    CreateIndexedSourceMutationHandler,
)
from local_workspace_application.workspaces.knowledge_configuration_models import (
    WorkspaceIndexedSourceBinding,
    WorkspaceIndexedSourceBindingStatusV1,
    WorkspaceKnowledgeMutationOperationV1,
    WorkspaceKnowledgeMutationRecord,
    WorkspaceKnowledgeMutationStatusV1,
    IndexedSourceAudienceEligibilityV1,
    IndexedSourceSyncModeV1,
)
from local_workspace_application.workspaces.knowledge_configuration_mutation_engine import (
    WorkspaceKnowledgeConfigurationMutationEngine,
)
from local_workspace_application.workspaces.knowledge_configuration_service import (
    WorkspaceKnowledgeConfigurationService,
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

_TENANT = "tenant-runtime-binding"
_OTHER_TENANT = "tenant-other"
_WORKSPACE = "workspace-1"
_CONNECTION = "conn.slack.runtime"
_BINDING = "binding-runtime"
_INDEXED = "indexed-runtime"
_SOURCE = "source-runtime"
_CONVERSATION_ID = "C01234567"
_OLDEST = "1710000000.000000"
_LATEST = "1710003600.000000"
_SEMANTIC = "a" * 64
_SIGNING_KEY = "runtime-binding-signing-key"
_NOW = datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC)


class _SecretsStore:
    def get_secret(self, path: str) -> str:
        _ = path
        return '{"app_token":"xapp-test","bot_token":"xoxb-test"}'


class _SlackFakeBackend(SlackConversationChannelBackend):
    def __init__(self) -> None:
        super().__init__(
            config=SlackConversationChannelIntegrationConfig(
                enabled=True,
                app_token="xapp-test",
                bot_token="xoxb-test",
            )
        )

    async def list_accessible_conversations_page(self, *, cursor, limit):
        return SlackConversationInventoryPage(
            items=(
                SlackConversationSummary(
                    conversation_id=_CONVERSATION_ID,
                    kind=SlackConversationKind.PUBLIC_CHANNEL,
                    safe_name="#project-orion",
                    is_archived=False,
                    is_private=False,
                ),
            ),
            next_cursor=None,
        )

    async def read_conversation_history_page(self, **kwargs: Any) -> SlackConversationMessagePage:
        _ = kwargs
        message_ts = "1710000001.000001"
        created_at = parse_slack_ts(message_ts) or _NOW
        return SlackConversationMessagePage(
            conversation_id=_CONVERSATION_ID,
            oldest=_OLDEST,
            latest=_LATEST,
            items=(
                SlackConversationMessage(
                    conversation_id=_CONVERSATION_ID,
                    message_ts=message_ts,
                    root_thread_ts=None,
                    actor_provider_id="U111",
                    text="runtime-binding-marker",
                    subtype=None,
                    created_at=created_at,
                    edited_at=None,
                    reply_count=0,
                    files=(),
                    provider_metadata={},
                ),
            ),
            next_cursor=None,
            has_more=False,
        )

    async def read_thread_replies_page(self, **kwargs: Any) -> SlackConversationMessagePage:
        _ = kwargs
        return SlackConversationMessagePage(
            conversation_id=_CONVERSATION_ID,
            oldest=_OLDEST,
            latest=_LATEST,
            items=(),
            next_cursor=None,
            has_more=False,
        )

    async def read_exact_message(self, **kwargs: Any) -> SlackConversationExactMessageResult:
        message_ts = str(kwargs["message_ts"])
        created_at = parse_slack_ts(message_ts) or _NOW
        message = SlackConversationMessage(
            conversation_id=_CONVERSATION_ID,
            message_ts=message_ts,
            root_thread_ts=None,
            actor_provider_id="U111",
            text="runtime-binding",
            subtype=None,
            created_at=created_at,
            edited_at=None,
            reply_count=0,
            files=(),
            provider_metadata={},
        )
        revision = kwargs.get("expected_revision")
        if revision is not None and revision != compute_slack_conversation_message_revision(
            message
        ):
            from intergrax.integrations.providers.conversation_channel.slack.knowledge_read import (
                SlackConversationMessageChanged,
            )

            raise SlackConversationMessageChanged()
        return SlackConversationExactMessageResult(found=True, message=message)


def _factory_registry() -> TenantConnectionIntegrationFactoryRegistry:
    return TenantConnectionIntegrationFactoryRegistry(
        [
            (
                SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
                IntegrationCategory.CONVERSATION_CHANNEL,
                SlackTenantConnectionIntegrationFactory(
                    runtime_builder=lambda config: SlackConversationChannelIntegration.from_backend(
                        _SlackFakeBackend(),
                        enabled=True,
                        config=config,
                    )
                ),
            ),
        ]
    )


class _IndexingService:
    async def index_connected_source_one(self, **kwargs: Any) -> WorkspaceDocumentIndexingResult:
        _ = kwargs
        return WorkspaceDocumentIndexingResult(
            indexed=True,
            unchanged=False,
            document_id="doc-1",
            documents_indexed=1,
        )


@dataclass
class _SyncEnv:
    repo: ManagedWorkspaceRepository
    connected_sync: Any
    registry: KnowledgeConnectionRegistry
    operation_id: str


def _durable_connection(
    *,
    tenant_id: str = _TENANT,
    connection_ref: str = _CONNECTION,
    status: TenantConnectionAdministrativeStatus = TenantConnectionAdministrativeStatus.ACTIVE,
) -> TenantConnection:
    return TenantConnection(
        connection_ref=connection_ref,
        tenant_id=tenant_id,
        provider_id=SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
        integration_kind=IntegrationCategory.CONVERSATION_CHANNEL,
        safe_display_name="Slack",
        administrative_status=status,
        credential_ref="secrets/runtime/slack",
        validated_secret_free_config={},
        configuration_version=1,
        created_at=_NOW,
        updated_at=_NOW,
        connected_principal_ref="slack_team:T0TEST",
    )


def _seed_repo(repo: ManagedWorkspaceRepository) -> None:
    repo.put_workspace(
        Workspace(
            workspace_id=_WORKSPACE,
            tenant_id=_TENANT,
            name="ws",
            status=WorkspaceStatus.ACTIVE,
            created_at=_NOW,
            updated_at=_NOW,
        )
    )
    repo.put_knowledge_indexed_source_version_if_absent(
        WorkspaceIndexedSourceBinding(
            indexed_source_binding_id=_INDEXED,
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            knowledge_source_binding_ref=_BINDING,
            source_id=_SOURCE,
            sync_mode=IndexedSourceSyncModeV1.FULL,
            status=WorkspaceIndexedSourceBindingStatusV1.ACTIVE,
            audience_eligibility=IndexedSourceAudienceEligibilityV1.PERSONAL_ONLY,
            mutation_id="mut-runtime",
            effective_revision=1,
            semantic_identity_hash=_SEMANTIC,
            created_at=_NOW,
            updated_at=_NOW,
            cached_safe_display_label="#project-orion",
        )
    )
    head_mod = __import__(
        "local_workspace_application.workspaces.knowledge_configuration_models",
        fromlist=["WorkspaceKnowledgeConfigurationHead"],
    )
    repo.put_knowledge_configuration_head_if_absent(
        head_mod.WorkspaceKnowledgeConfigurationHead(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            committed_revision=1,
            updated_at=_NOW,
        )
    )
    repo.put_knowledge_configuration_mutation_if_absent(
        WorkspaceKnowledgeMutationRecord(
            mutation_id="mut-runtime",
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            operation=WorkspaceKnowledgeMutationOperationV1.CREATE_INDEXED_SOURCE,
            idempotency_key_hash="f" * 64,
            normalized_request_hash="a" * 64,
            semantic_identity_hash=_SEMANTIC,
            target_revision=1,
            committed_revision=1,
            status=WorkspaceKnowledgeMutationStatusV1.COMMITTED,
            outcome=__import__(
                "local_workspace_application.workspaces.knowledge_configuration_models",
                fromlist=["WorkspaceKnowledgeMutationOutcomeV1"],
            ).WorkspaceKnowledgeMutationOutcomeV1.APPLIED,
            result_entity_type="indexed_source_binding",
            result_entity_id=_INDEXED,
            created_at=_NOW,
            updated_at=_NOW,
            committed_at=_NOW,
        )
    )
    repo.put_source(
        WorkspaceSource(
            source_id=_SOURCE,
            workspace_id=_WORKSPACE,
            tenant_id=_TENANT,
            source_type=WorkspaceSourceType.CONNECTED_SOURCE,
            path="",
            recursive=False,
            status=WorkspaceSourceStatus.REGISTERED,
            created_at=_NOW,
            knowledge_configuration_creation_mutation_id="mut-runtime",
            knowledge_configuration_visibility_revision=1,
        )
    )
    binding_repo = DocumentStoreKnowledgeSourceBindingRepository(repo.document_store)
    encoded_scope = encode_slack_conversation_scope_id(
        conversation_id=_CONVERSATION_ID,
        conversation_kind=ReadSlackConversationKind.PUBLIC_CHANNEL,
        oldest=_OLDEST,
        latest=_LATEST,
    )
    binding_repo.create(
        KnowledgeSourceBinding(
            binding_id=_BINDING,
            tenant_id=_TENANT,
            provider_id=SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
            integration_kind=IntegrationCategory.CONVERSATION_CHANNEL,
            source_kind=SLACK_CONVERSATION_SOURCE_KIND,
            connection_ref=_CONNECTION,
            safe_display_name="#project-orion",
            scope=KnowledgeSourceScope(
                remote_scope_id=encoded_scope,
                remote_scope_type=SLACK_CONVERSATION_SCOPE_TYPE,
                safe_display_name="#project-orion",
                parameters={},
            ),
            status=KnowledgeSourceBindingStatus.ACTIVE,
            configuration_version=1,
        )
    )
    publication_fence = DocumentStoreKnowledgeSyncPublicationFenceRepository(
        repo.document_store
    )
    publication_fence.write_fence(
        KnowledgeSyncPublicationFenceV1(
            tenant_id=_TENANT,
            binding_id=_BINDING,
            lifecycle_revision=1,
            lifecycle_token="runtime-binding-token",
            enabled=True,
            detached=False,
        ),
        expected_revision=None,
    )


def _build_sync_env(tmp_path: Path, *, register_runtime: bool) -> _SyncEnv:
    store = InMemoryDocumentStore()
    repo = ManagedWorkspaceRepository(store)
    _seed_repo(repo)
    connection_repository = DocumentStoreTenantConnectionRepository(store)
    connection_repository.create(_durable_connection())
    settings = LocalWorkspaceBackendSettings(
        data_home=str(tmp_path / "data"),
        connected_source_opaque_ref_signing_key=_SIGNING_KEY,
    )
    service = ManagedWorkspaceService(repo)
    config = WorkspaceKnowledgeConfigurationService(repo, service)
    mutation_engine = WorkspaceKnowledgeConfigurationMutationEngine(
        repo,
        service,
        config,
        {
            WorkspaceKnowledgeMutationOperationV1.CREATE_INDEXED_SOURCE: (
                CreateIndexedSourceMutationHandler()
            ),
        },
    )
    registry = KnowledgeConnectionRegistry()
    secrets = _SecretsStore()
    factory_registry = _factory_registry()
    rehydrator = TenantConnectionRehydrator(
        repository=connection_repository,
        secrets_store=secrets,
        integration_factory=factory_registry,
        connection_registry=registry,
    )
    if register_runtime:
        result = rehydrator.rehydrate_connection(
            tenant_id=_TENANT,
            connection_ref=_CONNECTION,
        )
        assert result.status.value == "registered"

    wiring = build_connected_source_wiring(
        repository=repo,
        workspace_service=service,
        configuration_service=config,
        mutation_engine=mutation_engine,
        indexing_service=_IndexingService(),  # type: ignore[arg-type]
        settings=settings,
        connection_registry=registry,
        tenant_connection_rehydrator=rehydrator,
    )
    connected_sync = wiring.connected_source_sync_service
    operation = service.create_sync_operation(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        source_id=_SOURCE,
    )
    return _SyncEnv(
        repo=repo,
        connected_sync=connected_sync,
        registry=registry,
        operation_id=operation.operation_id,
    )


def test_runtime_reconciler_registers_durable_connection_when_registry_is_empty(
    tmp_path: Path,
) -> None:
    env = _build_sync_env(tmp_path, register_runtime=False)
    with pytest.raises(VendorKnowledgeError):
        env.registry.resolve(
            tenant_id=_TENANT,
            connection_ref=_CONNECTION,
            provider_id=SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
            integration_kind=IntegrationCategory.CONVERSATION_CHANNEL,
        )
    reconciler = TenantConnectionRuntimeRegistryReconciler(
        rehydrator=TenantConnectionRehydrator(
            repository=DocumentStoreTenantConnectionRepository(env.repo.document_store),
            secrets_store=_SecretsStore(),
            integration_factory=_factory_registry(),
            connection_registry=env.registry,
        ),
        connection_registry=env.registry,
    )
    reconciler.ensure_registered(
        tenant_id=_TENANT,
        connection_ref=_CONNECTION,
        provider_id=SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
        integration_kind=IntegrationCategory.CONVERSATION_CHANNEL,
    )
    env.registry.resolve(
        tenant_id=_TENANT,
        connection_ref=_CONNECTION,
        provider_id=SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
        integration_kind=IntegrationCategory.CONVERSATION_CHANNEL,
    )


@pytest.mark.asyncio
async def test_sync_does_not_fail_with_integration_not_found_when_registry_is_empty(
    tmp_path: Path,
) -> None:
    env = _build_sync_env(tmp_path, register_runtime=False)
    result = await env.connected_sync.run_operation(
        tenant_id=_TENANT,
        operation_id=env.operation_id,
    )
    assert result.error != "integration_not_found"


@pytest.mark.asyncio
async def test_sync_runtime_reconciliation_rejects_wrong_tenant(tmp_path: Path) -> None:
    env = _build_sync_env(tmp_path, register_runtime=False)
    reconciler = TenantConnectionRuntimeRegistryReconciler(
        rehydrator=TenantConnectionRehydrator(
            repository=DocumentStoreTenantConnectionRepository(env.repo.document_store),
            secrets_store=_SecretsStore(),
            integration_factory=_factory_registry(),
            connection_registry=env.registry,
        ),
        connection_registry=env.registry,
    )
    with pytest.raises(VendorKnowledgeError) as exc_info:
        reconciler.ensure_registered(
            tenant_id=_OTHER_TENANT,
            connection_ref=_CONNECTION,
            provider_id=SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
            integration_kind=IntegrationCategory.CONVERSATION_CHANNEL,
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.INTEGRATION_NOT_FOUND


@pytest.mark.asyncio
async def test_sync_runtime_reconciliation_rejects_revoked_connection(
    tmp_path: Path,
) -> None:
    store = InMemoryDocumentStore()
    repo = ManagedWorkspaceRepository(store)
    connection_repository = DocumentStoreTenantConnectionRepository(store)
    connection_repository.create(
        _durable_connection(status=TenantConnectionAdministrativeStatus.REVOKED)
    )
    registry = KnowledgeConnectionRegistry()
    reconciler = TenantConnectionRuntimeRegistryReconciler(
        rehydrator=TenantConnectionRehydrator(
            repository=connection_repository,
            secrets_store=_SecretsStore(),
            integration_factory=_factory_registry(),
            connection_registry=registry,
        ),
        connection_registry=registry,
    )
    with pytest.raises(VendorKnowledgeError) as exc_info:
        reconciler.ensure_registered(
            tenant_id=_TENANT,
            connection_ref=_CONNECTION,
            provider_id=SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
            integration_kind=IntegrationCategory.CONVERSATION_CHANNEL,
        )
    assert exc_info.value.code is VendorKnowledgeErrorCode.INTEGRATION_NOT_FOUND


def test_connection_registry_refresh_replaces_runtime_instance() -> None:
    registry = KnowledgeConnectionRegistry()
    first = SlackConversationChannelIntegration.from_backend(
        _SlackFakeBackend(),
        enabled=True,
        config=SlackConversationChannelIntegrationConfig(
            enabled=True,
            app_token="xapp-test",
            bot_token="xoxb-test",
        ),
    )
    second = SlackConversationChannelIntegration.from_backend(
        _SlackFakeBackend(),
        enabled=True,
        config=SlackConversationChannelIntegrationConfig(
            enabled=True,
            app_token="xapp-test",
            bot_token="xoxb-other",
        ),
    )
    registry.register(
        tenant_id=_TENANT,
        connection_ref=_CONNECTION,
        provider_id=SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
        integration_kind=IntegrationCategory.CONVERSATION_CHANNEL,
        integration=first,
    )
    registry.refresh(
        tenant_id=_TENANT,
        connection_ref=_CONNECTION,
        provider_id=SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
        integration_kind=IntegrationCategory.CONVERSATION_CHANNEL,
        integration=second,
    )
    resolved = registry.resolve(
        tenant_id=_TENANT,
        connection_ref=_CONNECTION,
        provider_id=SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
        integration_kind=IntegrationCategory.CONVERSATION_CHANNEL,
    )
    assert resolved is second
    assert resolved is not first
