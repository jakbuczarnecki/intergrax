# © Artur Czarnecki. All rights reserved.

"""Host wiring tests for shared Slack connected-source integration."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.integrations.contracts.base import HealthStatus, IntegrationCategory
from intergrax.integrations.contracts.conversation_channel import (
    ConversationDeliveryReceipt,
    ConversationEventHandler,
    OutboundConversationMessage,
)
from intergrax.integrations.providers.conversation_channel.slack.integration import (
    SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
    SlackConversationChannelIntegration,
)
from intergrax.integrations.providers.conversation_channel.slack.knowledge_read import (
    SlackConversationInventoryPage,
    SlackConversationKind,
    SlackConversationSummary,
)
from local_workspace_application.host.factory import create_local_workspace_backend_app
from local_workspace_application.host.settings import LocalWorkspaceBackendSettings
from local_workspace_application.slack_companion.companion import COMPONENT_NAME
from local_workspace_application.workspaces.connected_source_host_wiring import (
    ConnectedSourceReadinessState,
    build_connected_source_host_bundle,
)
from local_workspace_application.workspaces.document_indexing import WorkspaceDocumentIndexingService
from local_workspace_application.workspaces.knowledge_configuration_handlers import (
    CreateIndexedSourceMutationHandler,
)
from local_workspace_application.workspaces.knowledge_configuration_models import (
    WorkspaceKnowledgeMutationOperationV1,
)
from local_workspace_application.workspaces.knowledge_configuration_mutation_engine import (
    WorkspaceKnowledgeConfigurationMutationEngine,
)
from local_workspace_application.workspaces.knowledge_configuration_service import (
    WorkspaceKnowledgeConfigurationService,
)
from local_workspace_application.workspaces.repository import ManagedWorkspaceRepository
from local_workspace_application.workspaces.service import ManagedWorkspaceService

pytestmark = pytest.mark.unit

_TENANT = "tenant-a"
_CONNECTION = "conn.slack"
_SIGNING_KEY = "host-connected-source-signing-key"
_NOW = datetime(2024, 6, 1, 12, 0, tzinfo=UTC)


class _FakeBackend:
    def __init__(self) -> None:
        self.started = False

    async def list_accessible_conversations_page(self, *, cursor, limit):
        return SlackConversationInventoryPage(
            items=(
                SlackConversationSummary(
                    conversation_id="C01234567",
                    kind=SlackConversationKind.PUBLIC_CHANNEL,
                    safe_name="#project-orion",
                    is_archived=False,
                    is_private=False,
                ),
            ),
            next_cursor=None,
        )

    async def read_conversation_history_page(self, **kwargs):
        raise NotImplementedError

    async def read_thread_replies_page(self, **kwargs):
        raise NotImplementedError

    async def read_exact_message(self, **kwargs):
        raise NotImplementedError

    async def read_file_info(self, **kwargs):
        raise NotImplementedError

    async def start(self, handler: ConversationEventHandler) -> None:
        _ = handler
        self.started = True

    async def stop(self) -> None:
        self.started = False

    async def send(self, message: OutboundConversationMessage) -> ConversationDeliveryReceipt:
        return ConversationDeliveryReceipt(
            message_id="m1",
            address=message.address,
            delivered_at=datetime.now(UTC),
        )

    def health(self) -> HealthStatus:
        return HealthStatus(slug="slack", healthy=True, detail="fake-ok")


def _host_services(store: InMemoryDocumentStore, settings: LocalWorkspaceBackendSettings):
    repo = ManagedWorkspaceRepository(store)
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
    indexing = WorkspaceDocumentIndexingService(repo, task_executor=object())  # type: ignore[arg-type]
    return repo, service, config, mutation_engine, indexing, settings


def test_build_connected_source_host_bundle_shares_slack_integration() -> None:
    store = InMemoryDocumentStore()
    settings = LocalWorkspaceBackendSettings(
        data_home="/tmp/lkw-host-wiring",
        connected_source_opaque_ref_signing_key=_SIGNING_KEY,
        slack_tenant_id=_TENANT,
        connected_source_slack_connection_ref=_CONNECTION,
    )
    repo, service, config, mutation_engine, indexing, settings = _host_services(store, settings)
    backend = _FakeBackend()
    integration = SlackConversationChannelIntegration.from_backend(
        backend,  # type: ignore[arg-type]
        enabled=True,
    )

    bundle = build_connected_source_host_bundle(
        settings=settings,
        repository=repo,
        workspace_service=service,
        configuration_service=config,
        mutation_engine=mutation_engine,
        indexing_service=indexing,
        slack_integration=integration,
    )

    assert bundle.wiring is not None
    assert bundle.slack_integration is integration
    resolved = bundle.wiring.connection_registry.resolve(
        tenant_id=_TENANT,
        connection_ref=_CONNECTION,
        provider_id=SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
        integration_kind=IntegrationCategory.CONVERSATION_CHANNEL,
    )
    assert resolved is integration


def test_missing_signing_key_returns_no_wiring() -> None:
    store = InMemoryDocumentStore()
    settings = LocalWorkspaceBackendSettings(
        data_home="/tmp/lkw-host-wiring",
        connected_source_opaque_ref_signing_key="",
        slack_tenant_id=_TENANT,
        connected_source_slack_connection_ref=_CONNECTION,
        slack_companion_enabled=True,
    )
    repo, service, config, mutation_engine, indexing, settings = _host_services(store, settings)

    bundle = build_connected_source_host_bundle(
        settings=settings,
        repository=repo,
        workspace_service=service,
        configuration_service=config,
        mutation_engine=mutation_engine,
        indexing_service=indexing,
        slack_integration=SlackConversationChannelIntegration.from_backend(
            _FakeBackend(),  # type: ignore[arg-type]
            enabled=True,
        ),
    )

    assert bundle.wiring is None
    assert bundle.slack_integration is None
    assert bundle.readiness.state.value == "signing_key_missing"
    assert bundle.readiness.signing_key_configured is False
    assert bundle.readiness.reason == "connected_source_signing_key_missing"


def test_create_local_workspace_backend_app_wires_connected_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = InMemoryDocumentStore()
    data_home = tmp_path / "data"
    sqlite_dir = tmp_path / "sqlite"
    shadow_dir = tmp_path / "shadow"
    for path in (data_home, sqlite_dir, shadow_dir):
        path.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("LOCAL_WORKSPACE_VECTOR_STORE", "inmemory")
    monkeypatch.setenv("LOCAL_WORKSPACE_INCLUDE_MCP", "false")
    monkeypatch.setenv("LOCAL_WORKSPACE_INCLUDE_SCHEDULER", "false")
    monkeypatch.setenv("DATA_HOME", str(data_home))
    monkeypatch.setenv("LKW_DATA_HOME", str(data_home))
    monkeypatch.setenv("INTERGRAX_SQLITE_DATA_DIR", str(sqlite_dir))
    monkeypatch.setenv("INTERGRAX_SHADOW_ROOT", str(shadow_dir))
    monkeypatch.setenv("LOCAL_WORKSPACE_CONNECTED_SOURCE_OPAQUE_REF_SIGNING_KEY", _SIGNING_KEY)
    monkeypatch.setenv("LOCAL_WORKSPACE_SLACK_TENANT_ID", _TENANT)
    monkeypatch.setenv("LOCAL_WORKSPACE_CONNECTED_SOURCE_SLACK_CONNECTION_REF", _CONNECTION)
    monkeypatch.setenv("LOCAL_WORKSPACE_SLACK_COMPANION_ENABLED", "true")
    monkeypatch.setenv("LOCAL_WORKSPACE_SLACK_APPROVED_TEAM_ID", "T1")
    monkeypatch.setenv("LOCAL_WORKSPACE_SLACK_APPROVED_USER_ID", "U1")
    monkeypatch.setenv("LOCAL_WORKSPACE_SLACK_ACTIVE_WORKSPACE_ID", "workspace-1")
    monkeypatch.setenv("LOCAL_WORKSPACE_SLACK_ASK_BASE_URL", "http://127.0.0.1:8020")
    monkeypatch.delenv("INTERGRAX_MONGODB_URI", raising=False)
    monkeypatch.setattr(
        "local_workspace_application.serving.workspace_routes.resolve_managed_workspace_document_store",
        lambda document_store=None: store,
    )

    backend = _FakeBackend()
    integration = SlackConversationChannelIntegration.from_backend(
        backend,  # type: ignore[arg-type]
        enabled=True,
    )
    monkeypatch.setattr(
        "local_workspace_application.workspaces.connected_source_host_wiring.build_shared_slack_integration_for_host",
        lambda: integration,
    )

    settings = replace(
        LocalWorkspaceBackendSettings.from_env(),
        data_home=str(data_home),
        connected_source_opaque_ref_signing_key=_SIGNING_KEY,
        slack_tenant_id=_TENANT,
        connected_source_slack_connection_ref=_CONNECTION,
        slack_companion_enabled=True,
        slack_approved_team_id="T1",
        slack_approved_user_id="U1",
        slack_active_workspace_id="workspace-1",
        slack_ask_base_url="http://127.0.0.1:8020",
        include_mcp=False,
        include_scheduler=False,
    )
    app = create_local_workspace_backend_app(settings=settings)

    with TestClient(app):
        wiring = app.state.lkw_connected_source_wiring
        assert wiring is not None
        resolved = wiring.connection_registry.resolve(
            tenant_id=_TENANT,
            connection_ref=_CONNECTION,
            provider_id=SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
            integration_kind=IntegrationCategory.CONVERSATION_CHANNEL,
        )
        assert resolved is integration
        assert backend.started is True
        lifecycle = app.state.lkw_host_lifecycle
        slack = next(item for item in lifecycle.component_health() if item.name == COMPONENT_NAME)
        assert slack.enabled is True


@pytest.mark.parametrize(
    ("signing_key", "companion", "slack_integration", "tenant", "connection", "expected_state"),
    [
        ("", False, None, "", "", ConnectedSourceReadinessState.DISABLED),
        ("", True, object(), _TENANT, _CONNECTION, ConnectedSourceReadinessState.SIGNING_KEY_MISSING),
        (_SIGNING_KEY, False, None, _TENANT, _CONNECTION, ConnectedSourceReadinessState.SLACK_INTEGRATION_UNAVAILABLE),
        (_SIGNING_KEY, False, object(), _TENANT, "", ConnectedSourceReadinessState.MAPPING_INCOMPLETE),
        (_SIGNING_KEY, False, object(), _TENANT, _CONNECTION, ConnectedSourceReadinessState.READY),
    ],
)
def test_connected_source_readiness_state_matrix(
    signing_key: str,
    companion: bool,
    slack_integration: object | None,
    tenant: str,
    connection: str,
    expected_state: ConnectedSourceReadinessState,
) -> None:
    store = InMemoryDocumentStore()
    settings = LocalWorkspaceBackendSettings(
        data_home="/tmp/lkw-host-readiness",
        connected_source_opaque_ref_signing_key=signing_key,
        slack_tenant_id=tenant,
        connected_source_slack_connection_ref=connection,
        slack_companion_enabled=companion,
    )
    repo, service, config, mutation_engine, indexing, settings = _host_services(store, settings)
    integration = None
    if slack_integration is not None:
        integration = SlackConversationChannelIntegration.from_backend(
            _FakeBackend(),  # type: ignore[arg-type]
            enabled=True,
        )
    bundle = build_connected_source_host_bundle(
        settings=settings,
        repository=repo,
        workspace_service=service,
        configuration_service=config,
        mutation_engine=mutation_engine,
        indexing_service=indexing,
        slack_integration=integration,
    )
    assert bundle.readiness.state is expected_state
    if expected_state is ConnectedSourceReadinessState.READY:
        assert bundle.wiring is not None
    else:
        assert bundle.wiring is None


@pytest.mark.parametrize(
    (
        "signing_key",
        "companion",
        "use_slack_integration",
        "tenant",
        "connection",
        "expected_state",
    ),
    [
        ("", False, False, "", "", ConnectedSourceReadinessState.DISABLED),
        ("", True, True, _TENANT, _CONNECTION, ConnectedSourceReadinessState.SIGNING_KEY_MISSING),
        (_SIGNING_KEY, False, False, _TENANT, _CONNECTION, ConnectedSourceReadinessState.SLACK_INTEGRATION_UNAVAILABLE),
        (_SIGNING_KEY, False, True, _TENANT, "", ConnectedSourceReadinessState.MAPPING_INCOMPLETE),
        (_SIGNING_KEY, False, True, _TENANT, _CONNECTION, ConnectedSourceReadinessState.READY),
    ],
)
def test_create_local_workspace_backend_app_connected_source_readiness_states(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    signing_key: str,
    companion: bool,
    use_slack_integration: bool,
    tenant: str,
    connection: str,
    expected_state: ConnectedSourceReadinessState,
) -> None:
    store = InMemoryDocumentStore()
    data_home = tmp_path / "data"
    sqlite_dir = tmp_path / "sqlite"
    shadow_dir = tmp_path / "shadow"
    for path in (data_home, sqlite_dir, shadow_dir):
        path.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("LOCAL_WORKSPACE_VECTOR_STORE", "inmemory")
    monkeypatch.setenv("LOCAL_WORKSPACE_INCLUDE_MCP", "false")
    monkeypatch.setenv("LOCAL_WORKSPACE_INCLUDE_SCHEDULER", "false")
    monkeypatch.setenv("DATA_HOME", str(data_home))
    monkeypatch.setenv("LKW_DATA_HOME", str(data_home))
    monkeypatch.setenv("INTERGRAX_SQLITE_DATA_DIR", str(sqlite_dir))
    monkeypatch.setenv("INTERGRAX_SHADOW_ROOT", str(shadow_dir))
    monkeypatch.setenv("LOCAL_WORKSPACE_CONNECTED_SOURCE_OPAQUE_REF_SIGNING_KEY", signing_key)
    monkeypatch.setenv("LOCAL_WORKSPACE_SLACK_TENANT_ID", tenant)
    monkeypatch.setenv("LOCAL_WORKSPACE_CONNECTED_SOURCE_SLACK_CONNECTION_REF", connection)
    monkeypatch.setenv("LOCAL_WORKSPACE_SLACK_COMPANION_ENABLED", "true" if companion else "false")
    monkeypatch.delenv("INTERGRAX_MONGODB_URI", raising=False)
    monkeypatch.setattr(
        "local_workspace_application.serving.workspace_routes.resolve_managed_workspace_document_store",
        lambda document_store=None: store,
    )

    integration = None
    if use_slack_integration:
        integration = SlackConversationChannelIntegration.from_backend(
            _FakeBackend(),  # type: ignore[arg-type]
            enabled=True,
        )
        monkeypatch.setattr(
            "local_workspace_application.workspaces.connected_source_host_wiring.build_shared_slack_integration_for_host",
            lambda: integration,
        )

    settings = replace(
        LocalWorkspaceBackendSettings.from_env(),
        data_home=str(data_home),
        connected_source_opaque_ref_signing_key=signing_key,
        slack_tenant_id=tenant,
        connected_source_slack_connection_ref=connection,
        slack_companion_enabled=companion,
        include_mcp=False,
        include_scheduler=False,
    )
    app = create_local_workspace_backend_app(settings=settings)

    with TestClient(app):
        readiness = app.state.lkw_connected_source_readiness
        assert readiness.state is expected_state
        if expected_state is ConnectedSourceReadinessState.READY:
            assert hasattr(app.state, "lkw_connected_source_wiring")
            assert app.state.lkw_connected_source_wiring is not None
        else:
            assert not hasattr(app.state, "lkw_connected_source_wiring")
