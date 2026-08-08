# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from intergrax.integrations.contracts.base import (
    IntegrationCategory,
    IntegrationConfigurationError,
)
from intergrax.integrations.providers.conversation_channel.slack.integration import (
    SlackConversationChannelIntegration,
)
from intergrax.integrations.providers.conversation_channel.slack.tenant_connection_factory import (
    SlackTenantConnectionIntegrationFactory,
)
from intergrax.runtime.vendor_knowledge.connections import KnowledgeConnectionRegistry
from intergrax.runtime.vendor_knowledge.errors import VendorKnowledgeError
from intergrax.runtime.vendor_knowledge.plugin import VendorKnowledgeMode
from intergrax.runtime.vendor_knowledge.plugin_composition import (
    build_default_vendor_knowledge_source_plugin_registry,
)
from intergrax.runtime.vendor_knowledge.source_catalog import (
    TenantVendorKnowledgeSourceCatalog,
)
from intergrax.runtime.vendor_knowledge.tenant_connection_capabilities import (
    RepositoryTenantConnectionPort,
)
from intergrax.runtime.vendor_knowledge.tenant_connection_document_store import (
    DocumentStoreTenantConnectionRepository,
)
from intergrax.runtime.vendor_knowledge.tenant_connection_rehydration import (
    TenantConnectionRehydrationStatus,
    TenantConnectionRehydrator,
)
from intergrax.runtime.vendor_knowledge.tenant_connections import (
    TenantConnection,
    TenantConnectionAdministrativeStatus,
)
from tests.unit.runtime.vendor_knowledge.test_tenant_connection_document_store import (
    ConditionalInMemoryDocumentStore,
)


class _FakeBackend:
    pass


def _factory() -> SlackTenantConnectionIntegrationFactory:
    return SlackTenantConnectionIntegrationFactory(
        runtime_builder=lambda config: SlackConversationChannelIntegration.from_backend(
            _FakeBackend(),  # type: ignore[arg-type]
            enabled=True,
            config=config,
        ),
    )


def _kwargs(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "tenant_id": "tenant-1",
        "connection_ref": "slack-connection",
        "provider_id": "slack",
        "integration_kind": IntegrationCategory.CONVERSATION_CHANNEL,
        "credential_ref": "secrets/tenant-1/slack",
        "credential": json.dumps(
            {
                "app_token": "xapp-test",
                "bot_token": "xoxb-test",
            }
        ),
        "secret_free_config": {"api_timeout_seconds": 15.0},
    }
    payload.update(overrides)
    return payload


@pytest.mark.unit
def test_factory_builds_slack_runtime_from_secret_ref_payload() -> None:
    integration = _factory().create_integration(**_kwargs())

    assert isinstance(integration, SlackConversationChannelIntegration)
    assert integration.config.enabled is True
    assert integration.config.api_timeout_seconds == 15.0
    assert integration.config.require_runtime_tokens() == ("xapp-test", "xoxb-test")


@pytest.mark.unit
@pytest.mark.parametrize(
    ("credential", "message"),
    [
        ("not-json", "valid JSON"),
        (json.dumps({"app_token": "xapp-test"}), "missing bot_token"),
        (json.dumps({"bot_token": "xoxb-test"}), "missing app_token"),
        (json.dumps({"app_token": "bad", "bot_token": "xoxb-test"}), "app_token"),
    ],
)
def test_factory_rejects_invalid_credentials_without_echoing_secret(
    credential: str,
    message: str,
) -> None:
    secret = "xoxb-secret-value"
    with pytest.raises((ValueError, IntegrationConfigurationError)) as exc_info:
        _factory().create_integration(
            **_kwargs(
                credential=credential.replace("xoxb-test", secret),
            )
        )

    assert message in str(exc_info.value)
    assert secret not in str(exc_info.value)


@pytest.mark.unit
def test_factory_rejects_provider_and_secret_free_config_mismatch() -> None:
    with pytest.raises(ValueError, match="provider_id"):
        _factory().create_integration(**_kwargs(provider_id="other"))

    with pytest.raises(ValueError, match="unsupported fields"):
        _factory().create_integration(
            **_kwargs(secret_free_config={"bot_token": "xoxb-secret"})
        )


class _RecordingSecretsStore:
    def __init__(self, secret: str | None) -> None:
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


def _connection() -> TenantConnection:
    now = datetime(2026, 8, 8, tzinfo=UTC)
    return TenantConnection(
        connection_ref="slack-connection",
        tenant_id="tenant-1",
        provider_id="slack",
        integration_kind=IntegrationCategory.CONVERSATION_CHANNEL,
        safe_display_name="Slack production",
        administrative_status=TenantConnectionAdministrativeStatus.ACTIVE,
        credential_ref="secrets/tenant-1/slack",
        validated_secret_free_config={"api_timeout_seconds": 15.0},
        configuration_version=1,
        created_at=now,
        updated_at=now,
    )


@pytest.mark.integration
def test_restart_rehydrates_slack_from_secret_ref_without_manual_registration() -> None:
    credential = json.dumps(
        {
            "app_token": "xapp-restart",
            "bot_token": "xoxb-restart",
        }
    )
    store = ConditionalInMemoryDocumentStore()
    repository = DocumentStoreTenantConnectionRepository(store)
    repository.create(_connection())
    registry = KnowledgeConnectionRegistry()
    secrets = _RecordingSecretsStore(credential)

    results = TenantConnectionRehydrator(
        repository=DocumentStoreTenantConnectionRepository(store),
        secrets_store=secrets,
        integration_factory=_factory(),
        connection_registry=registry,
    ).rehydrate_tenant(tenant_id="tenant-1")

    assert results[0].status is TenantConnectionRehydrationStatus.REGISTERED
    assert secrets.calls == ["secrets/tenant-1/slack"]
    integration = registry.resolve(
        tenant_id="tenant-1",
        connection_ref="slack-connection",
        provider_id="slack",
        integration_kind=IntegrationCategory.CONVERSATION_CHANNEL,
    )
    assert isinstance(integration, SlackConversationChannelIntegration)
    assert integration.config.require_runtime_tokens() == (
        "xapp-restart",
        "xoxb-restart",
    )
    persisted = store.get(
        "vendor_knowledge_connections:tenant-1",
        "connection:slack-connection",
    )
    assert persisted is not None
    assert "xapp-restart" not in str(persisted.data)
    assert "xoxb-restart" not in str(persisted.data)


@pytest.mark.integration
def test_missing_or_invalid_slack_credential_fails_closed() -> None:
    store = ConditionalInMemoryDocumentStore()
    repository = DocumentStoreTenantConnectionRepository(store)
    repository.create(_connection())

    for secret in (None, "not-json"):
        registry = KnowledgeConnectionRegistry()
        results = TenantConnectionRehydrator(
            repository=DocumentStoreTenantConnectionRepository(store),
            secrets_store=_RecordingSecretsStore(secret),
            integration_factory=_factory(),
            connection_registry=registry,
        ).rehydrate_tenant(tenant_id="tenant-1")
        assert results[0].status is TenantConnectionRehydrationStatus.UNAVAILABLE
        assert "not-json" not in str(results[0])
        with pytest.raises(VendorKnowledgeError):
            registry.resolve(
                tenant_id="tenant-1",
                connection_ref="slack-connection",
                provider_id="slack",
                integration_kind=IntegrationCategory.CONVERSATION_CHANNEL,
            )


@pytest.mark.integration
def test_slack_source_kind_catalog_is_provider_neutral() -> None:
    store = ConditionalInMemoryDocumentStore()
    repository = DocumentStoreTenantConnectionRepository(store)
    repository.create(_connection())
    catalog = TenantVendorKnowledgeSourceCatalog(
        connection_port=RepositoryTenantConnectionPort(repository),
        plugin_registry=build_default_vendor_knowledge_source_plugin_registry(),
    )

    capabilities = catalog.list_source_kind_capabilities(
        tenant_id="tenant-1",
        connection_ref="slack-connection",
    )

    assert len(capabilities) == 1
    assert capabilities[0].identity.source_kind == "slack_conversation"
    assert capabilities[0].modes == (
        VendorKnowledgeMode.DURABLE,
        VendorKnowledgeMode.INDEXED,
        VendorKnowledgeMode.LIVE,
    )
