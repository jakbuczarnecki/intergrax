# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Restart-safe connection proof for the independent Atlassian providers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pytest

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.issue_tracker.jira.integration import (
    JIRA_ISSUE_TRACKER_PROVIDER_ID,
    JiraIssueTrackerIntegration,
)
from intergrax.integrations.providers.wiki_knowledge.confluence.integration import (
    CONFLUENCE_WIKI_KNOWLEDGE_PROVIDER_ID,
    ConfluenceWikiKnowledgeIntegration,
)
from intergrax.runtime.vendor_knowledge.connections import KnowledgeConnectionRegistry
from intergrax.runtime.vendor_knowledge.errors import VendorKnowledgeError
from intergrax.runtime.vendor_knowledge.provider_composition import (
    build_default_vendor_knowledge_connection_factory_registry,
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
    TenantConnectionService,
)
from tests.unit.runtime.vendor_knowledge.test_tenant_connection_document_store import (
    ConditionalInMemoryDocumentStore,
)

pytestmark = pytest.mark.unit

_JIRA_TOKEN = "jira-token"
_CONFLUENCE_TOKEN = "confluence-token"


@dataclass
class _RecordingSecretsStore:
    secrets: dict[str, str]
    calls: list[str]

    def get_secret(self, path: str, *, version: str | None = None) -> str:
        self.calls.append(path)
        return self.secrets[path]

    def put_secret(self, path: str, value: str) -> None:
        self.secrets[path] = value

    def delete_secret(self, path: str) -> None:
        self.secrets.pop(path, None)


class _NoopHttpClient:
    pass


def _connection(
    *,
    connection_ref: str,
    provider_id: str,
    integration_kind: IntegrationCategory,
    credential_ref: str,
    base_url: str,
) -> TenantConnection:
    stamp = datetime(2026, 8, 9, 15, 0, tzinfo=timezone.utc)
    return TenantConnection(
        connection_ref=connection_ref,
        tenant_id="tenant-1",
        provider_id=provider_id,
        integration_kind=integration_kind,
        safe_display_name=provider_id.title(),
        administrative_status=TenantConnectionAdministrativeStatus.ACTIVE,
        credential_ref=credential_ref,
        validated_secret_free_config={"base_url": base_url},
        configuration_version=1,
        created_at=stamp,
        updated_at=stamp,
    )


@pytest.mark.parametrize(
    ("provider_id", "integration_kind", "credential_ref", "base_url"),
    [
        (
            JIRA_ISSUE_TRACKER_PROVIDER_ID,
            IntegrationCategory.ISSUE_TRACKER,
            "cred-jira",
            "https://example.atlassian.net",
        ),
        (
            CONFLUENCE_WIKI_KNOWLEDGE_PROVIDER_ID,
            IntegrationCategory.WIKI_KNOWLEDGE,
            "cred-confluence",
            "https://example.atlassian.net/wiki",
        ),
    ],
)
def test_atlassian_connections_rehydrate_independently(
    provider_id: str,
    integration_kind: IntegrationCategory,
    credential_ref: str,
    base_url: str,
) -> None:
    store = ConditionalInMemoryDocumentStore()
    repository = DocumentStoreTenantConnectionRepository(store)
    service = TenantConnectionService(tenant_id="tenant-1", repository=repository)
    service.create(
        _connection(
            connection_ref=f"{provider_id}-connection",
            provider_id=provider_id,
            integration_kind=integration_kind,
            credential_ref=credential_ref,
            base_url=base_url,
        )
    )

    runtime_configs: list[dict[str, Any]] = []

    def http_client_factory(config: Any) -> _NoopHttpClient:
        runtime_configs.append(
            {
                "base_url": config.base_url,
                "email": config.email,
                "api_token": config.api_token,
            }
        )
        return _NoopHttpClient()

    token = _JIRA_TOKEN if provider_id == JIRA_ISSUE_TRACKER_PROVIDER_ID else _CONFLUENCE_TOKEN
    secrets = _RecordingSecretsStore(
        secrets={
            credential_ref: json.dumps(
                {"email": "bot@example.com", "api_token": token}
            )
        },
        calls=[],
    )
    connection_registry = KnowledgeConnectionRegistry()
    factory_registry = build_default_vendor_knowledge_connection_factory_registry(
        jira_http_client_factory=http_client_factory,
        confluence_http_client_factory=http_client_factory,
    )

    results = TenantConnectionRehydrator(
        repository=DocumentStoreTenantConnectionRepository(store),
        secrets_store=secrets,
        integration_factory=factory_registry,
        connection_registry=connection_registry,
    ).rehydrate_tenant(tenant_id="tenant-1")

    assert results[0].status is TenantConnectionRehydrationStatus.REGISTERED
    assert secrets.calls == [credential_ref]
    assert credential_ref not in results[0].model_dump()["connection"]
    document = store.get(
        "vendor_knowledge_connections:tenant-1",
        f"connection:{provider_id}-connection",
    )
    assert document is not None
    assert token not in str(document.data)
    assert runtime_configs == [
        {
            "base_url": base_url,
            "email": "bot@example.com",
            "api_token": token,
        }
    ]

    integration = connection_registry.resolve(
        tenant_id="tenant-1",
        connection_ref=f"{provider_id}-connection",
        provider_id=provider_id,
        integration_kind=integration_kind,
    )
    if provider_id == JIRA_ISSUE_TRACKER_PROVIDER_ID:
        assert isinstance(integration, JiraIssueTrackerIntegration)
        assert integration.client is not None
        assert integration.client.rest_client.config.base_url == base_url
    else:
        assert isinstance(integration, ConfluenceWikiKnowledgeIntegration)
        assert integration.client is not None
        assert integration.client.rest_client.config.base_url == base_url

    with pytest.raises(VendorKnowledgeError):
        connection_registry.resolve(
            tenant_id="tenant-1",
            connection_ref=f"{provider_id}-connection",
            provider_id=(
                CONFLUENCE_WIKI_KNOWLEDGE_PROVIDER_ID
                if provider_id == JIRA_ISSUE_TRACKER_PROVIDER_ID
                else JIRA_ISSUE_TRACKER_PROVIDER_ID
            ),
            integration_kind=(
                IntegrationCategory.WIKI_KNOWLEDGE
                if integration_kind is IntegrationCategory.ISSUE_TRACKER
                else IntegrationCategory.ISSUE_TRACKER
            ),
        )
