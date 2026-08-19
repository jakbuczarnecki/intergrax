# © Artur Czarnecki. All rights reserved.

"""Default provider composition roots for Vendor Knowledge connections."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from intergrax.integrations.contracts.base import IntegrationCategory, IntegrationDependencyError
from intergrax.integrations.providers.collaboration_suite.google_workspace.contracts import (
    GoogleWorkspaceClientFactory,
)
from intergrax.integrations.providers.collaboration_suite.google_workspace.integration import (
    GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID,
)
from intergrax.integrations.providers.collaboration_suite.google_workspace.tenant_connection_factory import (
    GoogleWorkspaceTenantConnectionIntegrationFactory,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.integration import (
    MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.tenant_connection_factory import (
    Ms365GraphRuntimeBuilder,
    Ms365GraphTenantConnectionIntegrationFactory,
)
from intergrax.integrations.providers.conversation_channel.slack.integration import (
    SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
    SlackConversationChannelIntegration,
)
from intergrax.integrations.providers.conversation_channel.slack.tenant_connection_factory import (
    SlackRuntimeBuilder,
    SlackTenantConnectionIntegrationFactory,
)
from intergrax.integrations.providers.issue_tracker.jira.config import JiraIntegrationConfig
from intergrax.integrations.providers.issue_tracker.jira.integration import (
    JIRA_ISSUE_TRACKER_PROVIDER_ID,
)
from intergrax.integrations.providers.issue_tracker.jira.tenant_connection_factory import (
    JiraTenantConnectionIntegrationFactory,
)
from intergrax.integrations.providers.project_status.config import ProjectStatusIntegrationConfig
from intergrax.integrations.providers.project_status.knowledge_read import (
    PROJECT_STATUS_PROVIDER_ID,
)
from intergrax.integrations.providers.project_status.tenant_connection_factory import (
    ProjectStatusTenantConnectionIntegrationFactory,
)
from intergrax.integrations.providers.relational_store.databricks.integration import (
    DATABRICKS_RELATIONAL_STORE_PROVIDER_ID,
)
from intergrax.integrations.providers.relational_store.databricks.tenant_connection_factory import (
    DatabricksTenantConnectionIntegrationFactory,
)
from intergrax.integrations.providers.wiki_knowledge.confluence.config import (
    ConfluenceIntegrationConfig,
)
from intergrax.integrations.providers.wiki_knowledge.confluence.integration import (
    CONFLUENCE_WIKI_KNOWLEDGE_PROVIDER_ID,
)
from intergrax.integrations.providers.wiki_knowledge.confluence.tenant_connection_factory import (
    ConfluenceTenantConnectionIntegrationFactory,
)
from intergrax.runtime.vendor_knowledge.tenant_connection_factory_registry import (
    TenantConnectionIntegrationFactoryRegistry,
)
from intergrax.runtime.vendor_knowledge.contribution import (
    VendorKnowledgeConnectionFactoryContribution,
)
from intergrax.runtime.vendor_knowledge.contribution_catalog import (
    build_default_vendor_knowledge_contribution_catalog,
    build_vendor_knowledge_connection_factory_registry,
)


@dataclass(frozen=True, slots=True)
class VendorKnowledgeLegacyLocalBootstrap:
    """Provider-neutral contract for the bounded local-development fallback."""

    provider_id: str
    integration_kind: IntegrationCategory
    integration: object


class _UnavailableGoogleWorkspaceClientFactory:
    """Keep the canonical route present until an auth executor is composed."""

    def create_client_family(self, *, credential_material: object) -> object:
        raise IntegrationDependencyError(
            "Google Workspace client family is unavailable in this composition",
        )


def build_default_vendor_knowledge_connection_factory_registry(
    *,
    slack_runtime_builder: SlackRuntimeBuilder | None = None,
    msgraph_runtime_builder: Ms365GraphRuntimeBuilder | None = None,
    google_client_factory: GoogleWorkspaceClientFactory | None = None,
    jira_http_client_factory: Callable[[JiraIntegrationConfig], Any] | None = None,
    confluence_http_client_factory: Callable[[ConfluenceIntegrationConfig], Any] | None = None,
    project_status_http_client_factory: Callable[[ProjectStatusIntegrationConfig], Any] | None = None,
    databricks_connection_factory: Callable[[], Any] | None = None,
    discover_entry_points: bool = False,
) -> TenantConnectionIntegrationFactoryRegistry:
    """Compose provider-owned factories through the contribution catalog."""
    overrides = {
        (
            MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
            IntegrationCategory.COLLABORATION_SUITE,
        ): VendorKnowledgeConnectionFactoryContribution(
            provider_id=MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
            integration_category=IntegrationCategory.COLLABORATION_SUITE,
            factory=Ms365GraphTenantConnectionIntegrationFactory(
                runtime_builder=msgraph_runtime_builder,
            ),
        ),
        (
            SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
            IntegrationCategory.CONVERSATION_CHANNEL,
        ): VendorKnowledgeConnectionFactoryContribution(
            provider_id=SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
            integration_category=IntegrationCategory.CONVERSATION_CHANNEL,
            factory=SlackTenantConnectionIntegrationFactory(
                runtime_builder=slack_runtime_builder,
            ),
        ),
        (
            GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID,
            IntegrationCategory.COLLABORATION_SUITE,
        ): VendorKnowledgeConnectionFactoryContribution(
            provider_id=GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID,
            integration_category=IntegrationCategory.COLLABORATION_SUITE,
            factory=GoogleWorkspaceTenantConnectionIntegrationFactory(
                client_factory=google_client_factory
                or _UnavailableGoogleWorkspaceClientFactory(),
            ),
        ),
        (
            JIRA_ISSUE_TRACKER_PROVIDER_ID,
            IntegrationCategory.ISSUE_TRACKER,
        ): VendorKnowledgeConnectionFactoryContribution(
            provider_id=JIRA_ISSUE_TRACKER_PROVIDER_ID,
            integration_category=IntegrationCategory.ISSUE_TRACKER,
            factory=JiraTenantConnectionIntegrationFactory(
                http_client_factory=jira_http_client_factory,
            ),
        ),
        (
            CONFLUENCE_WIKI_KNOWLEDGE_PROVIDER_ID,
            IntegrationCategory.WIKI_KNOWLEDGE,
        ): VendorKnowledgeConnectionFactoryContribution(
            provider_id=CONFLUENCE_WIKI_KNOWLEDGE_PROVIDER_ID,
            integration_category=IntegrationCategory.WIKI_KNOWLEDGE,
            factory=ConfluenceTenantConnectionIntegrationFactory(
                http_client_factory=confluence_http_client_factory,
            ),
        ),
        (
            PROJECT_STATUS_PROVIDER_ID,
            IntegrationCategory.ISSUE_TRACKER,
        ): VendorKnowledgeConnectionFactoryContribution(
            provider_id=PROJECT_STATUS_PROVIDER_ID,
            integration_category=IntegrationCategory.ISSUE_TRACKER,
            factory=ProjectStatusTenantConnectionIntegrationFactory(
                http_client_factory=project_status_http_client_factory,
            ),
        ),
        (
            DATABRICKS_RELATIONAL_STORE_PROVIDER_ID,
            IntegrationCategory.RELATIONAL_STORE,
        ): VendorKnowledgeConnectionFactoryContribution(
            provider_id=DATABRICKS_RELATIONAL_STORE_PROVIDER_ID,
            integration_category=IntegrationCategory.RELATIONAL_STORE,
            factory=DatabricksTenantConnectionIntegrationFactory(
                connection_factory=databricks_connection_factory,
            ),
        ),
    }
    catalog = build_default_vendor_knowledge_contribution_catalog(
        discover_entry_points=discover_entry_points,
    )
    return build_vendor_knowledge_connection_factory_registry(
        catalog.with_connection_factory_overrides(overrides)
    )


def build_default_vendor_knowledge_legacy_local_bootstrap(
    *,
    slack_runtime_builder: SlackRuntimeBuilder | None = None,
) -> VendorKnowledgeLegacyLocalBootstrap | None:
    """Build the legacy local fallback behind the Vendor Knowledge boundary.

    This fallback is intentionally separate from durable tenant connection
    lifecycle. Production startup must use the generic factory registry and
    persisted connection identity instead.
    """
    from intergrax.integrations.providers.conversation_channel.slack.config import (
        SlackConversationChannelIntegrationConfig,
    )

    try:
        config = SlackConversationChannelIntegrationConfig.from_env(enabled=True)
        config.validate_for_runtime()
        integration = (slack_runtime_builder or SlackConversationChannelIntegration.from_config)(
            config
        )
    except Exception:
        return None
    return VendorKnowledgeLegacyLocalBootstrap(
        provider_id=SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
        integration_kind=IntegrationCategory.CONVERSATION_CHANNEL,
        integration=integration,
    )


__all__ = [
    "build_default_vendor_knowledge_connection_factory_registry",
    "build_default_vendor_knowledge_legacy_local_bootstrap",
    "VendorKnowledgeLegacyLocalBootstrap",
]
