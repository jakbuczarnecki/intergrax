from __future__ import annotations

import inspect

from applications.local_workspace_application.workspaces import connected_source_wiring
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.collaboration_suite.google_workspace.integration import (
    GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID,
)
from intergrax.integrations.providers.collaboration_suite.google_workspace.knowledge_read.calendar import (
    GOOGLE_CALENDAR_SOURCE_KIND,
)
from intergrax.integrations.providers.collaboration_suite.google_workspace.knowledge_read.docs import (
    GOOGLE_DOCS_SOURCE_KIND,
)
from intergrax.integrations.providers.collaboration_suite.google_workspace.knowledge_read.drive import (
    GOOGLE_DRIVE_SOURCE_KIND,
)
from intergrax.integrations.providers.collaboration_suite.google_workspace.knowledge_read.sheets import (
    GOOGLE_SHEETS_SOURCE_KIND,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.integration import (
    MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.calendar_inventory import (
    MSGRAPH_CALENDAR_SOURCE_KIND,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.drive import (
    MSGRAPH_DRIVE_SOURCE_KIND,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.mail_folders import (
    MSGRAPH_MAIL_SOURCE_KIND,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.teams_channel_inventory import (
    MSGRAPH_TEAMS_CHANNEL_SOURCE_KIND,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read.teams_chat_inventory import (
    MSGRAPH_TEAMS_CHAT_SOURCE_KIND,
)
from intergrax.integrations.providers.conversation_channel.slack.integration import (
    SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
)
from intergrax.integrations.providers.conversation_channel.slack.knowledge_read.common import (
    SLACK_CONVERSATION_SOURCE_KIND,
)
from intergrax.integrations.providers.issue_tracker.jira.integration import (
    JIRA_ISSUE_TRACKER_PROVIDER_ID,
)
from intergrax.integrations.providers.issue_tracker.jira.knowledge_read import (
    JIRA_ISSUES_SOURCE_KIND,
)
from intergrax.integrations.providers.wiki_knowledge.confluence.integration import (
    CONFLUENCE_WIKI_KNOWLEDGE_PROVIDER_ID,
)
from intergrax.integrations.providers.wiki_knowledge.confluence.knowledge_read import (
    CONFLUENCE_PAGES_SOURCE_KIND,
)
from intergrax.runtime.vendor_knowledge.adapter_composition import (
    build_default_vendor_knowledge_adapter_registry,
)
from intergrax.runtime.vendor_knowledge.live.bootstrap import (
    build_vendor_knowledge_live_registration_registry,
)
from intergrax.runtime.vendor_knowledge.plugin import (
    VendorKnowledgeMode,
    VendorKnowledgeSourceIdentity,
)
from intergrax.runtime.vendor_knowledge.plugin_composition import (
    build_default_vendor_knowledge_source_plugin_registry,
)
from intergrax.runtime.vendor_knowledge.provider_composition import (
    build_default_vendor_knowledge_connection_factory_registry,
)


def test_default_plugin_catalog_matches_all_implemented_source_adapters() -> None:
    plugins = build_default_vendor_knowledge_source_plugin_registry()
    identities = {
        plugin.identity.key
        for plugin in plugins.list_plugins()
    }
    expected = {
        (SLACK_CONVERSATION_CHANNEL_PROVIDER_ID, IntegrationCategory.CONVERSATION_CHANNEL, SLACK_CONVERSATION_SOURCE_KIND),
        *{
            (
                MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
                IntegrationCategory.COLLABORATION_SUITE,
                source_kind,
            )
            for source_kind in (
                MSGRAPH_DRIVE_SOURCE_KIND,
                MSGRAPH_MAIL_SOURCE_KIND,
                MSGRAPH_TEAMS_CHANNEL_SOURCE_KIND,
                MSGRAPH_TEAMS_CHAT_SOURCE_KIND,
                MSGRAPH_CALENDAR_SOURCE_KIND,
            )
        },
        *{
            (
                GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID,
                IntegrationCategory.COLLABORATION_SUITE,
                source_kind,
            )
            for source_kind in (
                GOOGLE_DRIVE_SOURCE_KIND,
                GOOGLE_DOCS_SOURCE_KIND,
                GOOGLE_SHEETS_SOURCE_KIND,
                GOOGLE_CALENDAR_SOURCE_KIND,
            )
        },
        (JIRA_ISSUE_TRACKER_PROVIDER_ID, IntegrationCategory.ISSUE_TRACKER, JIRA_ISSUES_SOURCE_KIND),
        (CONFLUENCE_WIKI_KNOWLEDGE_PROVIDER_ID, IntegrationCategory.WIKI_KNOWLEDGE, CONFLUENCE_PAGES_SOURCE_KIND),
    }
    assert identities == expected
    assert len(identities) == 12


def test_durable_plugin_declarations_have_registered_adapter_runtime() -> None:
    adapters = build_default_vendor_knowledge_adapter_registry()
    assert adapters.registered_keys() == (
        (
            MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
            IntegrationCategory.COLLABORATION_SUITE,
            MSGRAPH_DRIVE_SOURCE_KIND,
        ),
        (
            MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
            IntegrationCategory.COLLABORATION_SUITE,
            MSGRAPH_MAIL_SOURCE_KIND,
        ),
        (
            MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
            IntegrationCategory.COLLABORATION_SUITE,
            MSGRAPH_TEAMS_CHANNEL_SOURCE_KIND,
        ),
        (
            MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
            IntegrationCategory.COLLABORATION_SUITE,
            MSGRAPH_TEAMS_CHAT_SOURCE_KIND,
        ),
        (
            MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
            IntegrationCategory.COLLABORATION_SUITE,
            MSGRAPH_CALENDAR_SOURCE_KIND,
        ),
        (
            SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
            IntegrationCategory.CONVERSATION_CHANNEL,
            SLACK_CONVERSATION_SOURCE_KIND,
        ),
        (
            GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID,
            IntegrationCategory.COLLABORATION_SUITE,
            GOOGLE_DRIVE_SOURCE_KIND,
        ),
        (
            GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID,
            IntegrationCategory.COLLABORATION_SUITE,
            GOOGLE_DOCS_SOURCE_KIND,
        ),
        (
            GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID,
            IntegrationCategory.COLLABORATION_SUITE,
            GOOGLE_SHEETS_SOURCE_KIND,
        ),
        (
            GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID,
            IntegrationCategory.COLLABORATION_SUITE,
            GOOGLE_CALENDAR_SOURCE_KIND,
        ),
        (
            JIRA_ISSUE_TRACKER_PROVIDER_ID,
            IntegrationCategory.ISSUE_TRACKER,
            JIRA_ISSUES_SOURCE_KIND,
        ),
        (
            CONFLUENCE_WIKI_KNOWLEDGE_PROVIDER_ID,
            IntegrationCategory.WIKI_KNOWLEDGE,
            CONFLUENCE_PAGES_SOURCE_KIND,
        ),
    )

    plugins = build_default_vendor_knowledge_source_plugin_registry()
    for plugin in plugins.list_plugins():
        durable = plugin.capability(VendorKnowledgeMode.DURABLE)
        assert durable is not None
        expected_runtime_ref = (
            f"knowledge-adapter:{plugin.identity.provider_id}:"
            f"{plugin.identity.integration_category.value}:{plugin.identity.source_kind}"
        )
        assert durable.runtime_ref == expected_runtime_ref
        assert plugin.identity.key in adapters.registered_keys()


def test_lkw_wiring_delegates_adapter_composition_to_vendor_knowledge() -> None:
    source = inspect.getsource(connected_source_wiring)

    assert "build_default_vendor_knowledge_adapter_registry" in source
    assert not any(
        "register_" in line and "knowledge_adapter" in line
        for line in source.splitlines()
    )


def test_indexed_declarations_resolve_only_existing_materializers() -> None:
    plugins = build_default_vendor_knowledge_source_plugin_registry()
    indexed = {
        plugin.identity.key: plugin.capability(VendorKnowledgeMode.INDEXED).runtime_ref
        for plugin in plugins.list_plugins()
        if plugin.supports(VendorKnowledgeMode.INDEXED)
    }
    assert indexed == {
        (
            SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
            IntegrationCategory.CONVERSATION_CHANNEL,
            SLACK_CONVERSATION_SOURCE_KIND,
        ): "indexed-source:slack:slack_conversation",
        (
            MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
            IntegrationCategory.COLLABORATION_SUITE,
            MSGRAPH_TEAMS_CHAT_SOURCE_KIND,
        ): "indexed-source:ms365_graph:teams_chat",
    }


def test_graph_and_slack_live_plugins_are_registered_by_generic_bootstrap() -> None:
    registry = build_vendor_knowledge_live_registration_registry()
    for source_kind in (
        MSGRAPH_DRIVE_SOURCE_KIND,
        MSGRAPH_MAIL_SOURCE_KIND,
        MSGRAPH_TEAMS_CHANNEL_SOURCE_KIND,
        MSGRAPH_TEAMS_CHAT_SOURCE_KIND,
        MSGRAPH_CALENDAR_SOURCE_KIND,
    ):
        assert registry.resolve_for_source(
            VendorKnowledgeSourceIdentity(
                provider_id=MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
                integration_category=IntegrationCategory.COLLABORATION_SUITE,
                source_kind=source_kind,
            )
        )
    assert registry.resolve_for_source(
        VendorKnowledgeSourceIdentity(
            provider_id=SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
            integration_category=IntegrationCategory.CONVERSATION_CHANNEL,
            source_kind=SLACK_CONVERSATION_SOURCE_KIND,
        )
    )


def test_default_connection_factory_registry_routes_msgraph_by_canonical_identity() -> None:
    captured = {}

    def runtime_builder(config):
        captured["config"] = config
        return config

    registry = build_default_vendor_knowledge_connection_factory_registry(
        msgraph_runtime_builder=runtime_builder,
    )
    integration = registry.create_integration(
        tenant_id="tenant-1",
        connection_ref="connection-1",
        provider_id=MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
        integration_kind=IntegrationCategory.COLLABORATION_SUITE,
        credential_ref="secret-1",
        credential="client-secret",
        secret_free_config={"client_id": "client-id"},
    )

    assert integration is captured["config"]
    assert integration.tenant_id == "tenant-1"
    assert integration.client_id == "client-id"
    assert integration.client_secret == "client-secret"
