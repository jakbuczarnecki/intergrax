from __future__ import annotations

import inspect
from dataclasses import FrozenInstanceError

import pytest

from applications.local_workspace_application.workspaces.connected_source_materializer import (
    ConfluencePageRichTextMaterializer,
    GoogleCalendarStructuredRecordMaterializer,
    GoogleDocsStructuredRecordMaterializer,
    GoogleSheetsStructuredRecordMaterializer,
    JiraIssueStructuredRecordMaterializer,
    MsGraphCalendarStructuredRecordMaterializer,
    MsGraphMailStructuredRecordMaterializer,
    MsGraphTeamsChannelStructuredRecordMaterializer,
    MsGraphTeamsChatStructuredRecordMaterializer,
    SlackConversationStructuredRecordMaterializer,
)
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
from intergrax.integrations.providers.relational_store.databricks.integration import (
    DATABRICKS_RELATIONAL_STORE_PROVIDER_ID,
)
from intergrax.integrations.providers.wiki_knowledge.confluence.integration import (
    CONFLUENCE_WIKI_KNOWLEDGE_PROVIDER_ID,
)
from intergrax.integrations.providers.wiki_knowledge.confluence.knowledge_read import (
    CONFLUENCE_PAGES_SOURCE_KIND,
)
from intergrax.runtime.vendor_knowledge.adapters.confluence_pages import (
    ConfluencePagesKnowledgeAdapter,
)
from intergrax.runtime.vendor_knowledge.adapters.google_workspace_calendar import (
    GoogleWorkspaceCalendarKnowledgeAdapter,
)
from intergrax.runtime.vendor_knowledge.adapters.google_workspace_docs import (
    GoogleWorkspaceDocsKnowledgeAdapter,
)
from intergrax.runtime.vendor_knowledge.adapters.google_workspace_drive import (
    GoogleWorkspaceDriveKnowledgeAdapter,
)
from intergrax.runtime.vendor_knowledge.adapters.google_workspace_sheets import (
    GoogleWorkspaceSheetsKnowledgeAdapter,
)
from intergrax.runtime.vendor_knowledge.adapters.jira_issues import (
    JiraIssuesKnowledgeAdapter,
)
from intergrax.runtime.vendor_knowledge.adapters.ms365_graph_calendar import (
    MsGraphCalendarKnowledgeAdapter,
)
from intergrax.runtime.vendor_knowledge.adapters.ms365_graph_drive import (
    MsGraphDriveKnowledgeAdapter,
)
from intergrax.runtime.vendor_knowledge.adapters.ms365_graph_mail import (
    MsGraphMailKnowledgeAdapter,
)
from intergrax.runtime.vendor_knowledge.adapters.ms365_graph_teams_channel import (
    MsGraphTeamsChannelKnowledgeAdapter,
)
from intergrax.runtime.vendor_knowledge.adapters.ms365_graph_teams_chat import (
    MsGraphTeamsChatKnowledgeAdapter,
)
from intergrax.runtime.vendor_knowledge.adapters.slack_conversation import (
    SlackConversationKnowledgeAdapter,
)
from intergrax.runtime.vendor_knowledge.contribution import (
    APPLICATION_OWNED_EXTENSION_SURFACE,
    VendorKnowledgeConnectionFactoryContribution,
    VendorKnowledgeContributionError,
    VendorKnowledgeDiscoveryContribution,
    VendorKnowledgeIndexedMaterializerContribution,
    VendorKnowledgeProviderContribution,
)
from intergrax.runtime.vendor_knowledge.live.bootstrap import (
    build_vendor_knowledge_live_registration_registry,
)
from intergrax.runtime.vendor_knowledge.plugin import (
    VendorKnowledgeMode,
    VendorKnowledgeModeCapability,
    VendorKnowledgeSourceIdentity,
    VendorKnowledgeSourcePlugin,
)
from intergrax.runtime.vendor_knowledge.plugin_composition import (
    build_default_vendor_knowledge_source_plugin_registry,
)


class _StubConnectionFactory:
    def create_integration(self, **_: object) -> object:
        return object()


def _empty_discovery_factory() -> object:
    return object()


_ADAPTERS = (
    MsGraphDriveKnowledgeAdapter(),
    MsGraphMailKnowledgeAdapter(),
    MsGraphTeamsChannelKnowledgeAdapter(),
    MsGraphTeamsChatKnowledgeAdapter(),
    MsGraphCalendarKnowledgeAdapter(),
    SlackConversationKnowledgeAdapter(),
    GoogleWorkspaceCalendarKnowledgeAdapter(),
    GoogleWorkspaceDocsKnowledgeAdapter(),
    GoogleWorkspaceSheetsKnowledgeAdapter(),
    GoogleWorkspaceDriveKnowledgeAdapter(),
    JiraIssuesKnowledgeAdapter(),
    ConfluencePagesKnowledgeAdapter(),
)


_MATERIALIZER_FACTORIES = {
    (
        SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
        IntegrationCategory.CONVERSATION_CHANNEL,
        SLACK_CONVERSATION_SOURCE_KIND,
    ): SlackConversationStructuredRecordMaterializer,
    (
        MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
        IntegrationCategory.COLLABORATION_SUITE,
        MSGRAPH_MAIL_SOURCE_KIND,
    ): MsGraphMailStructuredRecordMaterializer,
    (
        MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
        IntegrationCategory.COLLABORATION_SUITE,
        MSGRAPH_TEAMS_CHANNEL_SOURCE_KIND,
    ): MsGraphTeamsChannelStructuredRecordMaterializer,
    (
        MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
        IntegrationCategory.COLLABORATION_SUITE,
        MSGRAPH_TEAMS_CHAT_SOURCE_KIND,
    ): MsGraphTeamsChatStructuredRecordMaterializer,
    (
        MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
        IntegrationCategory.COLLABORATION_SUITE,
        MSGRAPH_CALENDAR_SOURCE_KIND,
    ): MsGraphCalendarStructuredRecordMaterializer,
    (
        GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID,
        IntegrationCategory.COLLABORATION_SUITE,
        GOOGLE_CALENDAR_SOURCE_KIND,
    ): GoogleCalendarStructuredRecordMaterializer,
    (
        GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID,
        IntegrationCategory.COLLABORATION_SUITE,
        GOOGLE_DOCS_SOURCE_KIND,
    ): GoogleDocsStructuredRecordMaterializer,
    (
        GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID,
        IntegrationCategory.COLLABORATION_SUITE,
        GOOGLE_SHEETS_SOURCE_KIND,
    ): GoogleSheetsStructuredRecordMaterializer,
    (
        JIRA_ISSUE_TRACKER_PROVIDER_ID,
        IntegrationCategory.ISSUE_TRACKER,
        JIRA_ISSUES_SOURCE_KIND,
    ): JiraIssueStructuredRecordMaterializer,
    (
        CONFLUENCE_WIKI_KNOWLEDGE_PROVIDER_ID,
        IntegrationCategory.WIKI_KNOWLEDGE,
        CONFLUENCE_PAGES_SOURCE_KIND,
    ): ConfluencePageRichTextMaterializer,
}


def _contribution_for(
    provider_id: str,
    integration_category: IntegrationCategory,
) -> VendorKnowledgeProviderContribution:
    plugins = tuple(
        plugin
        for plugin in build_default_vendor_knowledge_source_plugin_registry().list_plugins()
        if plugin.identity.provider_id == provider_id
        and plugin.identity.integration_category is integration_category
    )
    adapters = tuple(
        adapter
        for adapter in _ADAPTERS
        if adapter.provider_id == provider_id
        and adapter.integration_kind is integration_category
    )
    materializers = tuple(
        VendorKnowledgeIndexedMaterializerContribution(
            identity=plugin.identity,
            runtime_ref=plugin.capability(VendorKnowledgeMode.INDEXED).runtime_ref,
            factory=_MATERIALIZER_FACTORIES[plugin.identity.key],
        )
        for plugin in plugins
        if plugin.supports(VendorKnowledgeMode.INDEXED)
    )
    live_bundles = tuple(
        bundle
        for bundle in build_vendor_knowledge_live_registration_registry().list_registrations()
        if bundle.descriptor.provider_id == provider_id
        and bundle.descriptor.integration_kind is integration_category
    )
    return VendorKnowledgeProviderContribution(
        provider_id=provider_id,
        integration_category=integration_category,
        adapters=adapters,
        source_plugins=plugins,
        indexed_materializers=materializers,
        live_contributions=live_bundles,
        connection_factories=(
            VendorKnowledgeConnectionFactoryContribution(
                provider_id=provider_id,
                integration_category=integration_category,
                factory=_StubConnectionFactory(),
            ),
        ),
    )


def test_current_provider_families_represent_all_twelve_source_tuples() -> None:
    contributions = (
        _contribution_for(
            SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
            IntegrationCategory.CONVERSATION_CHANNEL,
        ),
        _contribution_for(
            MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
            IntegrationCategory.COLLABORATION_SUITE,
        ),
        _contribution_for(
            GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID,
            IntegrationCategory.COLLABORATION_SUITE,
        ),
        _contribution_for(
            JIRA_ISSUE_TRACKER_PROVIDER_ID,
            IntegrationCategory.ISSUE_TRACKER,
        ),
        _contribution_for(
            CONFLUENCE_WIKI_KNOWLEDGE_PROVIDER_ID,
            IntegrationCategory.WIKI_KNOWLEDGE,
        ),
    )
    identities = tuple(
        identity
        for contribution in contributions
        for identity in contribution.source_identities
    )

    assert len(identities) == 12
    assert len({identity.key for identity in identities}) == 12
    assert sum(len(contribution.adapters) for contribution in contributions) == 12
    assert (
        sum(len(contribution.indexed_materializers) for contribution in contributions)
        == 10
    )
    assert sum(len(contribution.live_contributions) for contribution in contributions) == sum(
        len(plugin.capability(VendorKnowledgeMode.LIVE).capability_refs)
        for contribution in contributions
        for plugin in contribution.source_plugins
        if plugin.supports(VendorKnowledgeMode.LIVE)
    )
    assert all(
        plugin.supports(VendorKnowledgeMode.DURABLE)
        for contribution in contributions
        for plugin in contribution.source_plugins
    )
    assert not any(
        plugin.supports(VendorKnowledgeMode.INDEXED)
        and plugin.identity.key not in _MATERIALIZER_FACTORIES
        for contribution in contributions
        for plugin in contribution.source_plugins
    )


def test_contribution_is_immutable_and_deterministically_ordered() -> None:
    contribution = _contribution_for(
        JIRA_ISSUE_TRACKER_PROVIDER_ID,
        IntegrationCategory.ISSUE_TRACKER,
    )

    assert isinstance(contribution.adapters, tuple)
    assert isinstance(contribution.source_plugins, tuple)
    assert contribution.contract_version == "vendor-knowledge.provider-contribution.v1"
    assert contribution.source_identities == tuple(
        sorted(
            contribution.source_identities,
            key=lambda identity: (
                identity.provider_id,
                identity.integration_category.value,
                identity.source_kind,
            ),
        )
    )
    with pytest.raises(FrozenInstanceError):
        contribution.provider_id = "changed"  # type: ignore[misc]


def test_connection_only_provider_contribution_represents_databricks() -> None:
    contribution = VendorKnowledgeProviderContribution(
        provider_id=DATABRICKS_RELATIONAL_STORE_PROVIDER_ID,
        integration_category=IntegrationCategory.RELATIONAL_STORE,
        connection_factories=(
            VendorKnowledgeConnectionFactoryContribution(
                provider_id=DATABRICKS_RELATIONAL_STORE_PROVIDER_ID,
                integration_category=IntegrationCategory.RELATIONAL_STORE,
                factory=_StubConnectionFactory(),
            ),
        ),
    )

    assert contribution.source_identities == ()
    assert contribution.provider_key == (
        DATABRICKS_RELATIONAL_STORE_PROVIDER_ID,
        IntegrationCategory.RELATIONAL_STORE,
    )


def test_application_owned_discovery_hook_has_no_application_import_in_abi() -> None:
    source = inspect.getsource(__import__("intergrax.runtime.vendor_knowledge.contribution", fromlist=["*"]))
    discovery = VendorKnowledgeDiscoveryContribution(
        identity=VendorKnowledgeSourceIdentity(
            provider_id=JIRA_ISSUE_TRACKER_PROVIDER_ID,
            integration_category=IntegrationCategory.ISSUE_TRACKER,
            source_kind=JIRA_ISSUES_SOURCE_KIND,
        ),
        factory=_empty_discovery_factory,
    )
    contribution = VendorKnowledgeProviderContribution(
        provider_id=JIRA_ISSUE_TRACKER_PROVIDER_ID,
        integration_category=IntegrationCategory.ISSUE_TRACKER,
        source_plugins=(
            build_default_vendor_knowledge_source_plugin_registry().require(
                discovery.identity
            ),
        ),
        discovery_contributions=(discovery,),
    )

    assert APPLICATION_OWNED_EXTENSION_SURFACE in inspect.getdoc(
        VendorKnowledgeDiscoveryContribution
    )
    assert "applications.local_workspace_application" not in source
    assert "local_workspace_application" not in source
    assert contribution.discovery_contributions == (discovery,)


def test_provider_and_category_identity_mismatch_fails_closed() -> None:
    with pytest.raises(
        VendorKnowledgeContributionError,
        match="source_plugin_identity_mismatch",
    ):
        VendorKnowledgeProviderContribution(
            provider_id=JIRA_ISSUE_TRACKER_PROVIDER_ID,
            integration_category=IntegrationCategory.ISSUE_TRACKER,
            source_plugins=(
                build_default_vendor_knowledge_source_plugin_registry().require(
                    VendorKnowledgeSourceIdentity(
                        provider_id=SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
                        integration_category=IntegrationCategory.CONVERSATION_CHANNEL,
                        source_kind=SLACK_CONVERSATION_SOURCE_KIND,
                    )
                ),
            ),
        )


def test_source_component_mismatch_and_duplicates_fail_closed() -> None:
    jira_plugin = build_default_vendor_knowledge_source_plugin_registry().require(
        VendorKnowledgeSourceIdentity(
            provider_id=JIRA_ISSUE_TRACKER_PROVIDER_ID,
            integration_category=IntegrationCategory.ISSUE_TRACKER,
            source_kind=JIRA_ISSUES_SOURCE_KIND,
        )
    )
    materializer = VendorKnowledgeIndexedMaterializerContribution(
        identity=jira_plugin.identity,
        runtime_ref="indexed-source:jira:wrong",
        factory=JiraIssueStructuredRecordMaterializer,
    )
    with pytest.raises(
        VendorKnowledgeContributionError,
        match="materializer_runtime_ref_mismatch",
    ):
        VendorKnowledgeProviderContribution(
            provider_id=JIRA_ISSUE_TRACKER_PROVIDER_ID,
            integration_category=IntegrationCategory.ISSUE_TRACKER,
            source_plugins=(jira_plugin,),
            indexed_materializers=(materializer,),
        )

    with pytest.raises(
        VendorKnowledgeContributionError,
        match="duplicate_source_identity",
    ):
        VendorKnowledgeProviderContribution(
            provider_id=JIRA_ISSUE_TRACKER_PROVIDER_ID,
            integration_category=IntegrationCategory.ISSUE_TRACKER,
            source_plugins=(jira_plugin, jira_plugin),
        )

    with pytest.raises(
        VendorKnowledgeContributionError,
        match="duplicate_adapter_identity",
    ):
        VendorKnowledgeProviderContribution(
            provider_id=JIRA_ISSUE_TRACKER_PROVIDER_ID,
            integration_category=IntegrationCategory.ISSUE_TRACKER,
            adapters=(JiraIssuesKnowledgeAdapter(), JiraIssuesKnowledgeAdapter()),
            source_plugins=(jira_plugin,),
        )


def test_duplicate_materializer_runtime_ref_and_live_identity_fail_closed() -> None:
    plugins = build_default_vendor_knowledge_source_plugin_registry()
    first_identity = VendorKnowledgeSourceIdentity(
        provider_id=MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
        integration_category=IntegrationCategory.COLLABORATION_SUITE,
        source_kind=MSGRAPH_MAIL_SOURCE_KIND,
    )
    second_identity = VendorKnowledgeSourceIdentity(
        provider_id=MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
        integration_category=IntegrationCategory.COLLABORATION_SUITE,
        source_kind=MSGRAPH_TEAMS_CHANNEL_SOURCE_KIND,
    )
    first = VendorKnowledgeSourcePlugin(
        identity=first_identity,
        capabilities=(
            VendorKnowledgeModeCapability(
                mode=VendorKnowledgeMode.INDEXED,
                contract_version="vendor-knowledge.indexed.v1",
                operations=("index",),
                runtime_ref="indexed-source:shared",
            ),
        )
    )
    second = VendorKnowledgeSourcePlugin(
        identity=second_identity,
        capabilities=(
            VendorKnowledgeModeCapability(
                mode=VendorKnowledgeMode.INDEXED,
                contract_version="vendor-knowledge.indexed.v1",
                operations=("index",),
                runtime_ref="indexed-source:shared",
            ),
        ),
    )
    shared_runtime_ref = "indexed-source:shared"
    with pytest.raises(
        VendorKnowledgeContributionError,
        match="duplicate_materializer_runtime_ref",
    ):
        VendorKnowledgeProviderContribution(
            provider_id=MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
            integration_category=IntegrationCategory.COLLABORATION_SUITE,
            source_plugins=(first, second),
            indexed_materializers=(
                VendorKnowledgeIndexedMaterializerContribution(
                    identity=first.identity,
                    runtime_ref=shared_runtime_ref,
                    factory=MsGraphMailStructuredRecordMaterializer,
                ),
                VendorKnowledgeIndexedMaterializerContribution(
                    identity=second.identity,
                    runtime_ref=shared_runtime_ref,
                    factory=MsGraphTeamsChannelStructuredRecordMaterializer,
                ),
            ),
        )

    live = tuple(
        bundle
        for bundle in build_vendor_knowledge_live_registration_registry().list_registrations()
        if bundle.descriptor.provider_id == SLACK_CONVERSATION_CHANNEL_PROVIDER_ID
    )
    slack = plugins.require(
        VendorKnowledgeSourceIdentity(
            provider_id=SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
            integration_category=IntegrationCategory.CONVERSATION_CHANNEL,
            source_kind=SLACK_CONVERSATION_SOURCE_KIND,
        )
    )
    with pytest.raises(
        VendorKnowledgeContributionError,
        match="duplicate_live_capability_identity",
    ):
        VendorKnowledgeProviderContribution(
            provider_id=SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
            integration_category=IntegrationCategory.CONVERSATION_CHANNEL,
            source_plugins=(slack,),
            live_contributions=(*live, live[0]),
        )


def test_partial_modes_are_preserved_without_capability_inflation() -> None:
    plugins = build_default_vendor_knowledge_source_plugin_registry()
    contribution = _contribution_for(
        MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
        IntegrationCategory.COLLABORATION_SUITE,
    )
    source_plugins = {
        plugin.identity.source_kind: plugin for plugin in contribution.source_plugins
    }

    assert source_plugins[MSGRAPH_DRIVE_SOURCE_KIND].capabilities == plugins.require(
        VendorKnowledgeSourceIdentity(
            provider_id=MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
            integration_category=IntegrationCategory.COLLABORATION_SUITE,
            source_kind=MSGRAPH_DRIVE_SOURCE_KIND,
        )
    ).capabilities
    assert source_plugins[MSGRAPH_MAIL_SOURCE_KIND].supports(VendorKnowledgeMode.INDEXED)
    assert source_plugins[MSGRAPH_MAIL_SOURCE_KIND].supports(VendorKnowledgeMode.LIVE)
    assert not source_plugins[MSGRAPH_DRIVE_SOURCE_KIND].supports(
        VendorKnowledgeMode.INDEXED
    )


def test_empty_provider_and_invalid_category_are_rejected() -> None:
    with pytest.raises(
        VendorKnowledgeContributionError,
        match="provider_identity_invalid",
    ):
        VendorKnowledgeProviderContribution(
            provider_id="",
            integration_category=IntegrationCategory.ISSUE_TRACKER,
        )

    with pytest.raises(
        VendorKnowledgeContributionError,
        match="provider_identity_invalid",
    ):
        VendorKnowledgeProviderContribution(
            provider_id="safe-provider",
            integration_category="not-a-category",  # type: ignore[arg-type]
        )
