"""Application-owned Vendor Knowledge contribution hooks."""

from __future__ import annotations

from dataclasses import dataclass, replace

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
from intergrax.runtime.vendor_knowledge.contribution import (
    VendorKnowledgeDiscoveryContribution,
    VendorKnowledgeIndexedMaterializerContribution,
)
from intergrax.runtime.vendor_knowledge.contribution_catalog import (
    VendorKnowledgeContributionCatalog,
    build_default_vendor_knowledge_contribution_catalog,
)
from intergrax.runtime.vendor_knowledge.connections import KnowledgeConnectionRegistry
from intergrax.runtime.vendor_knowledge.plugin import VendorKnowledgeSourceIdentity
from intergrax.runtime.vendor_knowledge.confluence_indexed_materializers import (
    ConfluencePageRichTextMaterializer,
)
from intergrax.runtime.vendor_knowledge.google_workspace_indexed_materializers import (
    GoogleCalendarStructuredRecordMaterializer,
    GoogleDocsStructuredRecordMaterializer,
    GoogleSheetsStructuredRecordMaterializer,
)
from intergrax.runtime.vendor_knowledge.jira_indexed_materializers import (
    JiraIssueStructuredRecordMaterializer,
)
from intergrax.runtime.vendor_knowledge.ms365_graph_indexed_materializers import (
    MsGraphCalendarStructuredRecordMaterializer,
    MsGraphMailStructuredRecordMaterializer,
    MsGraphTeamsChannelStructuredRecordMaterializer,
)
from local_workspace_application.workspaces.connected_source_opaque_ref_codec import (
    RemoteResourceOpaqueRefCodec,
)
from local_workspace_application.workspaces.connected_source_discovery_atlassian import (
    ConfluenceKnownSpaceCatalog,
    ConfluenceSpaceDiscoveryStrategy,
    JiraKnownProjectCatalog,
    JiraProjectDiscoveryStrategy,
)
from local_workspace_application.workspaces.connected_source_discovery_google_workspace import (
    GoogleWorkspaceKnownResourceCatalog,
    GoogleWorkspaceKnownResourceDiscoveryStrategy,
)
from local_workspace_application.workspaces.connected_source_discovery_msgraph import (
    MsGraphCalendarDiscoveryStrategy,
    MsGraphMailFolderDiscoveryStrategy,
    MsGraphTeamsChannelDiscoveryStrategy,
    MsGraphTeamsChatDiscoveryStrategy,
)
from local_workspace_application.workspaces.connected_source_discovery_slack import (
    SlackRemoteResourceDiscoveryStrategy,
)
from local_workspace_application.workspaces.connected_source_materializer import (
    MsGraphTeamsChatStructuredRecordMaterializer,
    SlackConversationStructuredRecordMaterializer,
)
from local_workspace_application.workspaces.connected_source_models import RemoteResourceTypeV1


@dataclass(frozen=True, slots=True)
class VendorKnowledgeApplicationExtensionContext:
    """Typed host resources available to application-owned contribution hooks."""

    connection_registry: KnowledgeConnectionRegistry
    opaque_ref_codec: RemoteResourceOpaqueRefCodec
    google_known_resource_catalog: GoogleWorkspaceKnownResourceCatalog | None = None
    jira_known_project_catalog: JiraKnownProjectCatalog | None = None
    confluence_known_space_catalog: ConfluenceKnownSpaceCatalog | None = None
    msgraph_mailbox_user_id: str | None = None
    msgraph_teams_channel_team_id: str | None = None


def _identity(
    provider_id: str,
    integration_category: IntegrationCategory,
    source_kind: str,
) -> VendorKnowledgeSourceIdentity:
    return VendorKnowledgeSourceIdentity(
        provider_id=provider_id,
        integration_category=integration_category,
        source_kind=source_kind,
    )


def _discovery(
    identity: VendorKnowledgeSourceIdentity,
    factory,
) -> VendorKnowledgeDiscoveryContribution:
    return VendorKnowledgeDiscoveryContribution(identity=identity, factory=factory)


def _materializer(
    identity: VendorKnowledgeSourceIdentity,
    runtime_ref: str,
    factory,
) -> VendorKnowledgeIndexedMaterializerContribution:
    return VendorKnowledgeIndexedMaterializerContribution(
        identity=identity,
        runtime_ref=runtime_ref,
        factory=factory,
    )


def _augment_catalog(
    catalog: VendorKnowledgeContributionCatalog,
    extensions: dict[
        tuple[str, IntegrationCategory],
        tuple[
            tuple[VendorKnowledgeDiscoveryContribution, ...],
            tuple[VendorKnowledgeIndexedMaterializerContribution, ...],
        ],
    ],
) -> VendorKnowledgeContributionCatalog:
    updated = []
    for contribution in catalog.list_contributions():
        extension = extensions.get(contribution.provider_key)
        if extension is None:
            updated.append(contribution)
            continue
        discovery, materializers = extension
        updated.append(
            replace(
                contribution,
                discovery_contributions=discovery,
                indexed_materializers=materializers,
            )
        )
    return VendorKnowledgeContributionCatalog(updated)


def _build_extensions(
    context: VendorKnowledgeApplicationExtensionContext | None,
) -> dict[
    tuple[str, IntegrationCategory],
    tuple[
        tuple[VendorKnowledgeDiscoveryContribution, ...],
        tuple[VendorKnowledgeIndexedMaterializerContribution, ...],
    ],
]:
    category = IntegrationCategory.COLLABORATION_SUITE
    slack_identity = _identity(
        SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
        IntegrationCategory.CONVERSATION_CHANNEL,
        SLACK_CONVERSATION_SOURCE_KIND,
    )
    graph_identities = {
        source_kind: _identity(
            MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
            category,
            source_kind,
        )
        for source_kind in (
            MSGRAPH_MAIL_SOURCE_KIND,
            MSGRAPH_TEAMS_CHANNEL_SOURCE_KIND,
            MSGRAPH_TEAMS_CHAT_SOURCE_KIND,
            MSGRAPH_CALENDAR_SOURCE_KIND,
        )
    }
    google_identities = {
        source_kind: _identity(
            GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID,
            category,
            source_kind,
        )
        for source_kind in (
            GOOGLE_CALENDAR_SOURCE_KIND,
            GOOGLE_DOCS_SOURCE_KIND,
            GOOGLE_SHEETS_SOURCE_KIND,
        )
    }
    jira_identity = _identity(
        JIRA_ISSUE_TRACKER_PROVIDER_ID,
        IntegrationCategory.ISSUE_TRACKER,
        JIRA_ISSUES_SOURCE_KIND,
    )
    confluence_identity = _identity(
        CONFLUENCE_WIKI_KNOWLEDGE_PROVIDER_ID,
        IntegrationCategory.WIKI_KNOWLEDGE,
        CONFLUENCE_PAGES_SOURCE_KIND,
    )

    def slack_discovery(extension_context):
        return SlackRemoteResourceDiscoveryStrategy(
            connection_registry=extension_context.connection_registry,
            opaque_ref_codec=extension_context.opaque_ref_codec,
        )

    def graph_mail_discovery(extension_context):
        return MsGraphMailFolderDiscoveryStrategy(
            connection_registry=extension_context.connection_registry,
            opaque_ref_codec=extension_context.opaque_ref_codec,
            mailbox_user_id=extension_context.msgraph_mailbox_user_id,
        )

    def graph_channel_discovery(extension_context):
        return MsGraphTeamsChannelDiscoveryStrategy(
            connection_registry=extension_context.connection_registry,
            opaque_ref_codec=extension_context.opaque_ref_codec,
            team_remote_id=extension_context.msgraph_teams_channel_team_id,
        )

    def graph_chat_discovery(extension_context):
        return MsGraphTeamsChatDiscoveryStrategy(
            connection_registry=extension_context.connection_registry,
            opaque_ref_codec=extension_context.opaque_ref_codec,
            mailbox_user_id=extension_context.msgraph_mailbox_user_id,
        )

    def graph_calendar_discovery(extension_context):
        return MsGraphCalendarDiscoveryStrategy(
            connection_registry=extension_context.connection_registry,
            opaque_ref_codec=extension_context.opaque_ref_codec,
            mailbox_user_id=extension_context.msgraph_mailbox_user_id,
        )

    def google_discovery(resource_type, safe_description):
        def factory(extension_context):
            return GoogleWorkspaceKnownResourceDiscoveryStrategy(
                connection_registry=extension_context.connection_registry,
                opaque_ref_codec=extension_context.opaque_ref_codec,
                known_resources=extension_context.google_known_resource_catalog,
                resource_type=resource_type,
                safe_description=safe_description,
            )

        return factory

    def jira_discovery(extension_context):
        return JiraProjectDiscoveryStrategy(
            connection_registry=extension_context.connection_registry,
            opaque_ref_codec=extension_context.opaque_ref_codec,
            known_projects=extension_context.jira_known_project_catalog,
        )

    def confluence_discovery(extension_context):
        return ConfluenceSpaceDiscoveryStrategy(
            connection_registry=extension_context.connection_registry,
            opaque_ref_codec=extension_context.opaque_ref_codec,
            known_spaces=extension_context.confluence_known_space_catalog,
        )

    return {
        (
            SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
            IntegrationCategory.CONVERSATION_CHANNEL,
        ): (
            (_discovery(slack_identity, slack_discovery),),
            (
                _materializer(
                    slack_identity,
                    "indexed-source:slack:slack_conversation",
                    SlackConversationStructuredRecordMaterializer,
                ),
            ),
        ),
        (
            MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
            category,
        ): (
            (
                _discovery(
                    graph_identities[MSGRAPH_MAIL_SOURCE_KIND],
                    graph_mail_discovery,
                ),
                _discovery(
                    graph_identities[MSGRAPH_TEAMS_CHANNEL_SOURCE_KIND],
                    graph_channel_discovery,
                ),
                _discovery(
                    graph_identities[MSGRAPH_TEAMS_CHAT_SOURCE_KIND],
                    graph_chat_discovery,
                ),
                _discovery(
                    graph_identities[MSGRAPH_CALENDAR_SOURCE_KIND],
                    graph_calendar_discovery,
                ),
            ),
            (
                _materializer(
                    graph_identities[MSGRAPH_MAIL_SOURCE_KIND],
                    "indexed-source:ms365_graph:mail",
                    MsGraphMailStructuredRecordMaterializer,
                ),
                _materializer(
                    graph_identities[MSGRAPH_TEAMS_CHANNEL_SOURCE_KIND],
                    "indexed-source:ms365_graph:teams_channel",
                    MsGraphTeamsChannelStructuredRecordMaterializer,
                ),
                _materializer(
                    graph_identities[MSGRAPH_TEAMS_CHAT_SOURCE_KIND],
                    "indexed-source:ms365_graph:teams_chat",
                    MsGraphTeamsChatStructuredRecordMaterializer,
                ),
                _materializer(
                    graph_identities[MSGRAPH_CALENDAR_SOURCE_KIND],
                    "indexed-source:ms365_graph:calendar",
                    MsGraphCalendarStructuredRecordMaterializer,
                ),
            ),
        ),
        (
            GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID,
            category,
        ): (
            (
                _discovery(
                    google_identities[GOOGLE_CALENDAR_SOURCE_KIND],
                    google_discovery(
                        RemoteResourceTypeV1.GOOGLE_WORKSPACE_CALENDAR,
                        "Google Workspace Calendar",
                    ),
                ),
                _discovery(
                    google_identities[GOOGLE_DOCS_SOURCE_KIND],
                    google_discovery(
                        RemoteResourceTypeV1.GOOGLE_WORKSPACE_DOCS,
                        "Google Workspace known document",
                    ),
                ),
                _discovery(
                    google_identities[GOOGLE_SHEETS_SOURCE_KIND],
                    google_discovery(
                        RemoteResourceTypeV1.GOOGLE_WORKSPACE_SHEETS,
                        "Google Workspace known spreadsheet",
                    ),
                ),
            ),
            (
                _materializer(
                    google_identities[GOOGLE_CALENDAR_SOURCE_KIND],
                    "indexed-source:google_workspace:calendar",
                    GoogleCalendarStructuredRecordMaterializer,
                ),
                _materializer(
                    google_identities[GOOGLE_DOCS_SOURCE_KIND],
                    "indexed-source:google_workspace:docs",
                    GoogleDocsStructuredRecordMaterializer,
                ),
                _materializer(
                    google_identities[GOOGLE_SHEETS_SOURCE_KIND],
                    "indexed-source:google_workspace:sheets",
                    GoogleSheetsStructuredRecordMaterializer,
                ),
            ),
        ),
        (
            JIRA_ISSUE_TRACKER_PROVIDER_ID,
            IntegrationCategory.ISSUE_TRACKER,
        ): (
            (_discovery(jira_identity, jira_discovery),),
            (
                _materializer(
                    jira_identity,
                    "indexed-source:jira:issues",
                    JiraIssueStructuredRecordMaterializer,
                ),
            ),
        ),
        (
            CONFLUENCE_WIKI_KNOWLEDGE_PROVIDER_ID,
            IntegrationCategory.WIKI_KNOWLEDGE,
        ): (
            (_discovery(confluence_identity, confluence_discovery),),
            (
                _materializer(
                    confluence_identity,
                    "indexed-source:confluence:pages",
                    ConfluencePageRichTextMaterializer,
                ),
            ),
        ),
    }


def build_default_vendor_knowledge_application_contribution_catalog(
    context: VendorKnowledgeApplicationExtensionContext | None = None,
    *,
    discover_entry_points: bool = False,
) -> VendorKnowledgeContributionCatalog:
    catalog = build_default_vendor_knowledge_contribution_catalog(
        discover_entry_points=discover_entry_points,
    )
    return _augment_catalog(catalog, _build_extensions(context))


__all__ = [
    "VendorKnowledgeApplicationExtensionContext",
    "build_default_vendor_knowledge_application_contribution_catalog",
]
