# © Artur Czarnecki. All rights reserved.

"""Tenant knowledge source binding helpers for connected workspace sources."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from local_workspace_application.workspaces.connected_source_candidate import (
    decode_slack_conversation_candidate_ref,
    validate_candidate_scope,
)
from local_workspace_application.workspaces.connected_source_ids import (
    tenant_binding_id,
)
from local_workspace_application.workspaces.connected_source_models import (
    ConnectedSourceBindingError,
    ConnectedSourceDiscoveryError,
    RemoteResourceTypeV1,
    SlackConversationKindV1,
)
from local_workspace_application.workspaces.connected_source_opaque_ref_codec import (
    ConfluenceSpaceCandidatePayload,
    GoogleWorkspaceCandidatePayload,
    JiraProjectCandidatePayload,
    RemoteResourceOpaqueRefCodec,
    VendorKnowledgeScopedSourceCandidatePayload,
)

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read import (
    MSGRAPH_CALENDAR_SOURCE_KIND,
    MSGRAPH_MAIL_SOURCE_KIND,
    MSGRAPH_TEAMS_CHANNEL_SOURCE_KIND,
    MSGRAPH_TEAMS_CHAT_SOURCE_KIND,
    MsGraphCalendar,
    MsGraphCalendarOnlineMeetingProvider,
    MsGraphCalendarViewWindow,
    MsGraphTeamsChatMessageWindow,
)
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
from intergrax.integrations.providers.conversation_channel.slack.integration import (
    SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
)
from intergrax.integrations.providers.conversation_channel.slack.knowledge_read import (
    SLACK_CONVERSATION_SOURCE_KIND,
    SlackConversationKind,
)
from intergrax.integrations.providers.issue_tracker.jira.integration import (
    JIRA_ISSUE_TRACKER_PROVIDER_ID,
)
from intergrax.integrations.providers.issue_tracker.jira.knowledge_read import (
    JIRA_ISSUES_SOURCE_KIND,
    JIRA_PROJECT_SCOPE_TYPE,
)
from intergrax.integrations.providers.wiki_knowledge.confluence.integration import (
    CONFLUENCE_WIKI_KNOWLEDGE_PROVIDER_ID,
)
from intergrax.integrations.providers.wiki_knowledge.confluence.knowledge_read import (
    CONFLUENCE_PAGES_SOURCE_KIND,
    CONFLUENCE_SPACE_SCOPE_TYPE,
)
from intergrax.runtime.vendor_knowledge.adapters.slack_conversation import (
    SLACK_CONVERSATION_SCOPE_TYPE,
    encode_slack_conversation_scope_id,
)
from intergrax.runtime.vendor_knowledge.adapters.ms365_graph_teams_chat import (
    MSGRAPH_TEAMS_CHAT_SCOPE_TYPE,
    encode_msgraph_teams_chat_scope_id,
)
from intergrax.runtime.vendor_knowledge.adapters.ms365_graph_teams_channel import (
    MSGRAPH_TEAMS_CHANNEL_SCOPE_TYPE,
    encode_msgraph_teams_channel_scope_id,
)
from intergrax.runtime.vendor_knowledge.adapters.ms365_graph_mail import (
    MSGRAPH_MAIL_SCOPE_TYPE,
    encode_msgraph_mail_folder_scope_id,
)
from intergrax.runtime.vendor_knowledge.adapters.ms365_graph_calendar import (
    MSGRAPH_CALENDAR_SCOPE_TYPE,
    encode_msgraph_calendar_scope_id,
)
from intergrax.runtime.vendor_knowledge.adapters.google_workspace_calendar import (
    GOOGLE_CALENDAR_SCOPE_TYPE,
)
from intergrax.runtime.vendor_knowledge.adapters.google_workspace_docs import (
    GOOGLE_DOCS_DOCUMENT_SCOPE_TYPE,
)
from intergrax.runtime.vendor_knowledge.adapters.google_workspace_sheets import (
    GOOGLE_SHEETS_SPREADSHEET_SCOPE_TYPE,
)
from intergrax.runtime.vendor_knowledge.bindings import (
    KnowledgeSourceBinding,
    KnowledgeSourceBindingService,
    KnowledgeSourceBindingStatus,
)
from intergrax.runtime.vendor_knowledge.errors import VendorKnowledgeError
from intergrax.runtime.vendor_knowledge.models import KnowledgeSourceScope


@dataclass(frozen=True, slots=True)
class SlackConversationTenantBindingRequest:
    tenant_id: str
    connection_ref: str
    conversation_id: str
    conversation_kind: SlackConversationKindV1
    safe_display_name: str
    root_oldest: str
    root_latest: str


def slack_conversation_tenant_binding_id(
    request: SlackConversationTenantBindingRequest,
) -> str:
    encoded_scope = encode_slack_conversation_scope_id(
        conversation_id=request.conversation_id,
        conversation_kind=SlackConversationKind(request.conversation_kind.value),
        oldest=request.root_oldest,
        latest=request.root_latest,
    )
    return tenant_binding_id(
        tenant_id=request.tenant_id,
        connection_ref=request.connection_ref,
        provider_id=SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
        integration_kind=IntegrationCategory.CONVERSATION_CHANNEL.value,
        source_kind=SLACK_CONVERSATION_SOURCE_KIND,
        encoded_scope=encoded_scope,
    )


class WorkspaceConnectedSourceTenantBindingService:
    def __init__(
        self,
        binding_service_factory: Callable[[str], KnowledgeSourceBindingService],
    ) -> None:
        self._binding_service_factory = binding_service_factory

    def create_or_get_equivalent_for_slack_conversation(
        self,
        request: SlackConversationTenantBindingRequest,
    ) -> KnowledgeSourceBinding:
        return self.create_or_get_equivalent(_slack_conversation_binding(request))

    def create_or_get_equivalent(
        self,
        binding: KnowledgeSourceBinding,
    ) -> KnowledgeSourceBinding:
        binding_service = self._binding_service_factory(binding.tenant_id)
        try:
            return binding_service.create_or_get_equivalent(binding)
        except VendorKnowledgeError as exc:
            raise ConnectedSourceBindingError("knowledge_source_binding_unavailable") from exc


class ProviderNeutralConnectedSourceCandidateAdapter:
    """Dispatch opaque candidates to provider-owned binding strategies."""

    def __init__(
        self,
        *,
        slack: SlackConnectedSourceCandidateAdapter,
        codec: RemoteResourceOpaqueRefCodec,
        discovery_service: Any,
    ) -> None:
        self._slack = slack
        self._codec = codec
        self._discovery = discovery_service

    def _graph_payload(self, opaque_candidate_ref: str):
        try:
            return self._codec.decode_msgraph_teams_chat_candidate(opaque_candidate_ref)
        except ConnectedSourceDiscoveryError:
            return None

    def _mail_payload(self, opaque_candidate_ref: str):
        try:
            return self._codec.decode_msgraph_mail_folder_candidate(opaque_candidate_ref)
        except ConnectedSourceDiscoveryError:
            return None

    def _teams_channel_payload(self, opaque_candidate_ref: str):
        try:
            return self._codec.decode_msgraph_teams_channel_candidate(opaque_candidate_ref)
        except ConnectedSourceDiscoveryError:
            return None

    def _calendar_payload(self, opaque_candidate_ref: str):
        try:
            return self._codec.decode_msgraph_calendar_candidate(opaque_candidate_ref)
        except ConnectedSourceDiscoveryError:
            return None

    def _google_payload(
        self,
        opaque_candidate_ref: str,
    ) -> GoogleWorkspaceCandidatePayload | None:
        try:
            return self._codec.decode_google_workspace_candidate(opaque_candidate_ref)
        except ConnectedSourceDiscoveryError:
            return None

    def _jira_payload(
        self,
        opaque_candidate_ref: str,
    ) -> JiraProjectCandidatePayload | None:
        try:
            return self._codec.decode_jira_project_candidate(opaque_candidate_ref)
        except ConnectedSourceDiscoveryError:
            return None

    def _confluence_payload(
        self,
        opaque_candidate_ref: str,
    ) -> ConfluenceSpaceCandidatePayload | None:
        try:
            return self._codec.decode_confluence_space_candidate(opaque_candidate_ref)
        except ConnectedSourceDiscoveryError:
            return None

    def _vendor_knowledge_scoped_payload(
        self,
        opaque_candidate_ref: str,
    ) -> VendorKnowledgeScopedSourceCandidatePayload | None:
        try:
            return self._codec.decode_vendor_knowledge_scoped_source_candidate(
                opaque_candidate_ref
            )
        except ConnectedSourceDiscoveryError:
            return None

    def build_binding(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        connection_ref: str,
        opaque_candidate_ref: str,
        root_oldest: str,
        root_latest: str,
        safe_display_name: str | None = None,
    ) -> KnowledgeSourceBinding:
        graph = self._graph_payload(opaque_candidate_ref)
        if graph is not None:
            _validate_candidate_scope_values(
                graph.tenant_id,
                graph.workspace_id,
                graph.connection_ref,
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                connection_ref=connection_ref,
            )
            try:
                window = MsGraphTeamsChatMessageWindow(
                    start_at=_parse_datetime(root_oldest),
                    end_at=_parse_datetime(root_latest),
                )
                scope_id = encode_msgraph_teams_chat_scope_id(
                    mailbox_user_id=graph.mailbox_user_id,
                    chat_remote_id=graph.chat_remote_id,
                    window=window,
                )
            except (ValueError, TypeError):
                raise ConnectedSourceBindingError("candidate_inaccessible") from None
            return KnowledgeSourceBinding(
                binding_id=tenant_binding_id(
                    tenant_id=tenant_id,
                    connection_ref=connection_ref,
                    provider_id="ms365_graph",
                    integration_kind=IntegrationCategory.COLLABORATION_SUITE.value,
                    source_kind=MSGRAPH_TEAMS_CHAT_SOURCE_KIND,
                    encoded_scope=scope_id,
                ),
                tenant_id=tenant_id,
                provider_id="ms365_graph",
                integration_kind=IntegrationCategory.COLLABORATION_SUITE,
                source_kind=MSGRAPH_TEAMS_CHAT_SOURCE_KIND,
                connection_ref=connection_ref,
                safe_display_name=safe_display_name or graph.safe_display_label,
                scope=KnowledgeSourceScope(
                    remote_scope_id=scope_id,
                    remote_scope_type=MSGRAPH_TEAMS_CHAT_SCOPE_TYPE,
                    safe_display_name=safe_display_name or graph.safe_display_label,
                    parameters={},
                ),
                status=KnowledgeSourceBindingStatus.ACTIVE,
                configuration_version=1,
            )

        mail = self._mail_payload(opaque_candidate_ref)
        if mail is not None:
            _validate_candidate_scope_values(
                mail.tenant_id,
                mail.workspace_id,
                mail.connection_ref,
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                connection_ref=connection_ref,
            )
            try:
                scope_id = encode_msgraph_mail_folder_scope_id(
                    mailbox_user_id=mail.mailbox_user_id,
                    folder_id=mail.folder_id,
                )
            except (ValueError, TypeError):
                raise ConnectedSourceBindingError("candidate_inaccessible") from None
            return KnowledgeSourceBinding(
                binding_id=tenant_binding_id(
                    tenant_id=tenant_id,
                    connection_ref=connection_ref,
                    provider_id="ms365_graph",
                    integration_kind=IntegrationCategory.COLLABORATION_SUITE.value,
                    source_kind=MSGRAPH_MAIL_SOURCE_KIND,
                    encoded_scope=scope_id,
                ),
                tenant_id=tenant_id,
                provider_id="ms365_graph",
                integration_kind=IntegrationCategory.COLLABORATION_SUITE,
                source_kind=MSGRAPH_MAIL_SOURCE_KIND,
                connection_ref=connection_ref,
                safe_display_name=safe_display_name or mail.safe_display_label,
                scope=KnowledgeSourceScope(
                    remote_scope_id=scope_id,
                    remote_scope_type=MSGRAPH_MAIL_SCOPE_TYPE,
                    safe_display_name=safe_display_name or mail.safe_display_label,
                    parameters={},
                ),
                status=KnowledgeSourceBindingStatus.ACTIVE,
                configuration_version=1,
            )

        channel = self._teams_channel_payload(opaque_candidate_ref)
        if channel is not None:
            _validate_candidate_scope_values(
                channel.tenant_id,
                channel.workspace_id,
                channel.connection_ref,
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                connection_ref=connection_ref,
            )
            scope_id = encode_msgraph_teams_channel_scope_id(
                team_remote_id=channel.team_remote_id,
                channel_remote_id=channel.channel_remote_id,
            )
            return KnowledgeSourceBinding(
                binding_id=tenant_binding_id(
                    tenant_id=tenant_id,
                    connection_ref=connection_ref,
                    provider_id="ms365_graph",
                    integration_kind=IntegrationCategory.COLLABORATION_SUITE.value,
                    source_kind=MSGRAPH_TEAMS_CHANNEL_SOURCE_KIND,
                    encoded_scope=scope_id,
                ),
                tenant_id=tenant_id,
                provider_id="ms365_graph",
                integration_kind=IntegrationCategory.COLLABORATION_SUITE,
                source_kind=MSGRAPH_TEAMS_CHANNEL_SOURCE_KIND,
                connection_ref=connection_ref,
                safe_display_name=safe_display_name or channel.safe_display_label,
                scope=KnowledgeSourceScope(
                    remote_scope_id=scope_id,
                    remote_scope_type=MSGRAPH_TEAMS_CHANNEL_SCOPE_TYPE,
                    safe_display_name=safe_display_name or channel.safe_display_label,
                    parameters={},
                ),
                status=KnowledgeSourceBindingStatus.ACTIVE,
                configuration_version=1,
            )

        calendar = self._calendar_payload(opaque_candidate_ref)
        if calendar is not None:
            _validate_candidate_scope_values(
                calendar.tenant_id,
                calendar.workspace_id,
                calendar.connection_ref,
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                connection_ref=connection_ref,
            )
            try:
                calendar_model = MsGraphCalendar(
                    mailbox_user_id=calendar.mailbox_user_id,
                    remote_id=calendar.calendar_remote_id,
                    name=calendar.safe_display_label,
                    change_key="candidate",
                    is_default_calendar=calendar.is_default_calendar,
                    can_edit=False,
                    can_share=False,
                    can_view_private_items=False,
                    is_removable=False,
                    owner=None,
                    allowed_online_meeting_providers=(),
                    default_online_meeting_provider=(
                        MsGraphCalendarOnlineMeetingProvider.UNKNOWN
                    ),
                )
                window = MsGraphCalendarViewWindow(
                    start_at=_parse_datetime(root_oldest),
                    end_at=_parse_datetime(root_latest),
                )
                scope_id = encode_msgraph_calendar_scope_id(
                    calendar=calendar_model,
                    window=window,
                )
            except (ValueError, TypeError):
                raise ConnectedSourceBindingError("candidate_inaccessible") from None
            return KnowledgeSourceBinding(
                binding_id=tenant_binding_id(
                    tenant_id=tenant_id,
                    connection_ref=connection_ref,
                    provider_id="ms365_graph",
                    integration_kind=IntegrationCategory.COLLABORATION_SUITE.value,
                    source_kind=MSGRAPH_CALENDAR_SOURCE_KIND,
                    encoded_scope=scope_id,
                ),
                tenant_id=tenant_id,
                provider_id="ms365_graph",
                integration_kind=IntegrationCategory.COLLABORATION_SUITE,
                source_kind=MSGRAPH_CALENDAR_SOURCE_KIND,
                connection_ref=connection_ref,
                safe_display_name=safe_display_name or calendar.safe_display_label,
                scope=KnowledgeSourceScope(
                    remote_scope_id=scope_id,
                    remote_scope_type=MSGRAPH_CALENDAR_SCOPE_TYPE,
                    safe_display_name=safe_display_name or calendar.safe_display_label,
                    parameters={},
                ),
                status=KnowledgeSourceBindingStatus.ACTIVE,
                configuration_version=1,
            )

        jira = self._jira_payload(opaque_candidate_ref)
        if jira is not None:
            _validate_candidate_scope_values(
                jira.tenant_id,
                jira.workspace_id,
                jira.connection_ref,
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                connection_ref=connection_ref,
            )
            return _scoped_vendor_binding(
                tenant_id=tenant_id,
                connection_ref=connection_ref,
                provider_id=JIRA_ISSUE_TRACKER_PROVIDER_ID,
                integration_kind=IntegrationCategory.ISSUE_TRACKER,
                source_kind=JIRA_ISSUES_SOURCE_KIND,
                scope_type=JIRA_PROJECT_SCOPE_TYPE,
                scope_id=jira.project_key,
                safe_display_name=safe_display_name or jira.safe_display_label,
            )

        confluence = self._confluence_payload(opaque_candidate_ref)
        if confluence is not None:
            _validate_candidate_scope_values(
                confluence.tenant_id,
                confluence.workspace_id,
                confluence.connection_ref,
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                connection_ref=connection_ref,
            )
            return _scoped_vendor_binding(
                tenant_id=tenant_id,
                connection_ref=connection_ref,
                provider_id=CONFLUENCE_WIKI_KNOWLEDGE_PROVIDER_ID,
                integration_kind=IntegrationCategory.WIKI_KNOWLEDGE,
                source_kind=CONFLUENCE_PAGES_SOURCE_KIND,
                scope_type=CONFLUENCE_SPACE_SCOPE_TYPE,
                scope_id=confluence.space_id,
                safe_display_name=safe_display_name or confluence.safe_display_label,
            )

        google = self._google_payload(opaque_candidate_ref)
        if google is not None:
            _validate_candidate_scope_values(
                google.tenant_id,
                google.workspace_id,
                google.connection_ref,
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                connection_ref=connection_ref,
            )
            return _google_workspace_binding(
                payload=google,
                tenant_id=tenant_id,
                connection_ref=connection_ref,
                safe_display_name=safe_display_name,
            )

        scoped = self._vendor_knowledge_scoped_payload(opaque_candidate_ref)
        if scoped is not None:
            _validate_candidate_scope_values(
                scoped.tenant_id,
                scoped.workspace_id,
                scoped.connection_ref,
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                connection_ref=connection_ref,
            )
            try:
                integration_kind = IntegrationCategory(scoped.integration_kind)
            except ValueError:
                raise ConnectedSourceBindingError("candidate_inaccessible") from None
            return _scoped_vendor_binding(
                tenant_id=tenant_id,
                connection_ref=connection_ref,
                provider_id=scoped.provider_id,
                integration_kind=integration_kind,
                source_kind=scoped.source_kind,
                scope_type=scoped.scope_type,
                scope_id=scoped.scope_id,
                safe_display_name=safe_display_name or scoped.safe_display_label,
            )

        return self._slack.build_binding(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            connection_ref=connection_ref,
            opaque_candidate_ref=opaque_candidate_ref,
            root_oldest=root_oldest,
            root_latest=root_latest,
            safe_display_name=safe_display_name,
        )

    async def revalidate_candidate_label(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        connection_ref: str,
        opaque_candidate_ref: str,
    ) -> str:
        graph = self._graph_payload(opaque_candidate_ref)
        if graph is not None:
            _validate_candidate_scope_values(
                graph.tenant_id,
                graph.workspace_id,
                graph.connection_ref,
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                connection_ref=connection_ref,
            )
            return await self._discovery.revalidate_candidate_label(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                connection_ref=connection_ref,
                resource_type=RemoteResourceTypeV1.MSGRAPH_TEAMS_CHAT,
                opaque_candidate_ref=opaque_candidate_ref,
            )

        mail = self._mail_payload(opaque_candidate_ref)
        if mail is not None:
            _validate_candidate_scope_values(
                mail.tenant_id,
                mail.workspace_id,
                mail.connection_ref,
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                connection_ref=connection_ref,
            )
            return await self._discovery.revalidate_candidate_label(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                connection_ref=connection_ref,
                resource_type=RemoteResourceTypeV1.MSGRAPH_MAIL_FOLDER,
                opaque_candidate_ref=opaque_candidate_ref,
            )

        channel = self._teams_channel_payload(opaque_candidate_ref)
        if channel is not None:
            _validate_candidate_scope_values(
                channel.tenant_id,
                channel.workspace_id,
                channel.connection_ref,
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                connection_ref=connection_ref,
            )
            return await self._discovery.revalidate_candidate_label(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                connection_ref=connection_ref,
                resource_type=RemoteResourceTypeV1.MSGRAPH_TEAMS_CHANNEL,
                opaque_candidate_ref=opaque_candidate_ref,
            )

        calendar = self._calendar_payload(opaque_candidate_ref)
        if calendar is not None:
            _validate_candidate_scope_values(
                calendar.tenant_id,
                calendar.workspace_id,
                calendar.connection_ref,
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                connection_ref=connection_ref,
            )
            return await self._discovery.revalidate_candidate_label(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                connection_ref=connection_ref,
                resource_type=RemoteResourceTypeV1.MSGRAPH_CALENDAR,
                opaque_candidate_ref=opaque_candidate_ref,
            )

        jira = self._jira_payload(opaque_candidate_ref)
        if jira is not None:
            _validate_candidate_scope_values(
                jira.tenant_id,
                jira.workspace_id,
                jira.connection_ref,
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                connection_ref=connection_ref,
            )
            return await self._discovery.revalidate_candidate_label(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                connection_ref=connection_ref,
                resource_type=RemoteResourceTypeV1.JIRA_PROJECT,
                opaque_candidate_ref=opaque_candidate_ref,
            )

        confluence = self._confluence_payload(opaque_candidate_ref)
        if confluence is not None:
            _validate_candidate_scope_values(
                confluence.tenant_id,
                confluence.workspace_id,
                confluence.connection_ref,
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                connection_ref=connection_ref,
            )
            return await self._discovery.revalidate_candidate_label(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                connection_ref=connection_ref,
                resource_type=RemoteResourceTypeV1.CONFLUENCE_SPACE,
                opaque_candidate_ref=opaque_candidate_ref,
            )

        google = self._google_payload(opaque_candidate_ref)
        if google is not None:
            _validate_candidate_scope_values(
                google.tenant_id,
                google.workspace_id,
                google.connection_ref,
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                connection_ref=connection_ref,
            )
            return await self._discovery.revalidate_candidate_label(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                connection_ref=connection_ref,
                resource_type=google.resource_type,
                opaque_candidate_ref=opaque_candidate_ref,
            )

        scoped = self._vendor_knowledge_scoped_payload(opaque_candidate_ref)
        if scoped is not None:
            _validate_candidate_scope_values(
                scoped.tenant_id,
                scoped.workspace_id,
                scoped.connection_ref,
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                connection_ref=connection_ref,
            )
            return await self._discovery.revalidate_candidate_label(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                connection_ref=connection_ref,
                resource_type=RemoteResourceTypeV1.VENDOR_KNOWLEDGE_SCOPED_SOURCE,
                opaque_candidate_ref=opaque_candidate_ref,
            )

        return await self._slack.revalidate_candidate_label(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            connection_ref=connection_ref,
            opaque_candidate_ref=opaque_candidate_ref,
        )


def _validate_candidate_scope_values(
    payload_tenant: str,
    payload_workspace: str,
    payload_connection: str,
    *,
    tenant_id: str,
    workspace_id: str,
    connection_ref: str,
) -> None:
    if payload_tenant != tenant_id or payload_workspace != workspace_id:
        raise ConnectedSourceBindingError("workspace_not_found")
    if payload_connection != connection_ref:
        raise ConnectedSourceBindingError("connection_not_attached")


def _parse_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("datetime must be timezone-aware")
    return parsed


def _slack_conversation_binding(
    request: SlackConversationTenantBindingRequest,
) -> KnowledgeSourceBinding:
    binding_id = slack_conversation_tenant_binding_id(request)
    encoded_scope = encode_slack_conversation_scope_id(
        conversation_id=request.conversation_id,
        conversation_kind=SlackConversationKind(request.conversation_kind.value),
        oldest=request.root_oldest,
        latest=request.root_latest,
    )
    binding = KnowledgeSourceBinding(
        binding_id=binding_id,
        tenant_id=request.tenant_id,
        provider_id=SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
        integration_kind=IntegrationCategory.CONVERSATION_CHANNEL,
        source_kind=SLACK_CONVERSATION_SOURCE_KIND,
        connection_ref=request.connection_ref,
        safe_display_name=request.safe_display_name,
        scope=KnowledgeSourceScope(
            remote_scope_id=encoded_scope,
            remote_scope_type=SLACK_CONVERSATION_SCOPE_TYPE,
            safe_display_name=request.safe_display_name,
            parameters={},
        ),
        status=KnowledgeSourceBindingStatus.ACTIVE,
        configuration_version=1,
    )
    return binding


def _google_workspace_binding(
    *,
    payload: GoogleWorkspaceCandidatePayload,
    tenant_id: str,
    connection_ref: str,
    safe_display_name: str | None,
) -> KnowledgeSourceBinding:
    source_kind, scope_type = {
        RemoteResourceTypeV1.GOOGLE_WORKSPACE_CALENDAR: (
            GOOGLE_CALENDAR_SOURCE_KIND,
            GOOGLE_CALENDAR_SCOPE_TYPE,
        ),
        RemoteResourceTypeV1.GOOGLE_WORKSPACE_DOCS: (
            GOOGLE_DOCS_SOURCE_KIND,
            GOOGLE_DOCS_DOCUMENT_SCOPE_TYPE,
        ),
        RemoteResourceTypeV1.GOOGLE_WORKSPACE_SHEETS: (
            GOOGLE_SHEETS_SOURCE_KIND,
            GOOGLE_SHEETS_SPREADSHEET_SCOPE_TYPE,
        ),
    }.get(payload.resource_type, (None, None))
    if source_kind is None or scope_type is None:
        raise ConnectedSourceBindingError("candidate_inaccessible")
    display_name = safe_display_name or payload.safe_display_label
    binding_id = tenant_binding_id(
        tenant_id=tenant_id,
        connection_ref=connection_ref,
        provider_id=GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID,
        integration_kind=IntegrationCategory.COLLABORATION_SUITE.value,
        source_kind=source_kind,
        encoded_scope=payload.remote_resource_id,
    )
    return KnowledgeSourceBinding(
        binding_id=binding_id,
        tenant_id=tenant_id,
        provider_id=GOOGLE_WORKSPACE_COLLABORATION_SUITE_PROVIDER_ID,
        integration_kind=IntegrationCategory.COLLABORATION_SUITE,
        source_kind=source_kind,
        connection_ref=connection_ref,
        safe_display_name=display_name,
        scope=KnowledgeSourceScope(
            remote_scope_id=payload.remote_resource_id,
            remote_scope_type=scope_type,
            safe_display_name=display_name,
            parameters={},
        ),
        status=KnowledgeSourceBindingStatus.ACTIVE,
        configuration_version=1,
    )


def _scoped_vendor_binding(
    *,
    tenant_id: str,
    connection_ref: str,
    provider_id: str,
    integration_kind: IntegrationCategory,
    source_kind: str,
    scope_type: str,
    scope_id: str,
    safe_display_name: str,
) -> KnowledgeSourceBinding:
    binding_id = tenant_binding_id(
        tenant_id=tenant_id,
        connection_ref=connection_ref,
        provider_id=provider_id,
        integration_kind=integration_kind.value,
        source_kind=source_kind,
        encoded_scope=scope_id,
    )
    return KnowledgeSourceBinding(
        binding_id=binding_id,
        tenant_id=tenant_id,
        provider_id=provider_id,
        integration_kind=integration_kind,
        source_kind=source_kind,
        connection_ref=connection_ref,
        safe_display_name=safe_display_name,
        scope=KnowledgeSourceScope(
            remote_scope_id=scope_id,
            remote_scope_type=scope_type,
            safe_display_name=safe_display_name,
            parameters={},
        ),
        status=KnowledgeSourceBindingStatus.ACTIVE,
        configuration_version=1,
    )


class SlackConnectedSourceCandidateAdapter:
    """Provider-owned candidate codec and revalidation strategy."""

    def __init__(
        self,
        *,
        codec: RemoteResourceOpaqueRefCodec,
        discovery_service: Any,
    ) -> None:
        self._codec = codec
        self._discovery = discovery_service

    def _request(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        connection_ref: str,
        opaque_candidate_ref: str,
        root_oldest: str,
        root_latest: str,
        safe_display_name: str | None,
    ) -> SlackConversationTenantBindingRequest:
        payload = decode_slack_conversation_candidate_ref(
            self._codec,
            opaque_candidate_ref,
        )
        validate_candidate_scope(
            payload,
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            connection_ref=connection_ref,
        )
        return SlackConversationTenantBindingRequest(
            tenant_id=tenant_id,
            connection_ref=connection_ref,
            conversation_id=payload.conversation_id,
            conversation_kind=payload.conversation_kind,
            safe_display_name=safe_display_name or payload.safe_display_label,
            root_oldest=root_oldest,
            root_latest=root_latest,
        )

    def build_binding(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        connection_ref: str,
        opaque_candidate_ref: str,
        root_oldest: str,
        root_latest: str,
        safe_display_name: str | None = None,
    ) -> KnowledgeSourceBinding:
        return _slack_conversation_binding(
            self._request(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                connection_ref=connection_ref,
                opaque_candidate_ref=opaque_candidate_ref,
                root_oldest=root_oldest,
                root_latest=root_latest,
                safe_display_name=safe_display_name,
            )
        )

    async def revalidate_candidate_label(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        connection_ref: str,
        opaque_candidate_ref: str,
    ) -> str:
        payload = decode_slack_conversation_candidate_ref(
            self._codec,
            opaque_candidate_ref,
        )
        validate_candidate_scope(
            payload,
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            connection_ref=connection_ref,
        )
        return await self._discovery.revalidate_candidate_label(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            connection_ref=connection_ref,
            resource_type=RemoteResourceTypeV1.SLACK_CONVERSATION,
            opaque_candidate_ref=opaque_candidate_ref,
        )
