# © Artur Czarnecki. All rights reserved.

"""Provider-neutral remote resource discovery for connected workspace sources."""

from __future__ import annotations

from typing import Protocol

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.conversation_channel.slack.integration import (
    SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
    SlackConversationChannelIntegration,
)
from intergrax.integrations.providers.conversation_channel.slack.knowledge_read import (
    SlackConversationKind,
)
from intergrax.runtime.vendor_knowledge.connections import KnowledgeConnectionRegistry
from intergrax.runtime.vendor_knowledge.errors import VendorKnowledgeError, VendorKnowledgeErrorCode
from local_workspace_application.workspaces.connected_source_candidate import (
    encode_slack_conversation_candidate_ref,
)
from local_workspace_application.workspaces.connected_source_models import (
    ConnectedSourceDiscoveryError,
    RemoteResourceCandidateV1,
    RemoteResourceDiscoveryPageV1,
    RemoteResourceTypeV1,
    SlackConversationKindV1,
)
from local_workspace_application.workspaces.knowledge_configuration_models import (
    WorkspaceConnectionAttachmentStatusV1,
)
from local_workspace_application.workspaces.knowledge_configuration_service import (
    WorkspaceKnowledgeConfigurationService,
)
from local_workspace_application.workspaces.models import Workspace


class WorkspaceRemoteResourceDiscoveryPort(Protocol):
    async def list_remote_resources(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        connection_ref: str,
        resource_type: RemoteResourceTypeV1,
        cursor: str | None,
        limit: int,
    ) -> RemoteResourceDiscoveryPageV1:
        ...


class WorkspaceKnowledgeWorkspaceLookupPort(Protocol):
    def require_workspace(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
    ) -> Workspace | None:
        ...


def _map_kind(kind: SlackConversationKind) -> SlackConversationKindV1:
    return SlackConversationKindV1(kind.value)


class WorkspaceRemoteResourceDiscoveryService:
    def __init__(
        self,
        *,
        workspace_lookup: WorkspaceKnowledgeWorkspaceLookupPort,
        configuration_reader: WorkspaceKnowledgeConfigurationService,
        connection_registry: KnowledgeConnectionRegistry,
    ) -> None:
        self._workspace_lookup = workspace_lookup
        self._configuration_reader = configuration_reader
        self._connection_registry = connection_registry

    async def list_remote_resources(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        connection_ref: str,
        resource_type: RemoteResourceTypeV1,
        cursor: str | None,
        limit: int,
    ) -> RemoteResourceDiscoveryPageV1:
        if resource_type is not RemoteResourceTypeV1.SLACK_CONVERSATION:
            raise ConnectedSourceDiscoveryError("resource_type_unsupported")
        if limit < 1 or limit > 100:
            raise ConnectedSourceDiscoveryError("discovery_limit_invalid")

        self._require_attached_connection(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            connection_ref=connection_ref,
        )
        integration = self._resolve_slack_integration(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            connection_ref=connection_ref,
        )

        page = await integration.list_accessible_conversations_page(
            cursor=cursor,
            limit=limit,
        )

        items: list[RemoteResourceCandidateV1] = []
        for summary in page.items:
            kind = _map_kind(summary.kind)
            candidate_ref = encode_slack_conversation_candidate_ref(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                connection_ref=connection_ref,
                conversation_id=summary.conversation_id,
                conversation_kind=kind,
                safe_display_label=summary.safe_name,
            )
            description = summary.safe_topic or summary.safe_purpose
            items.append(
                RemoteResourceCandidateV1(
                    opaque_candidate_ref=candidate_ref,
                    resource_type=RemoteResourceTypeV1.SLACK_CONVERSATION,
                    safe_display_label=summary.safe_name,
                    conversation_kind=kind,
                    is_archived=summary.is_archived,
                    is_private=summary.is_private,
                    safe_description=description,
                )
            )

        return RemoteResourceDiscoveryPageV1(
            items=tuple(items),
            next_cursor=page.next_cursor,
        )

    def _require_attached_connection(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        connection_ref: str,
    ) -> None:
        workspace = self._workspace_lookup.require_workspace(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
        )
        if workspace is None:
            raise ConnectedSourceDiscoveryError("workspace_not_found")
        configuration = self._configuration_reader.get_configuration(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
        )
        if configuration is None:
            raise ConnectedSourceDiscoveryError("connection_not_attached")
        attachment = None
        for item in configuration.connection_attachments:
            if item.connection_ref == connection_ref:
                attachment = item
                break
        if attachment is None or attachment.status is not WorkspaceConnectionAttachmentStatusV1.ATTACHED:
            raise ConnectedSourceDiscoveryError("connection_not_attached")

    async def revalidate_candidate_label(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        connection_ref: str,
        conversation_id: str,
        conversation_kind: SlackConversationKindV1,
    ) -> str:
        integration = self._resolve_slack_integration(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            connection_ref=connection_ref,
        )
        cursor: str | None = None
        while True:
            page = await integration.list_accessible_conversations_page(
                cursor=cursor,
                limit=100,
            )
            for summary in page.items:
                if (
                    summary.conversation_id == conversation_id
                    and summary.kind.value == conversation_kind.value
                ):
                    return summary.safe_name
            if page.next_cursor is None:
                break
            cursor = page.next_cursor
        raise ConnectedSourceDiscoveryError("candidate_inaccessible")

    def _resolve_slack_integration(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        connection_ref: str,
    ) -> SlackConversationChannelIntegration:
        workspace = self._workspace_lookup.require_workspace(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
        )
        if workspace is None:
            raise ConnectedSourceDiscoveryError("workspace_not_found")
        configuration = self._configuration_reader.get_configuration(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
        )
        if configuration is None:
            raise ConnectedSourceDiscoveryError("connection_not_attached")
        attachment = None
        for item in configuration.connection_attachments:
            if item.connection_ref == connection_ref:
                attachment = item
                break
        if attachment is None or attachment.status is not WorkspaceConnectionAttachmentStatusV1.ATTACHED:
            raise ConnectedSourceDiscoveryError("connection_not_attached")
        try:
            integration = self._connection_registry.resolve(
                tenant_id=tenant_id,
                connection_ref=connection_ref,
                provider_id=SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
                integration_kind=IntegrationCategory.CONVERSATION_CHANNEL,
            )
        except VendorKnowledgeError as exc:
            if exc.code in {
                VendorKnowledgeErrorCode.INTEGRATION_NOT_FOUND,
                VendorKnowledgeErrorCode.AUTHORIZATION_DENIED,
                VendorKnowledgeErrorCode.AUTHENTICATION_FAILED,
            }:
                raise ConnectedSourceDiscoveryError("connection_unavailable") from exc
            raise ConnectedSourceDiscoveryError("connection_unavailable") from exc
        if not isinstance(integration, SlackConversationChannelIntegration):
            raise ConnectedSourceDiscoveryError("connection_incompatible")
        return integration
