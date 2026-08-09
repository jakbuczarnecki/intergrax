# © Artur Czarnecki. All rights reserved.

"""Microsoft Graph Teams Chat discovery strategy."""

from __future__ import annotations

import asyncio

from local_workspace_application.workspaces.connected_source_models import (
    ConnectedSourceDiscoveryError,
    RemoteResourceCandidateV1,
    RemoteResourceTypeV1,
)
from local_workspace_application.workspaces.connected_source_opaque_ref_codec import (
    RemoteResourceOpaqueRefCodec,
)
from local_workspace_application.workspaces.connected_source_discovery_strategy import (
    ConnectedSourceRevalidationLimits,
    RemoteResourceStrategyPage,
)

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.collaboration_suite.ms365_graph.integration import (
    MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
    Ms365GraphCollaborationSuiteIntegration,
)
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read import (
    MsGraphKnowledgeContinuation,
    MsGraphKnowledgeContinuationKind,
)
from intergrax.runtime.vendor_knowledge.connections import KnowledgeConnectionRegistry
from intergrax.runtime.vendor_knowledge.errors import VendorKnowledgeError


class MsGraphTeamsChatDiscoveryStrategy:
    resource_type = RemoteResourceTypeV1.MSGRAPH_TEAMS_CHAT

    def __init__(
        self,
        *,
        connection_registry: KnowledgeConnectionRegistry,
        opaque_ref_codec: RemoteResourceOpaqueRefCodec,
        mailbox_user_id: str | None,
    ) -> None:
        self._connection_registry = connection_registry
        self._codec = opaque_ref_codec
        self._mailbox_user_id = mailbox_user_id.strip() if mailbox_user_id else None

    async def list_remote_resources(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        connection_ref: str,
        provider_cursor: str | None,
        limit: int,
    ) -> RemoteResourceStrategyPage:
        if self._mailbox_user_id is None:
            raise ConnectedSourceDiscoveryError("connection_unavailable")
        integration = self._resolve_integration(
            tenant_id=tenant_id,
            connection_ref=connection_ref,
        )
        continuation = (
            MsGraphKnowledgeContinuation(
                kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
                url=provider_cursor,
            )
            if provider_cursor is not None
            else None
        )
        page = await asyncio.to_thread(
            integration.read_teams_chats_page,
            mailbox_user_id=self._mailbox_user_id,
            continuation=continuation,
            limit=min(limit, 50),
        )
        items: list[RemoteResourceCandidateV1] = []
        for chat in page.items:
            label = chat.topic or f"Teams chat {chat.remote_id}"
            items.append(
                RemoteResourceCandidateV1(
                    opaque_candidate_ref=self._codec.encode_msgraph_teams_chat_candidate(
                        tenant_id=tenant_id,
                        workspace_id=workspace_id,
                        connection_ref=connection_ref,
                        mailbox_user_id=chat.mailbox_user_id,
                        chat_remote_id=chat.remote_id,
                        safe_display_label=label,
                    ),
                    resource_type=self.resource_type,
                    safe_display_label=label,
                    remote_resource_id=chat.remote_id,
                    safe_description="Microsoft Graph Teams Chat",
                )
            )
        return RemoteResourceStrategyPage(
            items=tuple(items),
            provider_cursor=(
                page.continuation.url if page.continuation is not None else None
            ),
        )

    async def revalidate_candidate_label(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        connection_ref: str,
        opaque_candidate_ref: str,
        limits: ConnectedSourceRevalidationLimits,
    ) -> str:
        payload = self._codec.decode_msgraph_teams_chat_candidate(opaque_candidate_ref)
        if (
            payload.tenant_id != tenant_id
            or payload.workspace_id != workspace_id
            or payload.connection_ref != connection_ref
        ):
            raise ConnectedSourceDiscoveryError("candidate_inaccessible")
        integration = self._resolve_integration(
            tenant_id=tenant_id,
            connection_ref=connection_ref,
        )
        page = await asyncio.to_thread(
            integration.read_teams_chats_page,
            mailbox_user_id=payload.mailbox_user_id,
            continuation=None,
            limit=min(limits.page_size, 50),
        )
        for chat in page.items:
            if chat.remote_id == payload.chat_remote_id:
                return chat.topic or payload.safe_display_label
        raise ConnectedSourceDiscoveryError("candidate_inaccessible")

    def _resolve_integration(
        self,
        *,
        tenant_id: str,
        connection_ref: str,
    ) -> Ms365GraphCollaborationSuiteIntegration:
        try:
            integration = self._connection_registry.resolve(
                tenant_id=tenant_id,
                connection_ref=connection_ref,
                provider_id=MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
                integration_kind=IntegrationCategory.COLLABORATION_SUITE,
            )
        except (VendorKnowledgeError, ValueError) as exc:
            raise ConnectedSourceDiscoveryError("connection_unavailable") from exc
        if not isinstance(integration, Ms365GraphCollaborationSuiteIntegration):
            raise ConnectedSourceDiscoveryError("connection_incompatible")
        return integration


class MsGraphMailFolderDiscoveryStrategy:
    resource_type = RemoteResourceTypeV1.MSGRAPH_MAIL_FOLDER

    def __init__(
        self,
        *,
        connection_registry: KnowledgeConnectionRegistry,
        opaque_ref_codec: RemoteResourceOpaqueRefCodec,
        mailbox_user_id: str | None,
    ) -> None:
        self._connection_registry = connection_registry
        self._codec = opaque_ref_codec
        self._mailbox_user_id = mailbox_user_id.strip() if mailbox_user_id else None

    async def list_remote_resources(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        connection_ref: str,
        provider_cursor: str | None,
        limit: int,
    ) -> RemoteResourceStrategyPage:
        if self._mailbox_user_id is None:
            raise ConnectedSourceDiscoveryError("connection_unavailable")
        integration = self._resolve_integration(
            tenant_id=tenant_id,
            connection_ref=connection_ref,
        )
        continuation = (
            MsGraphKnowledgeContinuation(
                kind=MsGraphKnowledgeContinuationKind.NEXT_PAGE,
                url=provider_cursor,
            )
            if provider_cursor is not None
            else None
        )
        page = await asyncio.to_thread(
            integration.read_mail_folders_page,
            mailbox_user_id=self._mailbox_user_id,
            parent_folder_id=None,
            continuation=continuation,
            limit=min(limit, 100),
        )
        items = tuple(
            RemoteResourceCandidateV1(
                opaque_candidate_ref=self._codec.encode_msgraph_mail_folder_candidate(
                    tenant_id=tenant_id,
                    workspace_id=workspace_id,
                    connection_ref=connection_ref,
                    mailbox_user_id=folder.mailbox_user_id,
                    folder_id=folder.remote_id,
                    safe_display_label=folder.display_name,
                ),
                resource_type=self.resource_type,
                safe_display_label=folder.display_name,
                remote_resource_id=folder.remote_id,
                safe_description="Microsoft Graph Mail folder",
            )
            for folder in page.items
        )
        return RemoteResourceStrategyPage(
            items=items,
            provider_cursor=(
                page.continuation.url if page.continuation is not None else None
            ),
        )

    async def revalidate_candidate_label(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        connection_ref: str,
        opaque_candidate_ref: str,
        limits: ConnectedSourceRevalidationLimits,
    ) -> str:
        payload = self._codec.decode_msgraph_mail_folder_candidate(opaque_candidate_ref)
        if (
            payload.tenant_id != tenant_id
            or payload.workspace_id != workspace_id
            or payload.connection_ref != connection_ref
        ):
            raise ConnectedSourceDiscoveryError("candidate_inaccessible")
        integration = self._resolve_integration(
            tenant_id=tenant_id,
            connection_ref=connection_ref,
        )
        page = await asyncio.to_thread(
            integration.read_mail_folders_page,
            mailbox_user_id=payload.mailbox_user_id,
            parent_folder_id=None,
            continuation=None,
            limit=min(limits.page_size, 100),
        )
        for folder in page.items:
            if folder.remote_id == payload.folder_id:
                return folder.display_name
        raise ConnectedSourceDiscoveryError("candidate_inaccessible")

    def _resolve_integration(
        self,
        *,
        tenant_id: str,
        connection_ref: str,
    ) -> Ms365GraphCollaborationSuiteIntegration:
        try:
            integration = self._connection_registry.resolve(
                tenant_id=tenant_id,
                connection_ref=connection_ref,
                provider_id=MS365_GRAPH_COLLABORATION_SUITE_PROVIDER_ID,
                integration_kind=IntegrationCategory.COLLABORATION_SUITE,
            )
        except (VendorKnowledgeError, ValueError) as exc:
            raise ConnectedSourceDiscoveryError("connection_unavailable") from exc
        if not isinstance(integration, Ms365GraphCollaborationSuiteIntegration):
            raise ConnectedSourceDiscoveryError("connection_incompatible")
        return integration
