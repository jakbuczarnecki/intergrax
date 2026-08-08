# © Artur Czarnecki. All rights reserved.

"""Slack-owned connected-source discovery strategy."""

from __future__ import annotations

import time

from local_workspace_application.workspaces.connected_source_candidate import (
    decode_slack_conversation_candidate_ref,
    encode_slack_conversation_candidate_ref,
    validate_candidate_scope,
)
from local_workspace_application.workspaces.connected_source_models import (
    ConnectedSourceDiscoveryError,
    RemoteResourceCandidateV1,
    RemoteResourceTypeV1,
    SlackConversationKindV1,
)
from local_workspace_application.workspaces.connected_source_opaque_ref_codec import (
    RemoteResourceOpaqueRefCodec,
)
from local_workspace_application.workspaces.connected_source_discovery_strategy import (
    ConnectedSourceRevalidationLimits,
    RemoteResourceStrategyPage,
)

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.conversation_channel.slack.integration import (
    SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
    SlackConversationChannelIntegration,
)
from intergrax.runtime.vendor_knowledge.connections import KnowledgeConnectionRegistry
from intergrax.runtime.vendor_knowledge.errors import VendorKnowledgeError


class SlackRemoteResourceDiscoveryStrategy:
    resource_type = RemoteResourceTypeV1.SLACK_CONVERSATION

    def __init__(
        self,
        *,
        connection_registry: KnowledgeConnectionRegistry,
        opaque_ref_codec: RemoteResourceOpaqueRefCodec,
    ) -> None:
        self._connection_registry = connection_registry
        self._codec = opaque_ref_codec

    async def list_remote_resources(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        connection_ref: str,
        provider_cursor: str | None,
        limit: int,
    ) -> RemoteResourceStrategyPage:
        integration = self._resolve_integration(
            tenant_id=tenant_id,
            connection_ref=connection_ref,
        )
        page = await integration.list_accessible_conversations_page(
            cursor=provider_cursor,
            limit=limit,
        )
        items: list[RemoteResourceCandidateV1] = []
        for summary in page.items:
            kind = SlackConversationKindV1(summary.kind.value)
            candidate_ref = encode_slack_conversation_candidate_ref(
                codec=self._codec,
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                connection_ref=connection_ref,
                conversation_id=summary.conversation_id,
                conversation_kind=kind,
                safe_display_label=summary.safe_name,
            )
            items.append(
                RemoteResourceCandidateV1(
                    opaque_candidate_ref=candidate_ref,
                    resource_type=self.resource_type,
                    safe_display_label=summary.safe_name,
                    conversation_kind=kind,
                    is_archived=summary.is_archived,
                    is_private=summary.is_private,
                    safe_description=summary.safe_topic or summary.safe_purpose,
                )
            )
        return RemoteResourceStrategyPage(
            items=tuple(items),
            provider_cursor=page.next_cursor,
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
        payload = decode_slack_conversation_candidate_ref(self._codec, opaque_candidate_ref)
        validate_candidate_scope(
            payload,
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            connection_ref=connection_ref,
        )
        integration = self._resolve_integration(
            tenant_id=tenant_id,
            connection_ref=connection_ref,
        )
        provider_cursor: str | None = None
        seen_cursors: set[str | None] = set()
        total_candidates = 0
        started = time.monotonic()
        for _ in range(limits.max_pages):
            if time.monotonic() - started > limits.max_duration_seconds:
                raise ConnectedSourceDiscoveryError("candidate_revalidation_limit_exceeded")
            if provider_cursor in seen_cursors:
                raise ConnectedSourceDiscoveryError("candidate_revalidation_limit_exceeded")
            seen_cursors.add(provider_cursor)
            page = await integration.list_accessible_conversations_page(
                cursor=provider_cursor,
                limit=limits.page_size,
            )
            total_candidates += len(page.items)
            if total_candidates > limits.max_total_candidates:
                raise ConnectedSourceDiscoveryError("candidate_revalidation_limit_exceeded")
            for summary in page.items:
                if (
                    summary.conversation_id == payload.conversation_id
                    and summary.kind.value == payload.conversation_kind.value
                ):
                    return summary.safe_name
            next_cursor = page.next_cursor
            if next_cursor is None:
                break
            if next_cursor == provider_cursor or not str(next_cursor).strip():
                raise ConnectedSourceDiscoveryError("candidate_revalidation_limit_exceeded")
            provider_cursor = next_cursor
        else:
            raise ConnectedSourceDiscoveryError("candidate_revalidation_limit_exceeded")
        raise ConnectedSourceDiscoveryError("candidate_inaccessible")

    def _resolve_integration(
        self,
        *,
        tenant_id: str,
        connection_ref: str,
    ) -> SlackConversationChannelIntegration:
        try:
            integration = self._connection_registry.resolve(
                tenant_id=tenant_id,
                connection_ref=connection_ref,
                provider_id=SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
                integration_kind=IntegrationCategory.CONVERSATION_CHANNEL,
            )
        except (VendorKnowledgeError, ValueError) as exc:
            raise ConnectedSourceDiscoveryError("connection_unavailable") from exc
        if not isinstance(integration, SlackConversationChannelIntegration):
            raise ConnectedSourceDiscoveryError("connection_incompatible")
        return integration
