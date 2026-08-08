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
    RemoteResourceOpaqueRefCodec,
)

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.collaboration_suite.ms365_graph.knowledge_read import (
    MSGRAPH_TEAMS_CHAT_SOURCE_KIND,
    MsGraphTeamsChatMessageWindow,
)
from intergrax.integrations.providers.conversation_channel.slack.integration import (
    SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
)
from intergrax.integrations.providers.conversation_channel.slack.knowledge_read import (
    SLACK_CONVERSATION_SOURCE_KIND,
    SlackConversationKind,
)
from intergrax.runtime.vendor_knowledge.adapters.slack_conversation import (
    SLACK_CONVERSATION_SCOPE_TYPE,
    encode_slack_conversation_scope_id,
)
from intergrax.runtime.vendor_knowledge.adapters.ms365_graph_teams_chat import (
    MSGRAPH_TEAMS_CHAT_SCOPE_TYPE,
    encode_msgraph_teams_chat_scope_id,
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
        if graph is None:
            return self._slack.build_binding(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                connection_ref=connection_ref,
                opaque_candidate_ref=opaque_candidate_ref,
                root_oldest=root_oldest,
                root_latest=root_latest,
                safe_display_name=safe_display_name,
            )
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

    async def revalidate_candidate_label(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        connection_ref: str,
        opaque_candidate_ref: str,
    ) -> str:
        graph = self._graph_payload(opaque_candidate_ref)
        if graph is None:
            return await self._slack.revalidate_candidate_label(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                connection_ref=connection_ref,
                opaque_candidate_ref=opaque_candidate_ref,
            )
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
