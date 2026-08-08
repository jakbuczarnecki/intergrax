# © Artur Czarnecki. All rights reserved.

"""Tenant knowledge source binding helpers for connected workspace sources."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
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
    SlackConversationKindV1,
)
from local_workspace_application.workspaces.connected_source_opaque_ref_codec import (
    RemoteResourceOpaqueRefCodec,
)

from intergrax.integrations.contracts.base import IntegrationCategory
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
            conversation_id=payload.conversation_id,
            conversation_kind=payload.conversation_kind,
        )
