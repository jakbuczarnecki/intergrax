# © Artur Czarnecki. All rights reserved.

"""Tenant knowledge source binding helpers for connected workspace sources."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

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
from local_workspace_application.workspaces.connected_source_ids import tenant_binding_id
from local_workspace_application.workspaces.connected_source_models import (
    ConnectedSourceBindingError,
    SlackConversationKindV1,
)


@dataclass(frozen=True, slots=True)
class SlackConversationTenantBindingRequest:
    tenant_id: str
    connection_ref: str
    conversation_id: str
    conversation_kind: SlackConversationKindV1
    safe_display_name: str
    root_oldest: str
    root_latest: str


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
        binding_service = self._binding_service_factory(request.tenant_id)
        encoded_scope = encode_slack_conversation_scope_id(
            conversation_id=request.conversation_id,
            conversation_kind=SlackConversationKind(request.conversation_kind.value),
            oldest=request.root_oldest,
            latest=request.root_latest,
        )
        binding_id = tenant_binding_id(
            tenant_id=request.tenant_id,
            connection_ref=request.connection_ref,
            provider_id=SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
            integration_kind=IntegrationCategory.CONVERSATION_CHANNEL.value,
            source_kind=SLACK_CONVERSATION_SOURCE_KIND,
            encoded_scope=encoded_scope,
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
        try:
            return binding_service.create_or_get_equivalent(binding)
        except VendorKnowledgeError as exc:
            raise ConnectedSourceBindingError("knowledge_source_binding_unavailable") from exc
