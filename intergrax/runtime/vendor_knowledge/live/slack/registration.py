"""Slack live capability registration."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.providers.conversation_channel.slack.integration import (
    SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
)
from intergrax.integrations.providers.conversation_channel.slack.knowledge_read.common import (
    SLACK_CONVERSATION_SOURCE_KIND,
)
from intergrax.runtime.vendor_knowledge.live import LiveCapabilityExecutionResultV1
from intergrax.runtime.vendor_knowledge.live.registration import (
    LiveRegistrationBundleV1,
)
from intergrax.runtime.vendor_knowledge.live.schemas import (
    SchemaRegistrationV1,
    SchemaRoleV1,
)
from intergrax.runtime.vendor_knowledge.plugin import (
    VendorKnowledgeMode,
    VendorKnowledgeModeCapability,
    VendorKnowledgeSourceIdentity,
    VendorKnowledgeSourcePlugin,
)

from .conversation import (
    SLACK_CONVERSATION_LIST_REQUEST_SCHEMA_REF,
    SLACK_CONVERSATION_LIST_RESULT_SCHEMA_REF,
    SLACK_CONVERSATION_READ_REQUEST_SCHEMA_REF,
    SLACK_CONVERSATION_READ_RESULT_SCHEMA_REF,
    SLACK_CONVERSATION_THREAD_READ_REQUEST_SCHEMA_REF,
    SLACK_CONVERSATION_THREAD_READ_RESULT_SCHEMA_REF,
    SlackConversationListLiveHandlerV1,
    SlackConversationListLiveRequestV1,
    SlackConversationReadLiveHandlerV1,
    SlackConversationReadLiveRequestV1,
    SlackConversationThreadReadLiveHandlerV1,
    SlackConversationThreadReadLiveRequestV1,
    build_slack_conversation_list_descriptor,
    build_slack_conversation_read_descriptor,
    build_slack_conversation_thread_read_descriptor,
)


def _bundle(
    *,
    descriptor,
    handler,
    request_schema_ref: str,
    request_model,
    result_schema_ref: str,
) -> LiveRegistrationBundleV1:
    return LiveRegistrationBundleV1(
        descriptor=descriptor,
        handler=handler,
        request_schema=SchemaRegistrationV1(
            schema_ref=request_schema_ref,
            role=SchemaRoleV1.REQUEST,
            model=request_model,
            contract_version="1",
        ),
        result_schema=SchemaRegistrationV1(
            schema_ref=result_schema_ref,
            role=SchemaRoleV1.RESULT,
            model=LiveCapabilityExecutionResultV1,
            contract_version="1",
        ),
    )


def build_slack_live_registration_bundles() -> tuple[LiveRegistrationBundleV1, ...]:
    """Return the complete deterministic Slack live family."""

    return (
        _bundle(
            descriptor=build_slack_conversation_list_descriptor(),
            handler=SlackConversationListLiveHandlerV1(),
            request_schema_ref=SLACK_CONVERSATION_LIST_REQUEST_SCHEMA_REF,
            request_model=SlackConversationListLiveRequestV1,
            result_schema_ref=SLACK_CONVERSATION_LIST_RESULT_SCHEMA_REF,
        ),
        _bundle(
            descriptor=build_slack_conversation_thread_read_descriptor(),
            handler=SlackConversationThreadReadLiveHandlerV1(),
            request_schema_ref=SLACK_CONVERSATION_THREAD_READ_REQUEST_SCHEMA_REF,
            request_model=SlackConversationThreadReadLiveRequestV1,
            result_schema_ref=SLACK_CONVERSATION_THREAD_READ_RESULT_SCHEMA_REF,
        ),
        _bundle(
            descriptor=build_slack_conversation_read_descriptor(),
            handler=SlackConversationReadLiveHandlerV1(),
            request_schema_ref=SLACK_CONVERSATION_READ_REQUEST_SCHEMA_REF,
            request_model=SlackConversationReadLiveRequestV1,
            result_schema_ref=SLACK_CONVERSATION_READ_RESULT_SCHEMA_REF,
        ),
    )


def build_slack_vendor_knowledge_source_plugin() -> VendorKnowledgeSourcePlugin:
    """Compose the accepted Slack three-mode source declaration."""
    live_capability_refs = tuple(
        bundle.descriptor.capability_id for bundle in build_slack_live_registration_bundles()
    )
    identity = VendorKnowledgeSourceIdentity(
        provider_id=SLACK_CONVERSATION_CHANNEL_PROVIDER_ID,
        integration_category=IntegrationCategory.CONVERSATION_CHANNEL,
        source_kind=SLACK_CONVERSATION_SOURCE_KIND,
    )
    return VendorKnowledgeSourcePlugin(
        identity=identity,
        capabilities=(
            VendorKnowledgeModeCapability(
                mode=VendorKnowledgeMode.DURABLE,
                contract_version="vendor-knowledge.durable.v1",
                operations=("inventory", "snapshot", "incremental", "reconciliation", "exact_fetch"),
                runtime_ref="knowledge-adapter:slack:conversation_channel:slack_conversation",
                constraints={"application_sink": "slack_connected_source"},
            ),
            VendorKnowledgeModeCapability(
                mode=VendorKnowledgeMode.INDEXED,
                contract_version="vendor-knowledge.indexed.v1",
                operations=("eligible", "materialize", "publish", "index"),
                runtime_ref="indexed-source:slack:slack_conversation",
                constraints={"application_proof": "accepted"},
            ),
            VendorKnowledgeModeCapability(
                mode=VendorKnowledgeMode.LIVE,
                contract_version="vendor-knowledge.live.v1",
                operations=("list", "read", "thread.read"),
                runtime_ref="live-registration:slack:slack_conversation",
                capability_refs=live_capability_refs,
                constraints={"read_only": True, "bounded": True},
            ),
        ),
    )
